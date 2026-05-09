"""
3D Material Cutting Optimization Engine — Core Module
======================================================
설계 원칙:
  - 모든 단위(unit)는 호출자가 결정 (mm, inch 등 무관)
  - 부동소수점 누적 오차 방지를 위해 float + epsilon guard 사용
  - 불변(immutable) 값은 frozen dataclass / 가변 상태는 일반 dataclass
  - Node 트리는 절단 이력(Cut History)과 잔재(Offcut) 계층을 동시에 표현

물리적 제약 구현 위치:
  [C1] Strict Guillotine Cut   → CutAxis enum + split_node() 진입부 검증
  [C2] Kerf Deduction          → split_node() 내부 치수 계산
  [C3] Trimming Margins        → Stock.usable_dims property
  [C4] Orientation Mapping     → OrientationMode enum + Part.allowed_orientations()
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Tuple


# ══════════════════════════════════════════════════════════════════
# 1. 기본 값 객체 (Value Objects) — immutable, frozen
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Dims:
    """
    직육면체의 3차원 치수를 나타내는 불변 값 객체.

    l: X축 방향 길이(Length)
    w: Y축 방향 너비(Width)
    t: Z축 방향 두께(Thickness)
    """
    l: float   # noqa: E741
    w: float
    t: float

    def __post_init__(self) -> None:
        if self.l <= 0 or self.w <= 0 or self.t <= 0:
            raise ValueError(
                f"Dims의 모든 축 값은 양수여야 합니다. "
                f"입력값: l={self.l}, w={self.w}, t={self.t}"
            )

    @property
    def volume(self) -> float:
        return self.l * self.w * self.t

    def __repr__(self) -> str:
        return f"Dims(l={self.l}, w={self.w}, t={self.t})"


@dataclass(frozen=True)
class Point3D:
    """
    3D 공간 내 절대 좌표 — 원장(Stock) 원점 기준 오프셋.
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __repr__(self) -> str:
        return f"P({self.x}, {self.y}, {self.z})"


@dataclass(frozen=True)
class TrimmingMargins:
    """
    [C3] 축별 초기 테두리 여백.
    각 값은 해당 축의 양 끝단에서 각각 제거되는 양입니다.
    (x=10 → X축 좌우 각 10씩, 총 20 감소)
    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __post_init__(self) -> None:
        if self.x < 0 or self.y < 0 or self.z < 0:
            raise ValueError("Trimming 마진은 0 이상이어야 합니다.")


# ══════════════════════════════════════════════════════════════════
# 2. 열거형 (Enumerations)
# ══════════════════════════════════════════════════════════════════

class CutAxis(Enum):
    """
    [C1] Guillotine Cut이 수행될 축.

    X → YZ 평면으로 절단, 조각이 X 방향으로 분리
    Y → XZ 평면으로 절단, 조각이 Y 방향으로 분리
    Z → XY 평면으로 절단, 조각이 Z 방향으로 분리
    """
    X = "X"
    Y = "Y"
    Z = "Z"


class NodeState(Enum):
    """Node의 현재 생애 주기 상태."""
    FREE      = auto()   # 빈 공간 — 배치 가능
    OCCUPIED  = auto()   # Part가 배치됨 — 리프
    SPLIT     = auto()   # 절단 완료 — 반드시 child_a, child_b 존재
    DISCARDED = auto()   # Trimming/Kerf로 증발한 공간


class OptimizationGoal(Enum):
    MINIMIZE_WASTE  = auto()
    MINIMIZE_SHEETS = auto()
    MINIMIZE_CUTS   = auto()


class OrientationMode(Enum):
    """[C4] 방향 배치 모드."""
    FREE_3D        = auto()   # 6방향 자유 회전
    LOCK_Z_FREE_XY = auto()   # Z 두께 고정, XY 90도 회전만 허용
    FIXED          = auto()   # 회전 없음


# ══════════════════════════════════════════════════════════════════
# 3. 입력 데이터 클래스 (Input Data Models)
# ══════════════════════════════════════════════════════════════════

@dataclass
class Part:
    """
    잘라내야 하는 조각(부품) 정의.

    id              : 고유 식별자
    dims            : 원본 치수
    qty             : 필요 수량
    lock_z          : [C4] True → Z축 두께를 원장 Z축과 일치시켜야 함
    allow_xy_rotation: lock_z=True일 때 XY 평면 90도 회전 허용 여부
    priority        : 배치 우선순위 (높을수록 먼저 배치)
    """
    id:               str
    dims:             Dims
    qty:              int
    lock_z:           bool = True
    allow_xy_rotation: bool = True
    priority:         int  = 0

    def __post_init__(self) -> None:
        if self.qty < 1:
            raise ValueError(f"Part '{self.id}': qty는 1 이상이어야 합니다.")

    def allowed_orientations(self) -> List[Dims]:
        """
        [C4] 이 Part에 허용된 모든 배치 방향(Dims 변환 목록)을 반환합니다.

        lock_z=True:  t가 항상 Z 방향, XY 평면에서만 l↔w 교환 가능
        lock_z=False: 3D 자유 회전 — (l,w,t)의 모든 순열 허용
        """
        l, w, t = self.dims.l, self.dims.w, self.dims.t

        if self.lock_z:
            candidates = [Dims(l=l, w=w, t=t)]
            if self.allow_xy_rotation:
                candidates.append(Dims(l=w, w=l, t=t))
        else:
            perms = {
                (l, w, t), (l, t, w),
                (w, l, t), (w, t, l),
                (t, l, w), (t, w, l),
            }
            candidates = [Dims(l=p[0], w=p[1], t=p[2]) for p in perms]

        # 중복 제거 (정사각형 단면 등)
        seen: set[Tuple[float, float, float]] = set()
        result: List[Dims] = []
        for d in candidates:
            key = (d.l, d.w, d.t)
            if key not in seen:
                seen.add(key)
                result.append(d)
        return result

    @property
    def volume(self) -> float:
        return self.dims.volume

    def __repr__(self) -> str:
        return f"Part(id={self.id!r}, dims={self.dims}, qty={self.qty})"


@dataclass
class Stock:
    """
    원장(원자재 판재) 정의.

    id       : 고유 식별자
    dims     : 원본 전체 치수
    qty      : 보유 수량
    trimming : [C3] 축별 테두리 여백
    """
    id:       str
    dims:     Dims
    qty:      int
    trimming: TrimmingMargins = field(default_factory=TrimmingMargins)

    def __post_init__(self) -> None:
        if self.qty < 1:
            raise ValueError(f"Stock '{self.id}': qty는 1 이상이어야 합니다.")

    @property
    def usable_dims(self) -> Dims:
        """
        [C3] 트리밍 마진 제거 후 실제 사용 가능한 치수.
        각 축에서 trimming 값의 2배(양쪽)를 제거합니다.
        """
        ul = self.dims.l - 2 * self.trimming.x
        uw = self.dims.w - 2 * self.trimming.y
        ut = self.dims.t - 2 * self.trimming.z

        if ul <= 0 or uw <= 0 or ut <= 0:
            raise ValueError(
                f"Stock '{self.id}': Trimming이 너무 커서 유효 치수가 0 이하입니다. "
                f"원본={self.dims}, trimming={self.trimming}"
            )
        return Dims(l=ul, w=uw, t=ut)

    @property
    def usable_volume(self) -> float:
        return self.usable_dims.volume

    def __repr__(self) -> str:
        return f"Stock(id={self.id!r}, dims={self.dims}, qty={self.qty})"


@dataclass(frozen=True)
class EngineSettings:
    """
    최적화 엔진 전역 설정.

    kerf             : [C2] 절단 시 소모되는 톱날 두께
    trimming         : [C3] Stock에 적용할 기본 테두리 여백
    optimization_goal: 최적화 목표
    """
    kerf:              float             = 3.0
    trimming:          TrimmingMargins   = field(default_factory=TrimmingMargins)
    optimization_goal: OptimizationGoal  = OptimizationGoal.MINIMIZE_WASTE

    def __post_init__(self) -> None:
        if self.kerf < 0:
            raise ValueError("kerf는 0 이상이어야 합니다.")


# ══════════════════════════════════════════════════════════════════
# 4. 절단 액션 기록 (Cut Record) — immutable
# ══════════════════════════════════════════════════════════════════

@dataclass(frozen=True)
class Cut:
    """
    [C1][C2] 단일 Guillotine Cut 액션의 불변 기록.

    cut_id        : 절단 고유 ID
    axis          : 절단 축
    position      : Node 로컬 좌표 기준 절단 위치 (child_a의 해당 축 크기)
    kerf          : 이 절단에 적용된 톱날 두께
    parent_node_id: 이 절단이 수행된 Node ID
    """
    cut_id:         str
    axis:           CutAxis
    position:       float
    kerf:           float
    parent_node_id: str

    def __post_init__(self) -> None:
        if self.position <= 0:
            raise ValueError(f"Cut '{self.cut_id}': position은 양수여야 합니다.")
        if self.kerf < 0:
            raise ValueError(f"Cut '{self.cut_id}': kerf는 0 이상이어야 합니다.")

    def __repr__(self) -> str:
        return (
            f"Cut(axis={self.axis.value}, pos={self.position}, "
            f"kerf={self.kerf}, node={self.parent_node_id})"
        )


# ══════════════════════════════════════════════════════════════════
# 5. 핵심 트리 노드 (Node)
# ══════════════════════════════════════════════════════════════════

@dataclass
class Node:
    """
    Guillotine 절단 트리의 핵심 단위.

    각 Node는 3D 공간 내 직육면체 영역을 표현하며,
    트리 구조로 전체 절단 이력과 잔재(Offcut) 계층을 동시에 추적합니다.

    ┌──────────────────────────────────────────────┐
    │  Node                                        │
    │  ├─ dims         : 이 공간의 치수              │
    │  ├─ origin       : 원장 절대 좌표 원점         │
    │  ├─ state        : FREE/OCCUPIED/SPLIT/DISC.. │
    │  ├─ cut          : 이 노드를 만든 Cut 기록     │  ← 절단 이력
    │  ├─ placed_part  : 배치된 Part (있을 경우)    │
    │  ├─ child_a      : 앞쪽 자식 (position 크기)  │  ← 트리 분기
    │  └─ child_b      : 뒷쪽 자식 (잔재/Offcut)   │
    └──────────────────────────────────────────────┘

    불변 규칙:
      • child_a, child_b 모두 None  → 리프 노드
      • state == SPLIT              → 반드시 child_a, child_b 존재
      • 모든 절단 이력은 cut 필드로 루트까지 역추적 가능
    """

    node_id:          str       = field(default_factory=lambda: str(uuid.uuid4())[:8])
    dims:             Dims      = field(default_factory=lambda: Dims(1, 1, 1))
    origin:           Point3D   = field(default_factory=Point3D)
    state:            NodeState = field(default=NodeState.FREE)

    # 이 노드를 생성한 절단 기록 (루트는 None)
    cut:              Optional[Cut]  = field(default=None)

    # 배치 정보
    placed_part:      Optional[Part] = field(default=None)
    placed_part_dims: Optional[Dims] = field(default=None)  # 실제 배치된 방향

    # 자식 노드
    child_a:          Optional[Node] = field(default=None)   # 앞쪽 (position 크기)
    child_b:          Optional[Node] = field(default=None)   # 뒤쪽 (잔재/Offcut)

    # 역방향 참조 (부모)
    parent:           Optional[Node] = field(default=None, repr=False)

    depth:            int            = field(default=0)
    stock_id:         Optional[str]  = field(default=None)

    # ── 상태 조회 프로퍼티 ─────────────────────────────────────────

    @property
    def is_leaf(self) -> bool:
        return self.child_a is None and self.child_b is None

    @property
    def is_free(self) -> bool:
        """배치 가능한 빈 공간 (FREE 리프)."""
        return self.state == NodeState.FREE and self.is_leaf

    @property
    def volume(self) -> float:
        return self.dims.volume

    # ── 분석 메서드 ────────────────────────────────────────────────

    @property
    def wasted_volume(self) -> float:
        """
        이 서브트리 내 낭비된 총 부피 (재귀 계산).

        DISCARDED : 전체 부피가 낭비
        FREE 리프  : 미사용 공간 → 전체 낭비
        OCCUPIED   : 배치된 Part와의 부피 차이 (0이 이상적)
        SPLIT      : 자식에게 위임
        """
        if self.state == NodeState.DISCARDED:
            return self.volume
        if self.is_leaf and self.state == NodeState.FREE:
            return self.volume
        if self.state == NodeState.OCCUPIED:
            placed = self.placed_part_dims.volume if self.placed_part_dims else 0.0
            return self.volume - placed
        # SPLIT
        total = 0.0
        if self.child_a:
            total += self.child_a.wasted_volume
        if self.child_b:
            total += self.child_b.wasted_volume
        return total

    def collect_free_leaves(self) -> List[Node]:
        """배치 가능한 모든 FREE 리프 노드를 수집합니다."""
        if self.is_free:
            return [self]
        results: List[Node] = []
        if self.child_a:
            results.extend(self.child_a.collect_free_leaves())
        if self.child_b:
            results.extend(self.child_b.collect_free_leaves())
        return results

    def collect_cut_history(self) -> List[Cut]:
        """
        루트로부터 이 노드까지의 절단 이력을 순서대로 반환합니다.
        (부모 체인 역방향 추적 → 반전)
        """
        history: List[Cut] = []
        current: Optional[Node] = self
        while current is not None:
            if current.cut is not None:
                history.append(current.cut)
            current = current.parent
        return list(reversed(history))

    def summary(self, indent: int = 0) -> str:
        """트리 구조를 사람이 읽기 쉬운 형태로 출력합니다."""
        pad = "  " * indent
        o = self.origin
        part_str = f" → {self.placed_part.id!r}" if self.placed_part else ""
        cut_str  = f" [via {self.cut}]" if self.cut else " [root]"
        line = (
            f"{pad}[{self.state.name}] Node({self.node_id}) "
            f"{self.dims} @({o.x},{o.y},{o.z}){part_str}{cut_str}"
        )
        children = ""
        if self.child_a:
            children += "\n" + self.child_a.summary(indent + 1)
        if self.child_b:
            children += "\n" + self.child_b.summary(indent + 1)
        return line + children

    def __repr__(self) -> str:
        return (
            f"Node({self.node_id!r}, {self.dims}, "
            f"{self.state.name}, leaf={self.is_leaf})"
        )


# ══════════════════════════════════════════════════════════════════
# 6. 예외 클래스
# ══════════════════════════════════════════════════════════════════

class CuttingError(Exception):
    """절단 관련 모든 예외의 기반 클래스."""


class InvalidCutError(CuttingError):
    """물리적 제약 위반 또는 잘못된 절단 파라미터."""


# ══════════════════════════════════════════════════════════════════
# 7. 핵심 분할 함수: split_node()
# ══════════════════════════════════════════════════════════════════

def split_node(
    node: Node,
    axis: CutAxis,
    position: float,
    kerf: float,
) -> Tuple[Node, Node]:
    """
    [C1][C2] Strict Guillotine Cut — 핵심 분할 함수.

    주어진 Node를 지정된 축의 position에서 관통 절단하여,
    Kerf를 차감한 정확한 치수의 두 자식 Node를 반환합니다.

    절단 전:
      ├──────────────── total_dim ────────────────┤

    절단 후:
      ├─── position ───┤ kerf ├─── remainder ────┤
           child_a      소멸        child_b
           (앞쪽)      [C2]        (잔재/Offcut)

    Args:
        node    : 분할 대상 Node (FREE 상태 리프 노드여야 함)
        axis    : 절단 축
        position: 로컬 좌표 기준 절단 위치 (child_a의 해당 축 크기)
        kerf    : 톱날 두께 (≥ 0)

    Returns:
        (child_a, child_b) — 두 자식 노드 튜플

    Raises:
        InvalidCutError: 제약 위반 시

    물리 제약 보장:
        [C1] 리프 노드에만 적용 → 반드시 정확히 2개 직육면체 생성
        [C2] child_b.해당축 = total - position - kerf (kerf만큼 증발)
    """

    # ─── 사전 조건 검증 [C1] ───────────────────────────────────────

    if not node.is_leaf:
        raise InvalidCutError(
            f"Node '{node.node_id}'는 이미 분할된 SPLIT 상태입니다. "
            "Guillotine Cut은 리프 노드에만 적용 가능합니다."
        )

    if node.state != NodeState.FREE:
        raise InvalidCutError(
            f"Node '{node.node_id}'의 상태가 {node.state.name}입니다. "
            "절단은 FREE 상태인 노드에만 가능합니다."
        )

    if kerf < 0:
        raise InvalidCutError(f"kerf는 0 이상이어야 합니다. 입력값: {kerf}")

    # 해당 축의 전체 크기 선택
    total_dim: float = {"X": node.dims.l, "Y": node.dims.w, "Z": node.dims.t}[axis.value]

    # position 범위 검증
    if position <= 0:
        raise InvalidCutError(
            f"절단 위치 position={position}은 0보다 커야 합니다."
        )
    if position >= total_dim:
        raise InvalidCutError(
            f"절단 위치 position={position}이 해당 축 전체 크기 {total_dim} 이상입니다. "
            "완전 관통 절단이 불가능합니다."
        )

    # [C2] Kerf 차감 후 잔재 크기 검증
    remainder = total_dim - position - kerf
    if remainder <= 0:
        raise InvalidCutError(
            f"Kerf({kerf}) 차감 후 잔재 크기({remainder:.4f})가 0 이하입니다. "
            f"[total={total_dim}, position={position}, kerf={kerf}] "
            "절단 위치를 줄이거나 kerf 설정을 확인하세요."
        )

    # ─── Cut 액션 기록 생성 ────────────────────────────────────────

    cut_record = Cut(
        cut_id=str(uuid.uuid4())[:8],
        axis=axis,
        position=position,
        kerf=kerf,
        parent_node_id=node.node_id,
    )

    # ─── 자식 노드 치수 및 원점 계산 ──────────────────────────────

    ox, oy, oz     = node.origin.x, node.origin.y, node.origin.z
    l, w, t        = node.dims.l, node.dims.w, node.dims.t
    next_depth     = node.depth + 1

    if axis == CutAxis.X:
        #  child_a: X방향 [0, position)
        #  child_b: X방향 [position+kerf, total)
        dims_a   = Dims(l=position,  w=w, t=t)
        origin_a = Point3D(x=ox,                   y=oy, z=oz)

        dims_b   = Dims(l=remainder, w=w, t=t)
        origin_b = Point3D(x=ox + position + kerf,  y=oy, z=oz)

    elif axis == CutAxis.Y:
        dims_a   = Dims(l=l, w=position,  t=t)
        origin_a = Point3D(x=ox, y=oy,                    z=oz)

        dims_b   = Dims(l=l, w=remainder, t=t)
        origin_b = Point3D(x=ox, y=oy + position + kerf,   z=oz)

    else:  # CutAxis.Z
        dims_a   = Dims(l=l, w=w, t=position)
        origin_a = Point3D(x=ox, y=oy, z=oz)

        dims_b   = Dims(l=l, w=w, t=remainder)
        origin_b = Point3D(x=ox, y=oy, z=oz + position + kerf)

    # ─── 자식 노드 생성 ────────────────────────────────────────────

    child_a = Node(
        node_id  = str(uuid.uuid4())[:8],
        dims     = dims_a,
        origin   = origin_a,
        state    = NodeState.FREE,
        cut      = cut_record,   # 이 Cut으로 생성된 노드임을 기록
        parent   = node,
        depth    = next_depth,
        stock_id = node.stock_id,
    )

    child_b = Node(
        node_id  = str(uuid.uuid4())[:8],
        dims     = dims_b,
        origin   = origin_b,
        state    = NodeState.FREE,
        cut      = cut_record,   # 동일 Cut에서 생성된 형제 노드
        parent   = node,
        depth    = next_depth,
        stock_id = node.stock_id,
    )

    # ─── 부모 노드 상태 갱신 ───────────────────────────────────────

    node.state   = NodeState.SPLIT
    node.child_a = child_a
    node.child_b = child_b

    return child_a, child_b


# ══════════════════════════════════════════════════════════════════
# 8. 헬퍼: Stock → 루트 Node 초기화
# ══════════════════════════════════════════════════════════════════

def create_root_node(stock: Stock) -> Node:
    """
    [C3] Stock으로부터 트리밍 마진이 적용된 루트 Node를 생성합니다.

    루트 노드의 origin은 trimming 마진만큼 원장 원점에서 안쪽으로 이동합니다.
    이 노드가 Guillotine 탐색의 시작점이 됩니다.
    """
    usable = stock.usable_dims
    origin = Point3D(
        x=stock.trimming.x,
        y=stock.trimming.y,
        z=stock.trimming.z,
    )
    return Node(
        node_id  = str(uuid.uuid4())[:8],
        dims     = usable,
        origin   = origin,
        state    = NodeState.FREE,
        cut      = None,    # 루트는 절단으로 생성된 것이 아님
        parent   = None,
        depth    = 0,
        stock_id = stock.id,
    )