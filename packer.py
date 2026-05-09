"""
packer.py — Advanced 3D Guillotine Packer  (v3 — Hybrid 2D/3D + Full Z-Reclaim)
=================================================================================
core.py의 Node, Cut, split_node()를 그대로 사용합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[v3 핵심 개선 사항 — 4가지 Critical Requirements 완전 구현]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Z축 잔재의 완전한 회수 (Zero Dead-Space on Z)
   ────────────────────────────────────────────────
   Strip Packing 실행 중 두께(Z) 방향으로 남는 공간을 즉시 FREE 노드로 변환하여
   NodeHeap에 반환합니다.

   구현 위치: _execute_strip_plan() 내 _reclaim_z_slack() 호출
   - Strip 분리 직후: strip_node.T > 배치 부품 최대 T 이면 Z split
   - child_a (사용 두께) → Strip 내부 배치 계속
   - child_b (남은 두께) → 즉시 FREE로 NodeHeap.push()  ← 핵심
   - Fallback Best-Fit 루프에서도 동일 적용

2. 지능형 2D/3D 하이브리드 모드 자동 전환
   ─────────────────────────────────────────
   _detect_mode(parts) → PackingMode.FLAT_2D | SOLID_3D
   두께 표준편차 ≤ 1.0mm → FLAT_2D (Strip + 면적 가중치)
   두께 표준편차 >  1.0mm → SOLID_3D (Volume-Max 가중치)

3. 카메라 타겟 정보 제공 (App.jsx 연동)
   ──────────────────────────────────────
   pack_parts() → PackResult.stock_centers: List[StockCenter]
   각 Stock의 trimming 적용 후 유효 영역 정중앙 (cx, cy, cz)
   App.jsx OrbitControls.target으로 직접 사용

4. 구체적 에러 핸들링
   ────────────────────
   FailureReason enum + 수치 detail → PlacementFailure.to_dict()
   "[object Object]" 완전 제거
"""

from __future__ import annotations

import heapq
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from itertools import permutations
from typing import Dict, List, Optional, Tuple

from core import (
    CutAxis,
    Dims,
    EngineSettings,
    InvalidCutError,
    Node,
    NodeState,
    Part,
    Stock,
    create_root_node,
    split_node,
)

# ══════════════════════════════════════════════════════════════════
# 0. 알고리즘 파라미터 상수
# ══════════════════════════════════════════════════════════════════

STRIP_TOLERANCE: float  = 5.0     # Strip 폭 그룹화 허용 오차 (mm)
STRIP_MIN_EFF_2D: float = 0.45    # 2D 모드 Strip 최소 효율
STRIP_MIN_EFF_3D: float = 0.35    # 3D 모드 Strip 최소 효율 (낮춤 — 공간 활용 우선)
STRIP_MIN_PARTS:  int   = 2       # Strip 구성 최소 부품 수
THICKNESS_VAR_THRESH: float = 1.0 # 두께 표준편차 임계치(mm): 이하 → 2D 모드
EPSILON: float = 1e-6             # 부동소수점 비교 허용 오차


# ══════════════════════════════════════════════════════════════════
# 1. 내부 타입 & 열거형 정의
# ══════════════════════════════════════════════════════════════════

class PackingMode(Enum):
    """자동 감지된 패킹 모드."""
    FLAT_2D  = "2D"   # 판재: Strip 가중치, 면적 기반 점수
    SOLID_3D = "3D"   # 입체: Volume-Max 가중치, 부피 기반 점수


class FailureReason(Enum):
    """
    배치 실패 원인 코드.
    .value가 한국어 문자열로 프론트엔드에 직접 전달됩니다.
    "[object Object]" 방지를 위해 모든 실패를 이 enum으로 분류합니다.
    """
    DIMENSION_EXCEEDS_NODE = "치수 초과: 부품이 노드보다 큽니다"
    KERF_REMAINDER_ZERO    = "Kerf 잔재 없음: 절단 후 잔재가 0 이하입니다"
    TRIMMING_TOO_LARGE     = "트리밍 초과: 유효 원장 영역이 부품보다 작습니다"
    NO_VALID_ORIENTATION   = "배치 방향 없음: 허용된 방향 중 들어가는 것이 없습니다"
    STOCK_EXHAUSTED        = "원장 소진: 모든 원장 사용 후에도 배치 불가"


@dataclass
class PlacementFailure:
    """
    단일 배치 실패 기록.
    reason.value가 한국어 문자열이므로 JSON 직렬화 시 "[object Object]" 없음.
    """
    part_id:  str
    stock_id: str
    reason:   FailureReason
    detail:   str = ""

    def to_dict(self) -> dict:
        return {
            "part_id":  self.part_id,
            "stock_id": self.stock_id,
            "reason":   self.reason.value,  # ← 한국어 문자열, 절대 객체 아님
            "detail":   self.detail,
        }


@dataclass
class StockCenter:
    """
    원장 기하학적 정중앙 좌표 (요구사항 3: 카메라 타겟용).

    trimming 적용 후 usable 영역의 중심:
        cx = trimming_x + usable_l / 2
        cy = trimming_y + usable_w / 2
        cz = trimming_z + usable_t / 2

    App.jsx Scene 컴포넌트의 OrbitControls target으로 직접 사용합니다.
    """
    stock_id: str
    cx: float
    cy: float
    cz: float

    def to_dict(self) -> dict:
        return {"stock_id": self.stock_id, "cx": self.cx, "cy": self.cy, "cz": self.cz}


@dataclass
class PackResult:
    """
    pack_parts()의 반환값 (v3에서 List[Node]에서 변경).

    FastAPI 엔드포인트에서 이 객체를 직렬화할 때:
        failures      → [f.to_dict() for f in result.failures]
        stock_centers → [c.to_dict() for c in result.stock_centers]
    """
    occupied:      List[Node]
    failures:      List[PlacementFailure]
    stock_centers: List[StockCenter]
    mode:          PackingMode
    unplaced:      Dict[str, int] = field(default_factory=dict)


# 절단 순서: 3축 순열 6가지
CutOrder = Tuple[CutAxis, CutAxis, CutAxis]
ALL_CUT_ORDERS: List[CutOrder] = list(
    permutations([CutAxis.X, CutAxis.Y, CutAxis.Z])  # type: ignore[arg-type]
)


@dataclass
class PlacementCandidate:
    """(node, part, orientation, cut_order) 배치 후보."""
    score:      float   # 낭비 점수: 작을수록 좋음
    max_offcut: float   # 최대 잔재 크기: 클수록 좋음 (tie-break)
    node:       Node
    part:       Part
    part_dims:  Dims
    cut_order:  CutOrder

    def __lt__(self, other: PlacementCandidate) -> bool:
        if abs(self.score - other.score) > EPSILON:
            return self.score < other.score
        return self.max_offcut > other.max_offcut


@dataclass
class StripPlan:
    """하나의 Strip(띠) 배치 계획."""
    strip_width: float
    strip_axis:  CutAxis
    parts_seq:   List[Tuple[Part, Dims]]
    efficiency:  float


# ══════════════════════════════════════════════════════════════════
# 2. FREE Node 우선순위 큐 (Min-Heap, 부피 내림차순)
# ══════════════════════════════════════════════════════════════════

class NodeHeap:
    """
    FREE Node를 부피 내림차순으로 관리하는 우선순위 큐.
    heapq는 min-heap이므로 부피에 음수를 취합니다.
    stale 항목은 지연 삭제(lazy deletion)로 처리합니다.
    """

    def __init__(self) -> None:
        self._heap: list = []
        self._removed: set[str] = set()

    def push(self, node: Node) -> None:
        if node.state == NodeState.FREE:
            heapq.heappush(self._heap, (-node.volume, node.node_id, node))

    def pop(self) -> Optional[Node]:
        while self._heap:
            _, nid, node = heapq.heappop(self._heap)
            if nid in self._removed:
                continue
            if node.state != NodeState.FREE:
                continue
            return node
        return None

    def invalidate(self, node_id: str) -> None:
        self._removed.add(node_id)

    def __len__(self) -> int:
        return sum(
            1 for _, nid, n in self._heap
            if nid not in self._removed and n.state == NodeState.FREE
        )


# ══════════════════════════════════════════════════════════════════
# 3. 지능형 모드 감지 (요구사항 2)
# ══════════════════════════════════════════════════════════════════

def _detect_mode(parts: List[Part]) -> PackingMode:
    """
    부품 두께(T)의 표준편차를 분석하여 패킹 모드를 결정합니다.

    std_dev ≤ THICKNESS_VAR_THRESH → FLAT_2D (Strip 우선, 면적 점수)
    std_dev >  THICKNESS_VAR_THRESH → SOLID_3D (Volume-Max, 부피 점수)
    """
    if not parts:
        return PackingMode.FLAT_2D
    ts = [p.dims.t for p in parts]
    mean_t = sum(ts) / len(ts)
    std_dev = (sum((t - mean_t) ** 2 for t in ts) / len(ts)) ** 0.5
    return PackingMode.FLAT_2D if std_dev <= THICKNESS_VAR_THRESH else PackingMode.SOLID_3D


# ══════════════════════════════════════════════════════════════════
# 4. Z축 잔재 즉시 회수 (요구사항 1 핵심 함수)
# ══════════════════════════════════════════════════════════════════

def _reclaim_z_slack(
    node: Node,
    target_t: float,
    kerf: float,
    heap: NodeHeap,
) -> Node:
    """
    두께(Z) 방향 잔재를 즉시 FREE 노드로 분리하여 NodeHeap에 반환합니다.

    [요구사항 1 핵심 구현]
    Strip Packing 중에도 T 슬랙이 있으면 즉시 회수합니다.
    어떤 공간도 계산 외부로 누락되지 않습니다.

        node.dims.t ≈ target_t  → 분할 불필요, node 반환
        node.dims.t > target_t + kerf:
            split_node(Z, target_t)
            child_a (target_t 두께) → 반환 (배치 계속)
            child_b (Z 잔재)        → heap.push() ← 즉시 회수!
    """
    if abs(node.dims.t - target_t) < EPSILON:
        return node
    z_rem = node.dims.t - target_t - kerf
    if z_rem <= EPSILON:
        return node
    try:
        child_a, child_b = split_node(node, CutAxis.Z, target_t, kerf)
    except InvalidCutError:
        return node
    if child_b is not None and child_b.state == NodeState.FREE:
        heap.push(child_b)  # ★ Z 잔재 즉시 NodeHeap 반환
    return child_a


# ══════════════════════════════════════════════════════════════════
# 5. 절단 순서 평가 — Hybrid Best-Fit (요구사항 2)
# ══════════════════════════════════════════════════════════════════

def _offcut_metrics_for_order(
    node: Node,
    part_dims: Dims,
    cut_order: CutOrder,
    kerf: float,
) -> Optional[Tuple[float, float, float, float]]:
    """
    지정 절단 순서로 split 시뮬레이션 후 잔재 지표를 반환합니다.

    Returns:
        (max_offcut_area, sum_offcut_area, max_offcut_volume, sum_offcut_volume)
        None — 이 절단 순서로 배치 불가
    """
    remaining: dict[CutAxis, float] = {
        CutAxis.X: node.dims.l,
        CutAxis.Y: node.dims.w,
        CutAxis.Z: node.dims.t,
    }
    part_dim_map = {
        CutAxis.X: part_dims.l,
        CutAxis.Y: part_dims.w,
        CutAxis.Z: part_dims.t,
    }
    areas:   List[float] = []
    volumes: List[float] = []

    for axis in cut_order:
        total = remaining[axis]
        pos   = part_dim_map[axis]
        if abs(total - pos) < EPSILON:
            areas.append(0.0); volumes.append(0.0)
            continue
        if pos <= 0 or pos > total:
            return None
        remainder = total - pos - kerf
        if remainder <= 0:
            return None
        b = dict(remaining)
        b[axis] = remainder
        areas.append(b[CutAxis.X] * b[CutAxis.Y])
        volumes.append(b[CutAxis.X] * b[CutAxis.Y] * b[CutAxis.Z])
        remaining[axis] = pos

    if len(areas) < 3:
        return None
    return (max(areas), sum(areas), max(volumes), sum(volumes))


def _best_cut_order(
    node: Node,
    part_dims: Dims,
    kerf: float,
    mode: PackingMode,
) -> Optional[Tuple[CutOrder, float]]:
    """
    6가지 절단 순서 중 모드 기반 최적 순서를 반환합니다.

    FLAT_2D  → max XY 면적 잔재 최대화 (2차: sum 면적)
    SOLID_3D → max 부피 잔재 최대화   (2차: sum 부피)
    """
    best_order:     Optional[CutOrder] = None
    best_primary:   float = -1.0
    best_secondary: float = -1.0

    for order in ALL_CUT_ORDERS:
        result = _offcut_metrics_for_order(node, part_dims, order, kerf)
        if result is None:
            continue
        max_area, sum_area, max_vol, sum_vol = result
        primary, secondary = (
            (max_area, sum_area) if mode == PackingMode.FLAT_2D else (max_vol, sum_vol)
        )
        if primary > best_primary or (
            abs(primary - best_primary) < EPSILON and secondary > best_secondary
        ):
            best_primary = primary; best_secondary = secondary; best_order = order

    return None if best_order is None else (best_order, best_primary)


# ══════════════════════════════════════════════════════════════════
# 6. 단일 노드에 부품 배치 — split_node() 최대 3회
# ══════════════════════════════════════════════════════════════════

def _place_part_on_node(
    node: Node,
    part: Part,
    part_dims: Dims,
    cut_order: CutOrder,
    kerf: float,
) -> Node:
    """
    선택된 (node, part_dims, cut_order)에 split_node()를 최대 3회 호출하여
    part_dims 크기의 OCCUPIED 노드를 생성합니다.

    Z가 마지막 절단 축일 때 child_b (Z 잔재)가 split_node()에 의해 생성되며,
    _register_new_free_nodes()의 부모 체인 역추적으로 자동 NodeHeap 등록됩니다.
    """
    axis_to_part_dim = {
        CutAxis.X: part_dims.l,
        CutAxis.Y: part_dims.w,
        CutAxis.Z: part_dims.t,
    }
    axis_to_node_dim = {
        CutAxis.X: lambda n: n.dims.l,
        CutAxis.Y: lambda n: n.dims.w,
        CutAxis.Z: lambda n: n.dims.t,
    }
    current = node
    for axis in cut_order:
        if abs(axis_to_node_dim[axis](current) - axis_to_part_dim[axis]) < EPSILON:
            continue
        child_a, _ = split_node(current, axis, axis_to_part_dim[axis], kerf)
        current = child_a

    current.state            = NodeState.OCCUPIED
    current.placed_part      = part
    current.placed_part_dims = part_dims
    return current


# ══════════════════════════════════════════════════════════════════
# 7. Strip Packing (요구사항 1 Z-회수 완전 통합)
# ══════════════════════════════════════════════════════════════════

def _cluster_parts_by_width(
    remaining_parts: dict[str, int],
    parts_by_id: dict[str, Part],
    node: Node,
) -> List[List[Tuple[Part, Dims]]]:
    """부품을 폭(W) 기준으로 클러스터링합니다."""
    buckets: Dict[float, List[Tuple[Part, Dims]]] = {}
    for part_id, qty in remaining_parts.items():
        if qty <= 0:
            continue
        part = parts_by_id[part_id]
        for orientation in part.allowed_orientations():
            if (
                orientation.l > node.dims.l + EPSILON
                or orientation.w > node.dims.w + EPSILON
                or orientation.t > node.dims.t + EPSILON
            ):
                continue
            w = orientation.w
            matched: Optional[float] = None
            for key in buckets:
                if abs(key - w) <= STRIP_TOLERANCE:
                    matched = key; break
            if matched is None:
                buckets[w] = []; matched = w
            buckets[matched].append((part, orientation))
    clusters = list(buckets.values())
    clusters.sort(key=lambda c: -len(c))
    return clusters


def _plan_strip(
    cluster: List[Tuple[Part, Dims]],
    node: Node,
    remaining_parts: dict[str, int],
    parts_by_id: dict[str, Part],
    kerf: float,
    mode: PackingMode,
) -> Optional[StripPlan]:
    """하나의 폭 클러스터로부터 Strip 배치 계획을 수립합니다."""
    if len(cluster) < STRIP_MIN_PARTS:
        return None
    strip_w = max(d.w for _, d in cluster)
    if strip_w > node.dims.w + EPSILON:
        return None

    seen: Dict[str, Tuple[Part, Dims]] = {}
    for part, d in cluster:
        pid = part.id
        if pid not in seen or d.l > seen[pid][1].l:
            seen[pid] = (part, d)

    ordered = sorted(
        [(p, d) for pid, (p, d) in seen.items() if remaining_parts.get(pid, 0) > 0],
        key=lambda x: -x[1].l,
    )
    if not ordered:
        return None

    cursor_l, strip_l = 0.0, node.dims.l
    parts_seq: List[Tuple[Part, Dims]] = []
    total_area = 0.0

    for part, d in ordered:
        for _ in range(remaining_parts.get(part.id, 0)):
            if cursor_l + d.l > strip_l + EPSILON:
                break
            parts_seq.append((part, d))
            cursor_l += d.l + kerf
            total_area += d.l * d.w

    if not parts_seq:
        return None

    eff = total_area / (strip_l * strip_w) if strip_w > 0 else 0.0
    min_eff = STRIP_MIN_EFF_2D if mode == PackingMode.FLAT_2D else STRIP_MIN_EFF_3D
    if eff < min_eff:
        return None

    return StripPlan(strip_width=strip_w, strip_axis=CutAxis.Y,
                     parts_seq=parts_seq, efficiency=eff)


def _execute_strip_plan(
    node: Node,
    plan: StripPlan,
    remaining_parts: dict[str, int],
    kerf: float,
    mode: PackingMode,
    heap: NodeHeap,
    failures: List[PlacementFailure],
    stock_id: str,
) -> Tuple[List[Node], Optional[Node]]:
    """
    Strip 계획 실행 — Z축 잔재 즉시 회수 포함.

    Step 1: Y 방향으로 strip_width 절단
            → strip_node + residual_node
    Step 2: ★ strip_node의 Z 잔재 즉시 회수
            max_part_t 계산 → _reclaim_z_slack(strip_node, max_part_t, kerf, heap)
            child_b(Z 잔재)가 즉시 heap.push()됨
    Step 3: X 방향 순차 배치
    """
    occupied_list: List[Node] = []

    # ── Step 1: Y 방향 Strip 분리 ────────────────────────────────
    strip_w = plan.strip_width
    if abs(node.dims.w - strip_w) < EPSILON:
        strip_node: Node = node
        residual_node: Optional[Node] = None
    else:
        rem_w = node.dims.w - strip_w - kerf
        if rem_w <= EPSILON:
            strip_node = node; residual_node = None
        else:
            try:
                strip_node, residual_node = split_node(node, CutAxis.Y, strip_w, kerf)
            except InvalidCutError as exc:
                failures.append(PlacementFailure(
                    part_id="[strip]", stock_id=stock_id,
                    reason=FailureReason.KERF_REMAINDER_ZERO,
                    detail=f"Strip Y 분리 실패: {exc}",
                ))
                return [], None

    # ── Step 2: ★ Z축 잔재 즉시 회수 (요구사항 1) ────────────────
    if plan.parts_seq:
        max_part_t = max(d.t for _, d in plan.parts_seq)
        # strip_node.T > max_part_t + kerf 이면 Z 분리
        # child_b(Z 잔재)는 _reclaim_z_slack 내에서 heap.push()됨
        strip_node = _reclaim_z_slack(strip_node, max_part_t, kerf, heap)

    # ── Step 3: X 방향 순차 배치 ─────────────────────────────────
    current = strip_node
    for part, orientation in plan.parts_seq:
        if remaining_parts.get(part.id, 0) <= 0:
            continue
        if (
            orientation.l > current.dims.l + EPSILON
            or orientation.w > current.dims.w + EPSILON
            or orientation.t > current.dims.t + EPSILON
        ):
            failures.append(PlacementFailure(
                part_id=part.id, stock_id=stock_id,
                reason=FailureReason.DIMENSION_EXCEEDS_NODE,
                detail=(
                    f"Strip 내부 — "
                    f"부품({orientation.l:.1f}×{orientation.w:.1f}×{orientation.t:.1f})"
                    f" > 노드({current.dims.l:.1f}×{current.dims.w:.1f}×{current.dims.t:.1f})"
                ),
            ))
            continue

        order_result = _best_cut_order(current, orientation, kerf, mode)
        if order_result is None:
            failures.append(PlacementFailure(
                part_id=part.id, stock_id=stock_id,
                reason=FailureReason.KERF_REMAINDER_ZERO,
                detail=(
                    f"Strip 절단 순서 없음 — "
                    f"부품({orientation.l:.1f}×{orientation.w:.1f}×{orientation.t:.1f}), "
                    f"kerf={kerf:.1f}"
                ),
            ))
            continue

        best_order, _ = order_result
        try:
            occupied = _place_part_on_node(
                node=current, part=part, part_dims=orientation,
                cut_order=best_order, kerf=kerf,
            )
        except InvalidCutError as exc:
            failures.append(PlacementFailure(
                part_id=part.id, stock_id=stock_id,
                reason=FailureReason.KERF_REMAINDER_ZERO,
                detail=f"split_node 오류: {exc}",
            ))
            continue

        occupied_list.append(occupied)
        remaining_parts[part.id] -= 1

        nxt = _find_strip_remainder(occupied, strip_node)
        if nxt is None:
            break
        current = nxt

    return occupied_list, residual_node


def _find_strip_remainder(occupied: Node, strip_root: Node) -> Optional[Node]:
    """Strip 배치 후 X 방향 FREE 잔재 노드를 반환합니다."""
    cur: Optional[Node] = occupied
    steps = 0
    while cur is not None and steps <= 3:
        parent = cur.parent
        if parent is None:
            break
        cb = parent.child_b
        if cb is not None and cb.state == NodeState.FREE:
            return cb
        cur = parent; steps += 1
    return None


# ══════════════════════════════════════════════════════════════════
# 8. Best-Fit 후보 평가 (요구사항 2 점수 + 요구사항 4 에러)
# ══════════════════════════════════════════════════════════════════

def _find_best_candidate(
    node: Node,
    remaining_parts: dict[str, int],
    parts_by_id: dict[str, Part],
    kerf: float,
    mode: PackingMode,
    failures: List[PlacementFailure],
    stock_id: str,
) -> Optional[PlacementCandidate]:
    """
    하나의 FREE 노드에 대한 최선의 PlacementCandidate를 반환합니다.

    [요구사항 2] 모드별 점수:
        FLAT_2D  → score = node.XY_area  - part.XY_area
        SOLID_3D → score = node.volume   - part.volume

    [요구사항 4] 실패 원인 구체화:
        치수 초과 → 어느 축(L/W/T)이 얼마나 초과했는지 수치 명시
        Kerf 잔재 없음 → 부품·노드 치수와 kerf 값 함께 기록
    """
    best: Optional[PlacementCandidate] = None
    node_xy  = node.dims.l * node.dims.w
    node_vol = node.dims.l * node.dims.w * node.dims.t

    for part_id, qty in remaining_parts.items():
        if qty <= 0:
            continue
        part = parts_by_id[part_id]
        first_dim_fail_recorded = False

        for orientation in part.allowed_orientations():
            over = []
            if orientation.l > node.dims.l + EPSILON:
                over.append(f"L({orientation.l:.1f}>{node.dims.l:.1f})")
            if orientation.w > node.dims.w + EPSILON:
                over.append(f"W({orientation.w:.1f}>{node.dims.w:.1f})")
            if orientation.t > node.dims.t + EPSILON:
                over.append(f"T({orientation.t:.1f}>{node.dims.t:.1f})")

            if over:
                if not first_dim_fail_recorded:
                    failures.append(PlacementFailure(
                        part_id=part_id, stock_id=stock_id,
                        reason=FailureReason.DIMENSION_EXCEEDS_NODE,
                        detail="초과 축: " + ", ".join(over),
                    ))
                    first_dim_fail_recorded = True
                continue

            order_result = _best_cut_order(node, orientation, kerf, mode)
            if order_result is None:
                failures.append(PlacementFailure(
                    part_id=part_id, stock_id=stock_id,
                    reason=FailureReason.KERF_REMAINDER_ZERO,
                    detail=(
                        f"부품({orientation.l:.1f}×{orientation.w:.1f}×{orientation.t:.1f}), "
                        f"노드({node.dims.l:.1f}×{node.dims.w:.1f}×{node.dims.t:.1f}), "
                        f"kerf={kerf:.1f}"
                    ),
                ))
                continue

            best_order, max_offcut = order_result
            if mode == PackingMode.FLAT_2D:
                score = node_xy  - orientation.l * orientation.w
            else:
                score = node_vol - orientation.l * orientation.w * orientation.t

            cand = PlacementCandidate(
                score=score, max_offcut=max_offcut,
                node=node, part=part, part_dims=orientation, cut_order=best_order,
            )
            if best is None or cand < best:
                best = cand

    return best


# ══════════════════════════════════════════════════════════════════
# 9. 자유 노드 수집 유틸리티
# ══════════════════════════════════════════════════════════════════

def _collect_free_nodes(node: Node, heap: NodeHeap) -> None:
    """서브트리의 모든 FREE 노드를 heap에 등록합니다 (Z 회수 노드 포함)."""
    stack: list[Node] = [node]
    visited: set[str] = set()
    while stack:
        cur = stack.pop()
        if cur.node_id in visited:
            continue
        visited.add(cur.node_id)
        if cur.state == NodeState.FREE:
            heap.push(cur)
        elif cur.state == NodeState.SPLIT:
            if cur.child_a: stack.append(cur.child_a)
            if cur.child_b: stack.append(cur.child_b)


def _register_new_free_nodes(occupied: Node, heap: NodeHeap) -> None:
    """
    occupied 노드의 부모 체인을 역추적하여 FREE child_b들을 heap에 등록합니다.
    Z 방향 child_b(FREE 잔재 ③)도 여기서 자동 등록됩니다.
    """
    cur: Optional[Node] = occupied
    steps = 0
    while cur is not None and steps <= 3:
        parent = cur.parent
        if parent is None:
            break
        if parent.state == NodeState.SPLIT:
            cb = parent.child_b
            if cb is not None and cb.state == NodeState.FREE:
                heap.push(cb)
        cur = parent; steps += 1


# ══════════════════════════════════════════════════════════════════
# 10. 단일 Stock 처리 루프 (2-Phase Hybrid)
# ══════════════════════════════════════════════════════════════════

def _pack_single_stock(
    root: Node,
    remaining_parts: dict[str, int],
    parts_by_id: dict[str, Part],
    kerf: float,
    mode: PackingMode,
    failures: List[PlacementFailure],
    stock_id: str,
) -> List[Node]:
    """
    하나의 원장에 대해 2-Phase Hybrid Packing을 수행합니다.

    Phase 1 — Strip Packing + Z 잔재 즉시 회수:
      클러스터링 → 계획 → 실행(_execute_strip_plan)
      _execute_strip_plan 내에서 _reclaim_z_slack이 Z child_b를 heap.push()

    Phase 2 — Fallback Best-Fit (Hybrid 점수):
      heap 기반 탐색, 모드별 점수, Z 잔재 추가 회수
    """
    heap = NodeHeap()
    heap.push(root)
    occupied: List[Node] = []

    # ── Phase 1: Strip Packing ────────────────────────────────────
    if root.state == NodeState.FREE:
        clusters = _cluster_parts_by_width(remaining_parts, parts_by_id, root)
        for cluster in clusters:
            if all(q <= 0 for q in remaining_parts.values()):
                break
            if root.state != NodeState.FREE:
                break
            plan = _plan_strip(cluster, root, remaining_parts, parts_by_id, kerf, mode)
            if plan is None:
                continue
            strip_occ, residual = _execute_strip_plan(
                node=root, plan=plan, remaining_parts=remaining_parts,
                kerf=kerf, mode=mode, heap=heap,
                failures=failures, stock_id=stock_id,
            )
            occupied.extend(strip_occ)
            if residual is not None and residual.state == NodeState.FREE:
                heap.push(residual)
            _collect_free_nodes(root, heap)  # Z 회수 노드 포함 전체 수집
            break  # 첫 Strip 성공 → Phase 2로 전환

    # ── Phase 2: Fallback Best-Fit ────────────────────────────────
    while True:
        if all(q <= 0 for q in remaining_parts.values()):
            break
        node = heap.pop()
        if node is None:
            break

        # Fallback에서도 Z 잔재 회수 (요구사항 1 완전성)
        min_t = min(
            (parts_by_id[pid].dims.t for pid, q in remaining_parts.items() if q > 0),
            default=None,
        )
        if min_t is not None and node.dims.t > min_t + kerf + EPSILON:
            node = _reclaim_z_slack(node, min_t, kerf, heap)

        cand = _find_best_candidate(
            node=node, remaining_parts=remaining_parts, parts_by_id=parts_by_id,
            kerf=kerf, mode=mode, failures=failures, stock_id=stock_id,
        )
        if cand is None:
            node.state = NodeState.DISCARDED
            continue

        try:
            occ = _place_part_on_node(
                node=cand.node, part=cand.part, part_dims=cand.part_dims,
                cut_order=cand.cut_order, kerf=kerf,
            )
        except InvalidCutError as exc:
            node.state = NodeState.DISCARDED
            failures.append(PlacementFailure(
                part_id=cand.part.id, stock_id=stock_id,
                reason=FailureReason.KERF_REMAINDER_ZERO,
                detail=f"_place_part_on_node: {exc}",
            ))
            continue

        occupied.append(occ)
        remaining_parts[cand.part.id] -= 1
        _register_new_free_nodes(occ, heap)

    return occupied


# ══════════════════════════════════════════════════════════════════
# 11. 원장 중심 좌표 계산 (요구사항 3)
# ══════════════════════════════════════════════════════════════════

def _compute_stock_center(stock: Stock, trimming: object = None) -> StockCenter:
    """
    원장 기하학적 정중앙 좌표를 계산합니다 (요구사항 3).

    trimming 적용 후 usable 영역의 중심:
        cx = trim_x + (stock.l - 2*trim_x) / 2
        cy = trim_y + (stock.w - 2*trim_y) / 2
        cz = trim_z + (stock.t - 2*trim_z) / 2

    App.jsx buildSceneData()의 stockCenter 계산과 동일한 로직으로
    OrbitControls.target과 정확히 일치합니다.
    """
    def _get(key: str) -> float:
        if trimming is None: return 0.0
        if isinstance(trimming, dict): return float(trimming.get(key, 0.0))
        return float(getattr(trimming, key, 0.0))

    tx, ty, tz = _get("x"), _get("y"), _get("z")
    ul = max(0.0, stock.dims.l - 2 * tx)
    uw = max(0.0, stock.dims.w - 2 * ty)
    ut = max(0.0, stock.dims.t - 2 * tz)

    return StockCenter(
        stock_id=stock.id,
        cx=tx + ul / 2.0,
        cy=ty + uw / 2.0,
        cz=tz + ut / 2.0,
    )


# ══════════════════════════════════════════════════════════════════
# 12. Part 정렬 전략 (모드별)
# ══════════════════════════════════════════════════════════════════

def _sort_parts(parts: List[Part], mode: PackingMode) -> List[Part]:
    """
    FLAT_2D  → XY 면적 내림차순 → 두께 내림차순 → priority → id
    SOLID_3D → 부피 내림차순   → XY 면적 내림차순 → priority → id
    """
    if mode == PackingMode.FLAT_2D:
        key = lambda p: (-(p.dims.l * p.dims.w), -p.dims.t, -p.priority, p.id)
    else:
        key = lambda p: (-p.dims.volume, -(p.dims.l * p.dims.w), -p.priority, p.id)
    return sorted(parts, key=key)


# ══════════════════════════════════════════════════════════════════
# 13. 메인 공개 함수
# ══════════════════════════════════════════════════════════════════

def pack_parts(
    settings: EngineSettings,
    stocks: List[Stock],
    parts: List[Part],
) -> PackResult:
    """
    3D Guillotine Heuristic Packer v3 — 메인 진입점.

    반환값: PackResult
        .occupied      : OCCUPIED 노드 목록
        .failures      : PlacementFailure 목록 (구체적 원인, 요구사항 4)
        .stock_centers : StockCenter 목록 (카메라 타겟, 요구사항 3)
        .mode          : PackingMode (자동 감지, 요구사항 2)
        .unplaced      : {part_id: 잔여 수량}

    FastAPI 직렬화 예시:
        result = pack_parts(settings, stocks, parts)
        response["failures"]      = [f.to_dict() for f in result.failures]
        response["stock_centers"] = [c.to_dict() for c in result.stock_centers]
        response["mode"]          = result.mode.value
    """
    if not parts:
        return PackResult(occupied=[], failures=[], stock_centers=[],
                          mode=PackingMode.FLAT_2D, unplaced={})
    if not stocks:
        return PackResult(
            occupied=[], stock_centers=[], mode=PackingMode.FLAT_2D,
            failures=[PlacementFailure(
                part_id=p.id, stock_id="N/A",
                reason=FailureReason.STOCK_EXHAUSTED,
                detail="사용 가능한 원장이 없습니다.",
            ) for p in parts],
            unplaced={p.id: p.qty for p in parts},
        )

    # 모드 자동 감지 (요구사항 2)
    mode = _detect_mode(parts)
    sorted_parts  = _sort_parts(parts, mode)
    parts_by_id   = {p.id: p for p in sorted_parts}
    remaining     = {p.id: p.qty for p in sorted_parts}
    all_occupied: List[Node]            = []
    all_failures: List[PlacementFailure] = []
    centers:      List[StockCenter]      = []
    kerf     = settings.kerf
    trimming = getattr(settings, "trimming", None)

    for stock in stocks:
        base_center = _compute_stock_center(stock, trimming)

        for copy_idx in range(stock.qty):
            if all(q <= 0 for q in remaining.values()):
                break

            uid = f"{stock.id}-{copy_idx + 1}"
            centers.append(StockCenter(
                stock_id=uid, cx=base_center.cx, cy=base_center.cy, cz=base_center.cz,
            ))

            # trimming 사전 검증 (요구사항 4)
            def _get(key: str) -> float:
                if trimming is None: return 0.0
                if isinstance(trimming, dict): return float(trimming.get(key, 0.0))
                return float(getattr(trimming, key, 0.0))
            tx, ty, tz = _get("x"), _get("y"), _get("z")
            eff_l = stock.dims.l - 2 * tx
            eff_w = stock.dims.w - 2 * ty
            eff_t = stock.dims.t - 2 * tz

            for pid, qty in remaining.items():
                if qty <= 0: continue
                p = parts_by_id[pid]
                if not any(
                    o.l <= eff_l + EPSILON and o.w <= eff_w + EPSILON and o.t <= eff_t + EPSILON
                    for o in p.allowed_orientations()
                ):
                    all_failures.append(PlacementFailure(
                        part_id=pid, stock_id=uid,
                        reason=FailureReason.TRIMMING_TOO_LARGE,
                        detail=(
                            f"유효 영역({eff_l:.1f}×{eff_w:.1f}×{eff_t:.1f})"
                            f" < 부품({p.dims.l:.1f}×{p.dims.w:.1f}×{p.dims.t:.1f}), "
                            f"trimming=({tx:.1f},{ty:.1f},{tz:.1f})"
                        ),
                    ))

            root = create_root_node(stock)
            root.stock_id = uid
            occ = _pack_single_stock(
                root=root, remaining_parts=remaining, parts_by_id=parts_by_id,
                kerf=kerf, mode=mode, failures=all_failures, stock_id=uid,
            )
            all_occupied.extend(occ)

        if all(q <= 0 for q in remaining.values()):
            break

    unplaced = {pid: qty for pid, qty in remaining.items() if qty > 0}
    for pid, qty in unplaced.items():
        all_failures.append(PlacementFailure(
            part_id=pid, stock_id="ALL",
            reason=FailureReason.STOCK_EXHAUSTED,
            detail=f"잔여 {qty}개 — 모든 원장 소진 후에도 배치 불가",
        ))

    if unplaced:
        warnings.warn(
            f"[Packer v3] 미배치 Part: {unplaced} | failures 목록에서 원인 확인",
            stacklevel=2,
        )

    return PackResult(
        occupied=all_occupied, failures=all_failures,
        stock_centers=centers, mode=mode, unplaced=unplaced,
    )


# ══════════════════════════════════════════════════════════════════
# 14. 결과 분석 유틸리티
# ══════════════════════════════════════════════════════════════════

@dataclass
class PackingReport:
    """패킹 결과 요약 보고서."""
    total_part_volume:  float
    total_stock_volume: float
    placed_count:       int
    unplaced:           Dict[str, int]
    efficiency_pct:     float
    occupied_nodes:     List[Node]
    mode:               PackingMode
    failure_summary:    Dict[str, int]  # FailureReason.value → 횟수

    def __str__(self) -> str:
        lines = [
            "─" * 56,
            f"  패킹 모드:  {self.mode.value}",
            f"  배치 완료: {self.placed_count}개",
            f"  미배치:    {self.unplaced}",
            f"  부품 부피: {self.total_part_volume:,.1f} mm³",
            f"  원장 부피: {self.total_stock_volume:,.1f} mm³",
            f"  효율:      {self.efficiency_pct:.1f}%",
        ]
        if self.failure_summary:
            lines.append("  실패 원인 분류:")
            for reason, cnt in sorted(self.failure_summary.items(), key=lambda x: -x[1]):
                lines.append(f"    [{cnt:>3}회] {reason}")
        lines.append("─" * 56)
        return "\n".join(lines)


def analyze_packing(result: PackResult, stocks_used: List[Stock]) -> PackingReport:
    """PackResult로부터 패킹 효율 보고서를 생성합니다."""
    placed_vol = sum(
        n.placed_part_dims.volume for n in result.occupied
        if n.placed_part_dims is not None
    )
    stock_vol  = sum(s.usable_volume * s.qty for s in stocks_used)
    efficiency = (placed_vol / stock_vol * 100) if stock_vol > 0 else 0.0
    fail_summary: Dict[str, int] = defaultdict(int)
    for f in result.failures:
        fail_summary[f.reason.value] += 1
    return PackingReport(
        total_part_volume=placed_vol, total_stock_volume=stock_vol,
        placed_count=len(result.occupied), unplaced=result.unplaced,
        efficiency_pct=efficiency, occupied_nodes=result.occupied,
        mode=result.mode, failure_summary=dict(fail_summary),
    )
