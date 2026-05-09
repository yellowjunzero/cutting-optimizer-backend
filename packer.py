"""
packer.py — Heuristic 3D Guillotine Packer
==========================================
core.py의 Node, Cut, split_node()를 그대로 사용합니다.

탐색 흐름:
  1. Stock → create_root_node() → FREE 노드 풀(heap)에 등록
  2. heap에서 노드 추출 → 모든 잔여 Part × 모든 허용 방향 조합 평가
  3. Best-Fit 점수로 (node, part, orientation, cut_order) 선택
  4. cut_order(3축 순서)는 Max-Offcut 기준으로 결정
  5. split_node() × 최대 3회 → 정확히 part 크기의 리프 생성 → OCCUPIED
  6. 생성된 자식 FREE 노드를 heap에 재등록
  7. 모든 Part 수량 소진 또는 배치 불가 시 종료
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass, field
from itertools import permutations
from typing import List, Optional, Tuple

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
# 1. 내부 타입 정의
# ══════════════════════════════════════════════════════════════════

# 3축 절단 순서: CutAxis 3개의 순열
CutOrder = Tuple[CutAxis, CutAxis, CutAxis]

# 6가지 가능한 절단 순서 (X→Y→Z, X→Z→Y, ...)
ALL_CUT_ORDERS: List[CutOrder] = list(permutations([CutAxis.X, CutAxis.Y, CutAxis.Z]))  # type: ignore[arg-type]


@dataclass
class PlacementCandidate:
    """
    (node, part, orientation, cut_order)의 하나의 배치 후보.

    score      : Best-Fit 점수 — 낮을수록 공간 낭비가 적음
                 = node.volume - part_dims.volume
    max_offcut : 이 절단 순서에서 생기는 가장 큰 잔재 부피
                 (동점 시 tie-break에 사용)
    """
    score: float          # Best-Fit: 작을수록 좋음
    max_offcut: float     # Max-Offcut: 클수록 좋음 (tie-break)
    node: Node
    part: Part
    part_dims: Dims       # 실제 배치될 방향의 치수
    cut_order: CutOrder

    def __lt__(self, other: PlacementCandidate) -> bool:
        # 1차: score 오름차순, 2차: max_offcut 내림차순
        if self.score != other.score:
            return self.score < other.score
        return self.max_offcut > other.max_offcut


# ══════════════════════════════════════════════════════════════════
# 2. FREE Node 우선순위 큐 (Min-Heap)
# ══════════════════════════════════════════════════════════════════

class NodeHeap:
    """
    FREE Node를 부피 내림차순으로 관리하는 우선순위 큐.

    큰 노드를 먼저 처리해야 큰 부품을 앞쪽에 배치하고
    잔재를 작게 유지하는 효과를 얻습니다.

    heapq는 min-heap이므로 부피에 음수를 취합니다.
    """

    def __init__(self) -> None:
        self._heap: list = []
        # 이미 SPLIT/OCCUPIED된 stale 노드 ID를 추적해 지연 삭제
        self._removed: set[str] = set()

    def push(self, node: Node) -> None:
        if node.state == NodeState.FREE:
            # (-volume, node_id, node) — node_id로 동점 시 안정 정렬
            heapq.heappush(self._heap, (-node.volume, node.node_id, node))

    def pop(self) -> Optional[Node]:
        """FREE 상태인 노드가 나올 때까지 stale 항목을 제거합니다."""
        while self._heap:
            _, nid, node = heapq.heappop(self._heap)
            if nid in self._removed:
                continue
            if node.state != NodeState.FREE:
                # 상태가 바뀐 노드는 버림
                continue
            return node
        return None

    def invalidate(self, node_id: str) -> None:
        """특정 노드 ID를 stale로 표시합니다 (지연 삭제)."""
        self._removed.add(node_id)

    def __len__(self) -> int:
        return sum(
            1 for _, nid, n in self._heap
            if nid not in self._removed and n.state == NodeState.FREE
        )


# ══════════════════════════════════════════════════════════════════
# 3. 절단 순서 평가 — Max Offcut
# ══════════════════════════════════════════════════════════════════

def _offcut_volumes_for_order(
    node: Node,
    part_dims: Dims,
    cut_order: CutOrder,
    kerf: float,
) -> Optional[Tuple[float, float, float]]:
    """
    지정된 절단 순서로 3회 split_node()를 *시뮬레이션*하여
    생성되는 3개 잔재(child_b) 부피를 반환합니다.

    실제로 노드를 변경하지 않고 치수만 계산합니다.

    Returns:
        (vol_b1, vol_b2, vol_b3) — 각 단계의 child_b 부피
        None — 이 절단 순서로는 배치 자체가 불가능한 경우
               (kerf 차감 후 잔재 크기 ≤ 0)
    """
    # 현재 작업 중인 서브노드의 치수를 단계적으로 추적
    cur_l, cur_w, cur_t = node.dims.l, node.dims.w, node.dims.t

    # 축 → (현재 크기, 부품 크기) 매핑을 동적으로 계산
    dim_map = {
        CutAxis.X: (cur_l, part_dims.l),
        CutAxis.Y: (cur_w, part_dims.w),
        CutAxis.Z: (cur_t, part_dims.t),
    }

    offcut_vols: List[float] = []

    # 각 절단 후 남은 치수를 추적하기 위한 상태
    remaining: dict[CutAxis, float] = {
        CutAxis.X: cur_l,
        CutAxis.Y: cur_w,
        CutAxis.Z: cur_t,
    }
    # 이미 절단된 축에서의 child_a 크기 (이후 단계에서의 노드 크기)
    locked: dict[CutAxis, float] = {}

    for step, axis in enumerate(cut_order):
        total = remaining[axis]
        pos = {CutAxis.X: part_dims.l, CutAxis.Y: part_dims.w, CutAxis.Z: part_dims.t}[axis]

        # [버그 수정] 조각 크기가 현재 공간 크기와 완벽히 일치하면 자를 필요 없음 (Kerf 무시)
        if abs(total - pos) < 1e-6:
            offcut_vols.append(0.0)
            continue

        # 배치 가능 여부 검사 (>= 가 아니라 > 로 변경)
        if pos <= 0 or pos > total:
            return None
        remainder = total - pos - kerf
        if remainder <= 0:
            return None

        # child_b의 부피 계산:
        # 이 단계의 child_b는 (remainder × 이전 단계까지 고정된 치수들)
        # child_b는 절단된 축만 remainder로 바뀌고, 나머지 축은 현재 전체 크기를 유지
        b_dims = {
            CutAxis.X: remaining[CutAxis.X],
            CutAxis.Y: remaining[CutAxis.Y],
            CutAxis.Z: remaining[CutAxis.Z],
        }
        b_dims[axis] = remainder
        offcut_vols.append(b_dims[CutAxis.X] * b_dims[CutAxis.Y] * b_dims[CutAxis.Z])

        # 다음 단계를 위해 해당 축을 part 크기로 고정
        remaining[axis] = pos

    if len(offcut_vols) < 3:
        return None

    return (offcut_vols[0], offcut_vols[1], offcut_vols[2])


def _best_cut_order(
    node: Node,
    part_dims: Dims,
    kerf: float,
) -> Optional[Tuple[CutOrder, float]]:
    """
    6가지 절단 순서 중 '가장 큰 단일 잔재'를 생성하는 순서를 반환합니다.

    Max-Offcut 전략:
      각 순서에서 생기는 3개 child_b 중 max(vol)를 구하고,
      그 max가 가장 큰 순서를 선택합니다.
      → 가장 재활용 가능성이 높은 큰 잔재 하나를 보존하는 전략

    Returns:
        (best_order, max_offcut_volume)
        None — 모든 순서에서 배치 불가
    """
    best_order: Optional[CutOrder] = None
    best_max_vol: float = -1.0

    for order in ALL_CUT_ORDERS:
        result = _offcut_volumes_for_order(node, part_dims, order, kerf)
        if result is None:
            continue
        max_vol = max(result)
        if max_vol > best_max_vol:
            best_max_vol = max_vol
            best_order = order

    if best_order is None:
        return None
    return (best_order, best_max_vol)


# ══════════════════════════════════════════════════════════════════
# 4. 단일 노드에 부품 배치 — 3회 split_node() 실행
# ══════════════════════════════════════════════════════════════════

def _place_part_on_node(
    node: Node,
    part: Part,
    part_dims: Dims,
    cut_order: CutOrder,
    kerf: float,
) -> Node:
    """
    선택된 (node, part_dims, cut_order)에 대해 split_node()를 최대 3회 호출하여
    정확히 part_dims 크기의 OCCUPIED 노드를 생성합니다.

    절단 과정 (예: X→Y→Z):

      node (L, W, T)
        ├─ split X @ l  →  child_a (l, W, T)   ← 계속 분할
        │                   child_b (L-l-k, W, T) ← FREE 잔재 ①
        │
      child_a (l, W, T)
        ├─ split Y @ w  →  child_a_a (l, w, T)  ← 계속 분할
        │                   child_a_b (l, W-w-k, T) ← FREE 잔재 ②
        │
      child_a_a (l, w, T)
        ├─ split Z @ t  →  child_a_a_a (l, w, t) ← OCCUPIED ✓
        │                   child_a_a_b (l, w, T-t-k) ← FREE 잔재 ③

    각 단계에서 불필요한 절단을 건너뜁니다:
      - 해당 축에서 node 크기 == part 크기(± epsilon)이면 split 생략

    Returns:
        OCCUPIED 상태가 된 최종 노드
    """
    EPSILON = 1e-6

    # 현재 작업 노드 (단계마다 child_a로 이동)
    current = node

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

    for axis in cut_order:
        node_dim = axis_to_node_dim[axis](current)
        part_dim = axis_to_part_dim[axis]

        # 해당 축에서 이미 딱 맞으면 절단 불필요
        if abs(node_dim - part_dim) < EPSILON:
            continue

        # split_node() 호출 → child_a (part_dim 크기) + child_b (잔재)
        child_a, _child_b = split_node(current, axis, part_dim, kerf)

        # child_a를 따라 내려감
        current = child_a

    # 최종 노드를 OCCUPIED로 마킹
    current.state = NodeState.OCCUPIED
    current.placed_part = part
    current.placed_part_dims = part_dims

    return current


# ══════════════════════════════════════════════════════════════════
# 5. Best-Fit 후보 평가 — 공간 중심 탐색의 핵심
# ══════════════════════════════════════════════════════════════════

def _find_best_candidate(
    node: Node,
    remaining_parts: dict[str, int],
    parts_by_id: dict[str, Part],
    kerf: float,
) -> Optional[PlacementCandidate]:
    """
    하나의 FREE 노드에 배치할 수 있는 모든 (part × orientation) 조합을 평가하고
    Best-Fit 점수가 가장 좋은 PlacementCandidate를 반환합니다.

    [핵심: Dynamic Mixed Nesting 구현부]
    - 특정 Part에 고정되지 않고 remaining_parts에 있는 모든 Part를 후보로 올림
    - 각 Part의 모든 허용 방향(allowed_orientations)도 평가
    - Best-Fit score = node.volume - part_dims.volume (작을수록 낭비 ↓)

    Returns:
        최적 PlacementCandidate, 배치 불가 시 None
    """
    best: Optional[PlacementCandidate] = None

    for part_id, qty in remaining_parts.items():
        if qty <= 0:
            continue

        part = parts_by_id[part_id]

        for orientation in part.allowed_orientations():
            # 이 방향으로 현재 노드에 들어갈 수 있는지 기본 크기 체크
            if (
                orientation.l > node.dims.l + 1e-6
                or orientation.w > node.dims.w + 1e-6
                or orientation.t > node.dims.t + 1e-6
            ):
                continue

            # Max-Offcut 기준으로 최적 절단 순서 선택
            order_result = _best_cut_order(node, orientation, kerf)
            if order_result is None:
                continue

            best_order, max_offcut = order_result

            # Best-Fit 점수 계산
            score = node.volume - orientation.volume

            candidate = PlacementCandidate(
                score=score,
                max_offcut=max_offcut,
                node=node,
                part=part,
                part_dims=orientation,
                cut_order=best_order,
            )

            if best is None or candidate < best:
                best = candidate

    return best


# ══════════════════════════════════════════════════════════════════
# 6. 단일 Stock 처리 루프
# ══════════════════════════════════════════════════════════════════

def _pack_single_stock(
    root: Node,
    remaining_parts: dict[str, int],
    parts_by_id: dict[str, Part],
    kerf: float,
) -> List[Node]:
    """
    하나의 원장(root Node)에 대해 공간 중심 탐색을 수행합니다.
    배치된 OCCUPIED 노드 목록을 반환합니다.

    remaining_parts는 호출자와 공유되므로 직접 수정합니다.
    (다음 원장 처리 시 이어서 사용)
    """
    heap = NodeHeap()
    heap.push(root)

    occupied_nodes: List[Node] = []

    while True:
        # 모든 Part 수량 소진 확인
        if all(q <= 0 for q in remaining_parts.values()):
            break

        # 가장 큰 FREE 노드 추출
        node = heap.pop()
        if node is None:
            break  # 이 원장에서 더 이상 배치할 공간 없음

        # Best-Fit으로 최적 후보 탐색 (Dynamic Mixed Nesting)
        candidate = _find_best_candidate(node, remaining_parts, parts_by_id, kerf)

        if candidate is None:
            # 이 노드에 어떤 Part도 들어가지 않음 → 폐기
            node.state = NodeState.DISCARDED
            continue

        # ── 실제 배치 수행 ───────────────────────────────────────
        try:
            occupied = _place_part_on_node(
                node=candidate.node,
                part=candidate.part,
                part_dims=candidate.part_dims,
                cut_order=candidate.cut_order,
                kerf=kerf,
            )
        except InvalidCutError:
            # 수치 오차 등으로 절단 실패 시 해당 노드를 DISCARDED 처리
            node.state = NodeState.DISCARDED
            continue

        occupied_nodes.append(occupied)

        # 수량 차감
        remaining_parts[candidate.part.id] -= 1

        # ── 생성된 FREE 자식 노드들을 heap에 등록 ────────────────
        # occupied 노드의 부모 체인을 역추적하여 새로 생긴 child_b들을 찾음
        _register_new_free_nodes(occupied, heap)

    return occupied_nodes


def _register_new_free_nodes(occupied: Node, heap: NodeHeap) -> None:
    """
    배치가 완료된 occupied 노드로부터 부모 체인을 역추적하여
    새로 생성된 FREE child_b 노드들을 heap에 등록합니다.

    split_node()가 호출될 때마다 child_b(잔재)가 생성되며,
    이 함수는 그 잔재들을 수집합니다.

    탐색 범위: occupied → parent(들) → SPLIT 상태인 노드의 child_b
    단, 이미 한 번 등록된 노드를 중복 등록하지 않도록
    `depth`를 활용해 최근 절단 체인만 추적합니다.
    """
    current: Optional[Node] = occupied
    # occupied가 생성될 때까지의 split 체인 깊이만큼 역추적
    # (최대 3번 split이므로 최대 3단계 부모까지)
    steps = 0
    max_steps = 3  # 최대 3회 split

    while current is not None and steps <= max_steps:
        parent = current.parent
        if parent is None:
            break
        if parent.state == NodeState.SPLIT:
            # parent의 child_b가 잔재 노드
            cb = parent.child_b
            if cb is not None and cb.state == NodeState.FREE:
                heap.push(cb)
        current = parent
        steps += 1


# ══════════════════════════════════════════════════════════════════
# 7. Part 정렬 전략 — 초기 탐색 효율 향상
# ══════════════════════════════════════════════════════════════════

def _sort_parts_for_packing(parts: List[Part]) -> List[Part]:
    """
    초기 배치 효율을 높이기 위한 Part 정렬.

    전략: 부피 내림차순 (큰 부품 먼저)
      - 큰 부품을 먼저 배치해야 나중에 남은 잔재 공간에 작은 부품이 들어갈 수 있음
      - 반대 순서(작은 것 먼저)는 큰 부품이 들어갈 공간을 잘게 쪼개는 최악의 패턴

    동점 시: priority 내림차순 → id 오름차순 (결정론적 정렬 보장)
    """
    return sorted(
        parts,
        key=lambda p: (-p.dims.volume, -p.priority, p.id),
    )


# ══════════════════════════════════════════════════════════════════
# 8. 메인 공개 함수
# ══════════════════════════════════════════════════════════════════

def pack_parts(
    settings: EngineSettings,
    stocks: List[Stock],
    parts: List[Part],
) -> List[Node]:
    """
    3D Guillotine Heuristic Packer — 메인 진입점.

    주어진 Stock 목록과 Part 목록을 받아 최적 배치를 수행하고
    모든 OCCUPIED 노드(배치된 Part 정보 포함)를 반환합니다.

    알고리즘 개요:
      1. Parts를 부피 내림차순 정렬
      2. Stock을 순서대로 처리
         a. 각 Stock에서 create_root_node()로 루트 노드 생성 (trimming 적용)
         b. NodeHeap 기반 공간 중심 탐색
         c. 각 FREE 노드에서 전체 Part 풀 대상 Best-Fit 평가 (Interleaved)
         d. 선택된 (part, orientation, cut_order)로 _place_part_on_node() 실행
         e. 잔재(child_b)들을 heap에 등록, 반복
      3. 모든 Part 배치 완료 또는 Stock 소진 시 종료

    Args:
        settings: kerf, trimming, optimization_goal 등 전역 설정
        stocks  : 사용 가능한 원장 목록
        parts   : 배치해야 할 부품 목록

    Returns:
        배치된 OCCUPIED 노드 목록
        (미배치 Part는 remaining_parts에서 qty > 0으로 확인 가능)
    """
    if not parts:
        return []
    if not stocks:
        return []

    # ── 초기화 ─────────────────────────────────────────────────────
    sorted_parts = _sort_parts_for_packing(parts)
    parts_by_id: dict[str, Part] = {p.id: p for p in sorted_parts}

    # 잔여 수량 추적 (Part ID → 남은 수량)
    remaining_parts: dict[str, int] = {p.id: p.qty for p in sorted_parts}

    all_occupied: List[Node] = []
    kerf = settings.kerf

    # ── Stock 순서 처리 ───────────────────────────────────────────
    for stock in stocks:
        for _copy_idx in range(stock.qty):
            if all(q <= 0 for q in remaining_parts.values()):
                break
            
            # [추가] S1-1, S1-2 처럼 합판마다 고유 번호표 생성
            unique_stock_id = f"{stock.id}-{_copy_idx + 1}"

            root = create_root_node(stock)
            root.stock_id = unique_stock_id # [수정] 고유 번호표 부착

            # 이 원장에서 배치 수행
            occupied = _pack_single_stock(
                root=root,
                remaining_parts=remaining_parts,
                parts_by_id=parts_by_id,
                kerf=kerf,
            )
            all_occupied.extend(occupied)

        if all(q <= 0 for q in remaining_parts.values()):
            break

    # ── 미배치 Part 경고 출력 (선택적) ───────────────────────────
    unplaced = {pid: qty for pid, qty in remaining_parts.items() if qty > 0}
    if unplaced:
        import warnings
        warnings.warn(
            f"[Packer] 배치되지 못한 Part가 있습니다: {unplaced}\n"
            "Stock 수량 또는 크기를 확인하세요.",
            stacklevel=2,
        )

    return all_occupied


# ══════════════════════════════════════════════════════════════════
# 9. 결과 분석 유틸리티
# ══════════════════════════════════════════════════════════════════

@dataclass
class PackingReport:
    """
    패킹 결과 요약 보고서.
    디버깅 및 최적화 피드백 루프에 사용합니다.
    """
    total_part_volume:  float
    total_stock_volume: float
    placed_count:       int
    unplaced:           dict[str, int]
    efficiency_pct:     float   # = placed_volume / used_stock_volume × 100
    occupied_nodes:     List[Node]

    def __str__(self) -> str:
        lines = [
            "─" * 50,
            f"  배치 완료: {self.placed_count}개",
            f"  미배치:    {self.unplaced}",
            f"  부품 부피: {self.total_part_volume:,.1f}",
            f"  원장 부피: {self.total_stock_volume:,.1f}",
            f"  효율:      {self.efficiency_pct:.1f}%",
            "─" * 50,
        ]
        return "\n".join(lines)


def analyze_packing(
    occupied_nodes: List[Node],
    stocks_used: List[Stock],
    remaining: dict[str, int],
) -> PackingReport:
    """OCCUPIED 노드 목록으로부터 패킹 효율 보고서를 생성합니다."""
    placed_vol = sum(
        n.placed_part_dims.volume for n in occupied_nodes
        if n.placed_part_dims is not None
    )
    stock_vol = sum(s.usable_volume * s.qty for s in stocks_used)
    efficiency = (placed_vol / stock_vol * 100) if stock_vol > 0 else 0.0

    return PackingReport(
        total_part_volume=placed_vol,
        total_stock_volume=stock_vol,
        placed_count=len(occupied_nodes),
        unplaced={pid: q for pid, q in remaining.items() if q > 0},
        efficiency_pct=efficiency,
        occupied_nodes=occupied_nodes,
    )