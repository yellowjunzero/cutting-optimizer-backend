"""
packer.py — Advanced 3D Guillotine Packer  (v2 — Strip Packing + Best-Area Fit)
================================================================================
core.py의 Node, Cut, split_node()를 그대로 사용합니다.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
[v2에서 도입된 알고리즘 개선 원리]
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Strip Packing (띠 절단 우선 탐색)
   ─────────────────────────────────
   현장 목수가 판재를 자르는 방식은 다음과 같습니다:
     "폭 400mm 부품이 여러 개 있으면 → 원장에서 폭 400mm 띠(Strip)를 먼저
      세로로 통째로 잘라내고 → 그 안에서 길이 방향으로 개별 부품을 쪼갠다."

   이 방식이 탐욕적 Best-Fit보다 효율적인 이유:
   - 절단 횟수가 줄어든다 (띠 1번 → 내부 N번, vs 개별 N×2번)
   - Kerf 손실이 집약된다 (공유 절단선)
   - 잔재가 L자가 아닌 직사각형으로 남아 재활용이 쉽다

   구현 전략:
   a) 폭(W)이 동일하거나 근사한 부품들을 "Strip 후보 그룹"으로 클러스터링
      (허용 오차: STRIP_TOLERANCE = 5mm)
   b) 각 클러스터에 대해 "총 길이 합"을 계산하여 띠로 잘랐을 때 효율 추정
   c) 효율이 높은 클러스터부터 Strip 배치를 시도
   d) Strip 배치 성공 시 해당 노드의 잔재는 큰 직사각형 형태로 남음
   e) Strip 배치 불가 노드는 기존 Best-Fit(Fallback)으로 처리

2. Best-Area Fit (잔재 면적 최대화 절단 축 선택)
   ─────────────────────────────────────────────
   기존 v1의 Max-Offcut은 3개 잔재(child_b) 중 "가장 큰 부피 하나"를 기준으로
   절단 순서를 선택했습니다.

   v2에서는 여기서 한 단계 더 나아가:
   - 3개 잔재 중 "가장 큰 것의 면적"을 우선 최대화
   - 동점 시 "나머지 잔재들의 면적 합"도 비교

   이 방식이 부피 기준보다 나은 이유:
   - 기다란 얇은 잔재(높은 부피, 낮은 면적)보다 정사각형에 가까운 잔재가
     다음 부품을 받아들이기 유리하다
   - 2D 패킹(lock_z 부품이 많은 현실)에서는 면적이 실질적 지표

3. Width-Grouping + Adaptive Strip Height
   ────────────────────────────────────────
   Strip을 구성할 때 같은 Width 그룹 내에서:
   - Strip 높이(H) = 그룹 내 최대 Width 부품에 맞춤
   - 남은 Width 공차(≤ STRIP_TOLERANCE)는 Kerf 손실과 함께 잔재로 처리
   - 이로써 Strip 내부에서도 Best-Area Fit을 재귀 적용

4. Graceful Degradation (우아한 성능 저하)
   ──────────────────────────────────────
   Strip 배치가 실패하거나 남은 부품이 Strip에 맞지 않을 때:
   - 기존 v1의 Best-Fit + Max-Offcut 로직으로 자동 낙하(Fallback)
   - 단일 부품도 빠짐없이 배치 시도

탐색 흐름 (v2):
  Phase 1 — Strip Planning
    1. 부품을 폭 기준으로 클러스터링 (STRIP_TOLERANCE 이내)
    2. 클러스터별 Strip 효율 추정 (부품 면적 합 / Strip 면적)
    3. 효율 높은 순서로 Strip 배치 시도
       a. 원장에서 Strip 폭만큼 Y축(또는 X축) 절단
       b. 그 Strip 내에서 부품을 길이 방향 순차 배치
  Phase 2 — Fallback Best-Fit
    4. 남은 FREE 공간에 나머지 부품을 v1 방식으로 배치
"""

from __future__ import annotations

import heapq
import itertools
from collections import defaultdict
from dataclasses import dataclass, field
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

# Strip 그룹화 허용 오차 (mm): 이 범위 내의 폭은 동일 Strip으로 취급
STRIP_TOLERANCE = 5.0

# Strip 배치 최소 효율 임계값: 이보다 낮으면 Strip 시도 포기 → Fallback
STRIP_MIN_EFFICIENCY = 0.45

# Strip 내 최소 부품 수: 이 수 이상일 때만 Strip 계획 수립
STRIP_MIN_PARTS = 2


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

    score      : Best-Area Fit 점수 — 낮을수록 공간 낭비가 적음
                 = node.face_area(XY) - part_dims.face_area
    max_offcut : 이 절단 순서에서 생기는 가장 큰 잔재 면적
                 (동점 시 tie-break에 사용)
    """
    score: float          # Best-Area Fit: 작을수록 좋음
    max_offcut: float     # Max-Offcut 면적: 클수록 좋음 (tie-break)
    node: Node
    part: Part
    part_dims: Dims       # 실제 배치될 방향의 치수
    cut_order: CutOrder

    def __lt__(self, other: PlacementCandidate) -> bool:
        # 1차: score 오름차순 (면적 낭비 최소화)
        # 2차: max_offcut 내림차순 (큰 잔재 보존)
        if abs(self.score - other.score) > 1e-6:
            return self.score < other.score
        return self.max_offcut > other.max_offcut


@dataclass
class StripPlan:
    """
    하나의 Strip(띠) 배치 계획.

    strip_width : Strip의 폭 (절단 축의 크기, mm)
    strip_axis  : Strip을 가르는 절단 축 (보통 CutAxis.Y — 폭 방향)
    parts_seq   : Strip 내부에 배치할 (part, orientation) 순서 목록
    efficiency  : 예상 효율 (부품 면적 합 / Strip 면적)
    """
    strip_width: float
    strip_axis: CutAxis
    parts_seq: List[Tuple[Part, Dims]]
    efficiency: float


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
# 3. 절단 순서 평가 — Best-Area Fit (v2 개선)
# ══════════════════════════════════════════════════════════════════

def _offcut_areas_for_order(
    node: Node,
    part_dims: Dims,
    cut_order: CutOrder,
    kerf: float,
) -> Optional[Tuple[float, float, float]]:
    """
    지정된 절단 순서로 3회 split_node()를 *시뮬레이션*하여
    생성되는 3개 잔재(child_b)의 XY 단면적을 반환합니다.

    [v2 변경] 부피(volume) → 면적(area) 기준으로 변경.
    면적 기준이 2D 지배적인 현실 패킹에서 더 실질적입니다.

    Returns:
        (area_b1, area_b2, area_b3)
        None — 이 절단 순서로 배치 불가
    """
    remaining: dict[CutAxis, float] = {
        CutAxis.X: node.dims.l,
        CutAxis.Y: node.dims.w,
        CutAxis.Z: node.dims.t,
    }

    offcut_areas: List[float] = []

    for axis in cut_order:
        total = remaining[axis]
        pos = {CutAxis.X: part_dims.l, CutAxis.Y: part_dims.w, CutAxis.Z: part_dims.t}[axis]

        # 딱 맞으면 절단 불필요
        if abs(total - pos) < 1e-6:
            offcut_areas.append(0.0)
            continue

        if pos <= 0 or pos > total:
            return None

        remainder = total - pos - kerf
        if remainder <= 0:
            return None

        # child_b의 면적 계산 (XY 단면 기준)
        b_dims = {
            CutAxis.X: remaining[CutAxis.X],
            CutAxis.Y: remaining[CutAxis.Y],
            CutAxis.Z: remaining[CutAxis.Z],
        }
        b_dims[axis] = remainder
        # XY 단면적: l × w
        offcut_areas.append(b_dims[CutAxis.X] * b_dims[CutAxis.Y])

        remaining[axis] = pos

    if len(offcut_areas) < 3:
        return None

    return (offcut_areas[0], offcut_areas[1], offcut_areas[2])


def _best_cut_order(
    node: Node,
    part_dims: Dims,
    kerf: float,
) -> Optional[Tuple[CutOrder, float]]:
    """
    6가지 절단 순서 중 Best-Area Fit 기준 최적 순서를 반환합니다.

    [v2 Best-Area Fit 전략]
    각 순서에서 생기는 3개 child_b 중:
      1차: max(area)가 가장 큰 순서 선택
          → 재활용 가능한 큰 직사각형 잔재 보존
      2차 (동점): sum(area)가 큰 순서
          → 전체 잔재 면적 총량 보존

    Returns:
        (best_order, max_offcut_area)
        None — 모든 순서에서 배치 불가
    """
    best_order: Optional[CutOrder] = None
    best_max_area: float = -1.0
    best_sum_area: float = -1.0

    for order in ALL_CUT_ORDERS:
        result = _offcut_areas_for_order(node, part_dims, order, kerf)
        if result is None:
            continue
        max_area = max(result)
        sum_area = sum(result)
        # 1차: max_area, 2차: sum_area
        if (max_area > best_max_area) or (
            abs(max_area - best_max_area) < 1e-6 and sum_area > best_sum_area
        ):
            best_max_area = max_area
            best_sum_area = sum_area
            best_order = order

    if best_order is None:
        return None
    return (best_order, best_max_area)


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

    Returns:
        OCCUPIED 상태가 된 최종 노드
    """
    EPSILON = 1e-6
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

        if abs(node_dim - part_dim) < EPSILON:
            continue

        child_a, _child_b = split_node(current, axis, part_dim, kerf)
        current = child_a

    current.state = NodeState.OCCUPIED
    current.placed_part = part
    current.placed_part_dims = part_dims

    return current


# ══════════════════════════════════════════════════════════════════
# 5. Strip Packing — 핵심 신규 모듈
# ══════════════════════════════════════════════════════════════════

def _cluster_parts_by_width(
    remaining_parts: dict[str, int],
    parts_by_id: dict[str, Part],
    node: Node,
    kerf: float,
) -> List[List[Tuple[Part, Dims]]]:
    """
    남은 부품들을 폭(W) 기준으로 클러스터링합니다.

    [Strip Packing Phase 1 — Grouping]
    - 각 부품의 allowed_orientations() 중 node에 들어갈 수 있는 것을 골라
      폭(W)이 STRIP_TOLERANCE 이내인 것들끼리 묶습니다.
    - 결과: [(part, orientation), ...] 의 리스트 목록
    - 각 클러스터는 공통 Strip 폭 후보를 공유합니다.
    """
    # 폭 → [(part, orientation), ...] 매핑 (근사 그룹화)
    width_buckets: Dict[float, List[Tuple[Part, Dims]]] = {}

    for part_id, qty in remaining_parts.items():
        if qty <= 0:
            continue
        part = parts_by_id[part_id]

        for orientation in part.allowed_orientations():
            # 노드에 물리적으로 들어가는지 확인
            if (
                orientation.l > node.dims.l + 1e-6
                or orientation.w > node.dims.w + 1e-6
                or orientation.t > node.dims.t + 1e-6
            ):
                continue

            w = orientation.w
            # 기존 버킷 중 STRIP_TOLERANCE 이내인 것 찾기
            matched_key = None
            for key in width_buckets:
                if abs(key - w) <= STRIP_TOLERANCE:
                    matched_key = key
                    break

            if matched_key is None:
                width_buckets[w] = []
                matched_key = w

            width_buckets[matched_key].append((part, orientation))

    # 부품 수 기준 내림차순 (큰 클러스터 우선)
    clusters = list(width_buckets.values())
    clusters.sort(key=lambda c: -len(c))
    return clusters


def _plan_strip(
    cluster: List[Tuple[Part, Dims]],
    node: Node,
    remaining_parts: dict[str, int],
    kerf: float,
) -> Optional[StripPlan]:
    """
    하나의 클러스터로부터 Strip 배치 계획을 수립합니다.

    [Strip Packing Phase 2 — Planning]
    - Strip 폭: 클러스터 내 최대 W값 (+ 여유 없음, Kerf는 분리 시 처리)
    - Strip 길이: node.dims.l (원장 전체 길이 방향)
    - 내부 배치: 클러스터 부품을 길이(L) 내림차순으로 순차 배치
    - 효율 추정: Σ(부품 면적) / Strip 면적

    Returns:
        StripPlan 또는 None (비효율적이거나 배치 불가)
    """
    if len(cluster) < STRIP_MIN_PARTS:
        return None

    # Strip 폭 = 클러스터 내 최대 W
    strip_w = max(orientation.w for _, orientation in cluster)

    # 노드의 W가 Strip 폭보다 작으면 불가
    if strip_w > node.dims.w + 1e-6:
        return None

    # 중복 제거: 동일 part_id는 남은 수량만큼만
    # cluster에는 allowed_orientations 전부 들어 있으므로
    # part_id별로 가장 L이 긴 방향 하나만 선택
    seen_part: Dict[str, Dims] = {}
    for part, orientation in cluster:
        pid = part.id
        if pid not in seen_part or orientation.l > seen_part[pid].l:
            seen_part[pid] = orientation

    # 배치 순서: L 내림차순 (긴 것 먼저 — 남은 공간에 짧은 것이 더 잘 맞음)
    ordered: List[Tuple[Part, Dims]] = sorted(
        [(parts_by_id_ref[pid], dims) for pid, dims in seen_part.items()
         if remaining_parts.get(pid, 0) > 0],
        key=lambda x: -x[1].l,
    )
    # parts_by_id_ref는 호출 시점에 주입 — 클로저 대신 파라미터로 받음
    # (아래 _plan_strip_with_parts 함수에서 처리)

    if not ordered:
        return None

    # Strip 내 순차 배치 시뮬레이션
    cursor_l = 0.0
    strip_l = node.dims.l
    parts_seq: List[Tuple[Part, Dims]] = []
    total_part_area = 0.0

    for part, orientation in ordered:
        qty_left = remaining_parts.get(part.id, 0)
        placed_count = 0
        for _ in range(qty_left):
            needed_l = orientation.l
            if cursor_l + needed_l > strip_l + 1e-6:
                break  # 이 Strip에 더 이상 안 들어감
            parts_seq.append((part, orientation))
            cursor_l += needed_l + kerf
            placed_count += 1
            total_part_area += orientation.l * orientation.w

    if not parts_seq:
        return None

    strip_area = strip_l * strip_w
    efficiency = total_part_area / strip_area if strip_area > 0 else 0.0

    if efficiency < STRIP_MIN_EFFICIENCY:
        return None

    return StripPlan(
        strip_width=strip_w,
        strip_axis=CutAxis.Y,
        parts_seq=parts_seq,
        efficiency=efficiency,
    )


def _plan_strip_with_parts(
    cluster: List[Tuple[Part, Dims]],
    node: Node,
    remaining_parts: dict[str, int],
    parts_by_id: dict[str, Part],
    kerf: float,
) -> Optional[StripPlan]:
    """
    parts_by_id를 외부에서 주입하여 _plan_strip을 실행하는 래퍼.
    (모듈 수준 전역 참조 없이 순수 함수 구조 유지)
    """
    global parts_by_id_ref
    parts_by_id_ref = parts_by_id
    return _plan_strip(cluster, node, remaining_parts, kerf)


# 모듈 수준 임시 참조 (순수 함수화 한계를 최소 범위로 처리)
parts_by_id_ref: dict[str, Part] = {}


def _execute_strip_plan(
    node: Node,
    plan: StripPlan,
    remaining_parts: dict[str, int],
    kerf: float,
) -> Tuple[List[Node], Optional[Node]]:
    """
    Strip 계획을 실제로 실행합니다.

    [Strip Packing Phase 3 — Execution]
    실행 순서:
      1. node를 Y축으로 strip_width만큼 절단
         → strip_node (l, strip_w, t)  — Strip 영역
         → residual_node (l, W-strip_w-k, t)  — 나머지 큰 잔재
      2. strip_node 내에서 부품을 X축(길이) 방향으로 순차 절단 배치
      3. Strip 내 남은 공간은 FREE로 heap에 반환

    Returns:
        (occupied_list, residual_node)
        residual_node는 Strip 바깥의 잔재 (None이면 딱 맞게 채워진 것)
    """
    occupied_list: List[Node] = []

    # ── Step 1: Strip 분리 절단 ───────────────────────────────────
    strip_w = plan.strip_width

    # Strip과 원장 W가 거의 같으면 별도 절단 불필요
    if abs(node.dims.w - strip_w) < 1e-6:
        strip_node = node
        residual_node = None
    else:
        remainder_w = node.dims.w - strip_w - kerf
        if remainder_w <= 0:
            # Kerf 때문에 잔재가 없어지는 경우 → Strip 전체 사용
            strip_node = node
            residual_node = None
        else:
            try:
                strip_node, residual_node = split_node(node, CutAxis.Y, strip_w, kerf)
            except InvalidCutError:
                return [], None

    # ── Step 2: Strip 내부 순차 배치 (X축 방향) ──────────────────
    current_strip = strip_node  # 현재 작업 공간 (점점 잘려나감)

    for part, orientation in plan.parts_seq:
        if remaining_parts.get(part.id, 0) <= 0:
            continue

        # 현재 strip 공간에 이 부품이 들어가는지 재확인
        if (
            orientation.l > current_strip.dims.l + 1e-6
            or orientation.w > current_strip.dims.w + 1e-6
            or orientation.t > current_strip.dims.t + 1e-6
        ):
            continue

        # Best-Area Fit으로 절단 순서 결정
        order_result = _best_cut_order(current_strip, orientation, kerf)
        if order_result is None:
            continue

        best_order, _ = order_result

        try:
            occupied = _place_part_on_node(
                node=current_strip,
                part=part,
                part_dims=orientation,
                cut_order=best_order,
                kerf=kerf,
            )
        except InvalidCutError:
            continue

        occupied_list.append(occupied)
        remaining_parts[part.id] -= 1

        # 배치 후 남은 X 방향 잔재를 다음 current_strip으로
        # (split_node가 child_b를 생성했으므로 부모 체인에서 찾기)
        next_strip = _find_strip_remainder(occupied, strip_node)
        if next_strip is None:
            break  # Strip 소진
        current_strip = next_strip

    return occupied_list, residual_node


def _find_strip_remainder(occupied: Node, strip_root: Node) -> Optional[Node]:
    """
    Strip 내 배치 후 X 방향으로 남은 FREE 잔재 노드를 찾습니다.

    occupied의 부모 체인을 역추적하여 strip_root 범위 내에서
    가장 큰 FREE child_b (X 방향 잔재)를 반환합니다.
    """
    current: Optional[Node] = occupied
    steps = 0

    while current is not None and steps <= 3:
        parent = current.parent
        if parent is None:
            break
        cb = parent.child_b
        if cb is not None and cb.state == NodeState.FREE:
            # X 방향 잔재인지 확인 (L이 클수록 Strip 내부 잔재)
            return cb
        current = parent
        steps += 1

    return None


# ══════════════════════════════════════════════════════════════════
# 6. Best-Fit 후보 평가 — 공간 중심 탐색 (v1 유지 + 면적 기준 개선)
# ══════════════════════════════════════════════════════════════════

def _find_best_candidate(
    node: Node,
    remaining_parts: dict[str, int],
    parts_by_id: dict[str, Part],
    kerf: float,
) -> Optional[PlacementCandidate]:
    """
    하나의 FREE 노드에 배치할 수 있는 모든 (part × orientation) 조합을 평가하고
    Best-Area Fit 점수가 가장 좋은 PlacementCandidate를 반환합니다.

    [v2 개선] score = node.XY_area - part_dims.XY_area (면적 기준)
    → 두께(Z)가 얇은 판재가 지배적인 현실에서 더 정확한 낭비 지표
    """
    best: Optional[PlacementCandidate] = None
    node_area = node.dims.l * node.dims.w  # XY 단면적

    for part_id, qty in remaining_parts.items():
        if qty <= 0:
            continue

        part = parts_by_id[part_id]

        for orientation in part.allowed_orientations():
            if (
                orientation.l > node.dims.l + 1e-6
                or orientation.w > node.dims.w + 1e-6
                or orientation.t > node.dims.t + 1e-6
            ):
                continue

            order_result = _best_cut_order(node, orientation, kerf)
            if order_result is None:
                continue

            best_order, max_offcut_area = order_result

            # [v2] XY 면적 기준 Best-Area Fit score
            part_area = orientation.l * orientation.w
            score = node_area - part_area

            candidate = PlacementCandidate(
                score=score,
                max_offcut=max_offcut_area,
                node=node,
                part=part,
                part_dims=orientation,
                cut_order=best_order,
            )

            if best is None or candidate < best:
                best = candidate

    return best


# ══════════════════════════════════════════════════════════════════
# 7. 단일 Stock 처리 루프 (v2 — Strip Phase 우선)
# ══════════════════════════════════════════════════════════════════

def _pack_single_stock(
    root: Node,
    remaining_parts: dict[str, int],
    parts_by_id: dict[str, Part],
    kerf: float,
) -> List[Node]:
    """
    하나의 원장(root Node)에 대해 Strip Packing 우선, Fallback Best-Fit 순으로 탐색합니다.

    [v2 2-Phase 전략]
    Phase 1 — Strip Packing:
      원장 루트 노드에서 폭 클러스터를 분석하고,
      효율적인 Strip이 존재하면 Strip 배치를 먼저 수행합니다.
      Strip 배치 후 남은 잔재 노드들은 heap에 등록됩니다.

    Phase 2 — Fallback Best-Fit:
      Strip으로 처리되지 않은 부품들과 잔재 공간을
      기존 v1 방식(Best-Area Fit + Max-Offcut)으로 처리합니다.
    """
    heap = NodeHeap()
    heap.push(root)

    occupied_nodes: List[Node] = []

    # ── Phase 1: Strip Packing ────────────────────────────────────
    # 루트 노드(가장 큰 공간)에서만 Strip 계획 수립
    strip_attempted = False
    if not strip_attempted:
        strip_attempted = True
        clusters = _cluster_parts_by_width(remaining_parts, parts_by_id, root, kerf)

        for cluster in clusters:
            if all(q <= 0 for q in remaining_parts.values()):
                break

            plan = _plan_strip_with_parts(cluster, root, remaining_parts, parts_by_id, kerf)
            if plan is None:
                continue

            # root가 아직 FREE인지 확인
            if root.state != NodeState.FREE:
                break

            strip_occupied, residual = _execute_strip_plan(root, plan, remaining_parts, kerf)
            occupied_nodes.extend(strip_occupied)

            # Strip 후 남은 잔재(residual)와 Strip 내 미사용 공간을 heap에 등록
            if residual is not None and residual.state == NodeState.FREE:
                heap.push(residual)

            # Strip 내 미사용 공간도 heap에 등록
            _collect_free_nodes_from(root, heap)

            # 첫 번째 성공한 Strip 이후 heap 기반 Fallback으로 전환
            break

    # ── Phase 2: Fallback Best-Fit ────────────────────────────────
    while True:
        if all(q <= 0 for q in remaining_parts.values()):
            break

        node = heap.pop()
        if node is None:
            break

        candidate = _find_best_candidate(node, remaining_parts, parts_by_id, kerf)

        if candidate is None:
            node.state = NodeState.DISCARDED
            continue

        try:
            occupied = _place_part_on_node(
                node=candidate.node,
                part=candidate.part,
                part_dims=candidate.part_dims,
                cut_order=candidate.cut_order,
                kerf=kerf,
            )
        except InvalidCutError:
            node.state = NodeState.DISCARDED
            continue

        occupied_nodes.append(occupied)
        remaining_parts[candidate.part.id] -= 1
        _register_new_free_nodes(occupied, heap)

    return occupied_nodes


def _collect_free_nodes_from(node: Node, heap: NodeHeap) -> None:
    """
    주어진 노드의 서브트리에서 FREE 상태인 노드를 모두 찾아 heap에 등록합니다.
    Strip 실행 후 내부 잔재 공간을 수집할 때 사용합니다.
    """
    stack = [node]
    visited: set[str] = set()

    while stack:
        current = stack.pop()
        if current.node_id in visited:
            continue
        visited.add(current.node_id)

        if current.state == NodeState.FREE:
            heap.push(current)
        elif current.state == NodeState.SPLIT:
            if current.child_a is not None:
                stack.append(current.child_a)
            if current.child_b is not None:
                stack.append(current.child_b)


def _register_new_free_nodes(occupied: Node, heap: NodeHeap) -> None:
    """
    배치가 완료된 occupied 노드로부터 부모 체인을 역추적하여
    새로 생성된 FREE child_b 노드들을 heap에 등록합니다.
    """
    current: Optional[Node] = occupied
    steps = 0
    max_steps = 3

    while current is not None and steps <= max_steps:
        parent = current.parent
        if parent is None:
            break
        if parent.state == NodeState.SPLIT:
            cb = parent.child_b
            if cb is not None and cb.state == NodeState.FREE:
                heap.push(cb)
        current = parent
        steps += 1


# ══════════════════════════════════════════════════════════════════
# 8. Part 정렬 전략 (v2 — 면적 + 부피 복합 기준)
# ══════════════════════════════════════════════════════════════════

def _sort_parts_for_packing(parts: List[Part]) -> List[Part]:
    """
    초기 배치 효율을 높이기 위한 Part 정렬.

    [v2 개선] 부피 단일 기준 → (면적, 부피) 복합 기준
    - 1차: XY 면적 내림차순 (넓은 것 먼저 → Strip 그룹화 유리)
    - 2차: 부피 내림차순 (두꺼운 것 우선)
    - 3차: priority 내림차순, id 오름차순 (결정론적)
    """
    return sorted(
        parts,
        key=lambda p: (-(p.dims.l * p.dims.w), -p.dims.volume, -p.priority, p.id),
    )


# ══════════════════════════════════════════════════════════════════
# 9. 메인 공개 함수
# ══════════════════════════════════════════════════════════════════

def pack_parts(
    settings: EngineSettings,
    stocks: List[Stock],
    parts: List[Part],
) -> List[Node]:
    """
    3D Guillotine Heuristic Packer v2 — 메인 진입점.

    알고리즘 개요 (v2):
      1. Parts를 (면적, 부피) 복합 기준 내림차순 정렬
      2. Stock을 순서대로 처리
         a. 각 Stock에서 create_root_node()로 루트 노드 생성 (trimming 적용)
         b. [신규] Phase 1 — Strip Packing:
              폭 기준 클러스터링 → Strip 효율 추정 → Strip 배치 실행
         c. Phase 2 — Fallback Best-Area Fit:
              NodeHeap 기반 공간 중심 탐색 (v1 방식 + 면적 기준 개선)
      3. 모든 Part 배치 완료 또는 Stock 소진 시 종료

    Args:
        settings: kerf, trimming, optimization_goal 등 전역 설정
        stocks  : 사용 가능한 원장 목록
        parts   : 배치해야 할 부품 목록

    Returns:
        배치된 OCCUPIED 노드 목록
    """
    if not parts:
        return []
    if not stocks:
        return []

    # ── 초기화 ─────────────────────────────────────────────────────
    sorted_parts = _sort_parts_for_packing(parts)
    parts_by_id: dict[str, Part] = {p.id: p for p in sorted_parts}

    # 잔여 수량 추적
    remaining_parts: dict[str, int] = {p.id: p.qty for p in sorted_parts}

    all_occupied: List[Node] = []
    kerf = settings.kerf

    # ── Stock 순서 처리 ───────────────────────────────────────────
    for stock in stocks:
        for _copy_idx in range(stock.qty):
            if all(q <= 0 for q in remaining_parts.values()):
                break

            unique_stock_id = f"{stock.id}-{_copy_idx + 1}"

            root = create_root_node(stock)
            root.stock_id = unique_stock_id

            occupied = _pack_single_stock(
                root=root,
                remaining_parts=remaining_parts,
                parts_by_id=parts_by_id,
                kerf=kerf,
            )
            all_occupied.extend(occupied)

        if all(q <= 0 for q in remaining_parts.values()):
            break

    # ── 미배치 Part 경고 출력 ────────────────────────────────────
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
# 10. 결과 분석 유틸리티
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
