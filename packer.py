"""
packer.py — Ultimate 3D Guillotine Packer (v4 - Part-Centric Best-Fit)
================================================================================
인위적인 모드 전환이나 1D Strip 꼼수 없이, 순수 수학적 Best-Fit과 Max-Rest로
2D/3D 모든 환경에서 완벽한 테트리스(Z축 스태킹 포함)를 구현합니다.
"""

from __future__ import annotations
import time
from collections import defaultdict
from dataclasses import dataclass, field
from itertools import permutations
from typing import Dict, List, Optional, Tuple
from core import CutAxis, Dims, EngineSettings, InvalidCutError, Node, NodeState, Part, Stock, create_root_node, split_node

EPSILON = 1e-6
ALL_CUT_ORDERS: List[Tuple[CutAxis, CutAxis, CutAxis]] = list(permutations([CutAxis.X, CutAxis.Y, CutAxis.Z])) # type: ignore

@dataclass
class PlacementFailure:
    part_id: str; stock_id: str; reason: str; detail: str
    def to_dict(self): return {"part_id": self.part_id, "stock_id": self.stock_id, "reason": self.reason, "detail": self.detail}

@dataclass
class StockCenter:
    stock_id: str; cx: float; cy: float; cz: float
    def to_dict(self): return {"stock_id": self.stock_id, "cx": self.cx, "cy": self.cy, "cz": self.cz}

@dataclass
class PackResult:
    occupied: List[Node]; failures: List[PlacementFailure]; stock_centers: List[StockCenter]; mode: str; unplaced: Dict[str, int]

@dataclass
class PlacementCandidate:
    node: Node; part: Part; part_dims: Dims; cut_order: Tuple[CutAxis, CutAxis, CutAxis]
    node_vol: float; max_offcut_vol: float

    def __lt__(self, other: 'PlacementCandidate') -> bool:
        # 1순위: 들어갈 노드의 부피가 작을수록 좋음 (가장 꽉 끼는 곳 = Best Fit)
        if abs(self.node_vol - other.node_vol) > EPSILON:
            return self.node_vol < other.node_vol
        # 2순위: 남는 자투리 중 가장 큰 덩어리가 거대할수록 좋음 (Max-Rest)
        return self.max_offcut_vol > other.max_offcut_vol

def _simulate_cuts(node: Node, req_dims: Dims, cut_order: Tuple[CutAxis, CutAxis, CutAxis], kerf: float) -> Optional[List[float]]:
    rem = {CutAxis.X: node.dims.l, CutAxis.Y: node.dims.w, CutAxis.Z: node.dims.t}
    req = {CutAxis.X: req_dims.l, CutAxis.Y: req_dims.w, CutAxis.Z: req_dims.t}
    offcuts = []
    for axis in cut_order:
        total, needed = rem[axis], req[axis]
        if abs(total - needed) < EPSILON:
            offcuts.append(0.0)
            continue
        if needed <= 0 or needed > total + EPSILON: return None
        leftover = total - needed - kerf
        if leftover <= EPSILON: return None
        v = rem[CutAxis.X] * rem[CutAxis.Y] * rem[CutAxis.Z]
        offcuts.append((v / total) * leftover)
        rem[axis] = needed
    return offcuts

def _find_best_cand(free_nodes: List[Node], part: Part, kerf: float) -> Optional[PlacementCandidate]:
    best = None
    for node in free_nodes:
        if node.volume < part.dims.volume - EPSILON: continue
        for orient in part.allowed_orientations():
            if orient.l > node.dims.l + EPSILON or orient.w > node.dims.w + EPSILON or orient.t > node.dims.t + EPSILON: continue
            for order in ALL_CUT_ORDERS:
                offcuts = _simulate_cuts(node, orient, order, kerf)
                if offcuts is None: continue
                cand = PlacementCandidate(node, part, orient, order, node.volume, max(offcuts, default=0.0))
                if best is None or cand < best: best = cand
    return best

def _execute_placement(cand: PlacementCandidate, kerf: float) -> Tuple[Node, List[Node]]:
    cur = cand.node
    new_frees = []
    req = {CutAxis.X: cand.part_dims.l, CutAxis.Y: cand.part_dims.w, CutAxis.Z: cand.part_dims.t}
    for axis in cand.cut_order:
        total = {CutAxis.X: cur.dims.l, CutAxis.Y: cur.dims.w, CutAxis.Z: cur.dims.t}[axis]
        if abs(total - req[axis]) < EPSILON: continue
        child_a, child_b = split_node(cur, axis, req[axis], kerf)
        if child_b and child_b.state == NodeState.FREE: new_frees.append(child_b)
        cur = child_a
    cur.state = NodeState.OCCUPIED
    cur.placed_part = cand.part
    cur.placed_part_dims = cand.part_dims
    return cur, new_frees

def _compute_stock_center(stock: Stock, trimming: object) -> StockCenter:
    def _g(k): return float(getattr(trimming, k, 0.0)) if trimming else 0.0
    tx, ty, tz = _g("x"), _g("y"), _g("z")
    return StockCenter(stock.id, tx + max(0, stock.dims.l - 2*tx)/2, ty + max(0, stock.dims.w - 2*ty)/2, tz + max(0, stock.dims.t - 2*tz)/2)

def pack_parts(settings: EngineSettings, stocks: List[Stock], parts: List[Part]) -> PackResult:
    if not parts or not stocks: return PackResult([], [], [], "Universal", {})
    kerf = settings.kerf; trimming = getattr(settings, "trimming", None)
    
    # 1. 부품을 덩치가 큰 순서대로 정렬 (가장 큰 놈부터 자리를 선점)
    sorted_parts = sorted(parts, key=lambda p: (-p.dims.volume, -(p.dims.l * p.dims.w), -p.priority, p.id))
    
    stock_pool, stock_centers = [], []
    for s in stocks:
        c = _compute_stock_center(s, trimming)
        for i in range(s.qty):
            uid = f"{s.id}-{i+1}"
            stock_centers.append(StockCenter(uid, c.cx, c.cy, c.cz))
            root = create_root_node(s); root.stock_id = uid; stock_pool.append(root)

    free_nodes, occupied_nodes, failures = [], [], []
    unplaced = defaultdict(int)

    for part in sorted_parts:
        for _ in range(part.qty):
            placed = False
            # 이미 열려있는 원장/자투리 중에서 최적의 공간을 찾음
            cand = _find_best_cand(free_nodes, part, kerf)
            if cand:
                occ, new_frees = _execute_placement(cand, kerf)
                free_nodes.remove(cand.node); free_nodes.extend(new_frees); occupied_nodes.append(occ)
                placed = True
            else:
                # 안 맞으면 새 원장을 하나씩 꺼내서 뜯어봄
                while stock_pool and not placed:
                    new_root = stock_pool.pop(0)
                    free_nodes.append(new_root)
                    new_cand = _find_best_cand([new_root], part, kerf)
                    if new_cand:
                        occ, new_frees = _execute_placement(new_cand, kerf)
                        free_nodes.remove(new_cand.node); free_nodes.extend(new_frees); occupied_nodes.append(occ)
                        placed = True
            if not placed:
                unplaced[part.id] += 1
                failures.append(PlacementFailure(part.id, "ALL", "원장 공간 부족", "남은 자투리나 새 원장 중 어디에도 들어가지 않습니다."))

    return PackResult(occupied_nodes, failures, stock_centers, "Universal 3D", dict(unplaced))
