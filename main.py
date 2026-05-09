from __future__ import annotations
import functools, logging, time
from typing import Any, Dict, List, Optional
from fastapi import FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

from core import CuttingError, Dims, EngineSettings, InvalidCutError, OptimizationGoal, Part, Stock, TrimmingMargins
from packer import pack_parts

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("cutting_engine")

class TrimmingIn(BaseModel):
    x: float = Field(default=0.0, ge=0); y: float = Field(default=0.0, ge=0); z: float = Field(default=0.0, ge=0)

class SettingsIn(BaseModel):
    kerf: float = Field(default=3.0, ge=0); trimming: TrimmingIn = Field(default_factory=TrimmingIn)
    optimization_goal: str = Field(default="MINIMIZE_WASTE")

class StockIn(BaseModel):
    id: str; l: float = Field(gt=0); w: float = Field(gt=0); t: float = Field(gt=0); qty: int = Field(default=1, ge=1)

class PartIn(BaseModel):
    id: str; l: float = Field(gt=0); w: float = Field(gt=0); t: float = Field(gt=0); qty: int = Field(ge=1)
    lock_z: bool = Field(default=True); allow_xy_rotation: bool = Field(default=True); priority: int = Field(default=0)

class OptimizeRequest(BaseModel):
    settings: SettingsIn = Field(default_factory=SettingsIn)
    stocks: List[StockIn] = Field(min_length=1); parts: List[PartIn] = Field(min_length=1)

class DimsOut(BaseModel): l: float; w: float; t: float; volume: float
class OriginOut(BaseModel): x: float; y: float; z: float
class CutRecordOut(BaseModel): cut_id: str; axis: str; position: float; kerf: float; parent_node_id: str
class PlacedPartOut(BaseModel):
    node_id: str; stock_id: str; part_id: str; placed_dims: DimsOut; origin: OriginOut; cut_history: List[CutRecordOut]; depth: int
class StockSummaryOut(BaseModel):
    stock_id: str; original_dims: DimsOut; usable_dims: DimsOut; placed_count: int; placed_volume: float; usable_volume: float; efficiency_pct: float

class OptimizeResponse(BaseModel):
    placements: List[PlacedPartOut]; unplaced: Dict[str, int]; stock_summaries: List[StockSummaryOut]; stats: Dict[str, Any]
    failures: List[Dict[str, Any]] = Field(default_factory=list); stock_centers: List[Dict[str, Any]] = Field(default_factory=list); mode: str = Field(default="Universal")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

def _node_to_placed_out(node):
    history = [CutRecordOut(cut_id=c.cut_id, axis=c.axis.value, position=c.position, kerf=c.kerf, parent_node_id=c.parent_node_id) for c in node.collect_cut_history()]
    return PlacedPartOut(node_id=node.node_id, stock_id=node.stock_id or "unknown", part_id=node.placed_part.id, placed_dims=DimsOut(l=node.placed_part_dims.l, w=node.placed_part_dims.w, t=node.placed_part_dims.t, volume=node.placed_part_dims.volume), origin=OriginOut(x=node.origin.x, y=node.origin.y, z=node.origin.z), cut_history=history, depth=node.depth)

@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(body: OptimizeRequest):
    t_start = time.perf_counter()
    s = body.settings
    trim = TrimmingMargins(x=s.trimming.x, y=s.trimming.y, z=s.trimming.z)
    eng_set = EngineSettings(kerf=s.kerf, trimming=trim, optimization_goal=OptimizationGoal[s.optimization_goal])
    stocks = [Stock(id=st.id, dims=Dims(l=st.l, w=st.w, t=st.t), qty=st.qty, trimming=trim) for st in body.stocks]
    parts = [Part(id=p.id, dims=Dims(l=p.l, w=p.w, t=p.t), qty=p.qty, lock_z=p.lock_z, allow_xy_rotation=p.allow_xy_rotation, priority=p.priority) for p in body.parts]

    pack_result = await run_in_threadpool(functools.partial(pack_parts, eng_set, stocks, parts))
    
    placements = [_node_to_placed_out(n) for n in pack_result.occupied]
    by_stock = defaultdict(list)
    for n in pack_result.occupied: by_stock[n.stock_id or "unknown"].append(n)
    
    stock_map = {f"{s.id}-{i+1}": s for s in stocks for i in range(s.qty)}
    summaries = []
    for sid, nodes in by_stock.items():
        st = stock_map.get(sid)
        if not st: continue
        p_vol = sum(n.placed_part_dims.volume for n in nodes)
        eff = (p_vol / st.usable_volume * 100) if st.usable_volume > 0 else 0
        summaries.append(StockSummaryOut(stock_id=sid, original_dims=DimsOut(l=st.dims.l, w=st.dims.w, t=st.dims.t, volume=st.dims.volume), usable_dims=DimsOut(l=st.usable_dims.l, w=st.usable_dims.w, t=st.usable_dims.t, volume=st.usable_volume), placed_count=len(nodes), placed_volume=round(p_vol,4), usable_volume=round(st.usable_volume,4), efficiency_pct=round(eff,2)))

    t_vol = sum(p.placed_dims.volume for p in placements)
    u_vol = sum(s.usable_volume * s.qty for s in stocks)
    stats = {
        "total_placed": len(pack_result.occupied), "total_unplaced_types": len(pack_result.unplaced),
        "total_placed_volume": round(t_vol,4), "total_usable_volume": round(u_vol,4),
        "overall_efficiency_pct": round((t_vol/u_vol*100) if u_vol>0 else 0, 2),
        "stocks_used": len(by_stock), "processing_time_sec": round(time.perf_counter() - t_start, 4)
    }

    return OptimizeResponse(placements=placements, unplaced=pack_result.unplaced, stock_summaries=summaries, stats=stats, failures=[f.to_dict() for f in pack_result.failures], stock_centers=[c.to_dict() for c in pack_result.stock_centers], mode=pack_result.mode)
