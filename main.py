"""
main.py — 3D Guillotine Cutting Optimization Engine: FastAPI Server
===================================================================

실행 방법:
    pip install fastapi uvicorn pydantic
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

API 문서:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator, model_validator

# ── 엔진 임포트 ─────────────────────────────────────────────────
from core import (
    CuttingError,
    Dims,
    EngineSettings,
    InvalidCutError,
    Node,
    NodeState,
    OptimizationGoal,
    Part,
    Stock,
    TrimmingMargins,
)
from packer import pack_parts

# ══════════════════════════════════════════════════════════════════
# 1. 로거 설정
# ══════════════════════════════════════════════════════════════════

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("cutting_engine")


# ══════════════════════════════════════════════════════════════════
# 2. Pydantic Request Models
# ══════════════════════════════════════════════════════════════════

class TrimmingIn(BaseModel):
    x: float = Field(default=0.0, ge=0, description="X축 양단 여백 (단위: mm 등)")
    y: float = Field(default=0.0, ge=0, description="Y축 양단 여백")
    z: float = Field(default=0.0, ge=0, description="Z축 양단 여백")

class SettingsIn(BaseModel):
    kerf: float = Field(default=3.0, ge=0, description="절단 시 소모되는 톱날 두께 (loss)")
    trimming: TrimmingIn = Field(default_factory=TrimmingIn, description="원장 테두리 여백")
    optimization_goal: str = Field(default="MINIMIZE_WASTE")

    @field_validator("optimization_goal")
    @classmethod
    def validate_goal(cls, v: str) -> str:
        allowed = {g.name for g in OptimizationGoal}
        if v.upper() not in allowed:
            raise ValueError(f"optimization_goal은 {allowed} 중 하나여야 합니다.")
        return v.upper()

class StockIn(BaseModel):
    id: str = Field(description="원장 고유 ID")
    l: float = Field(gt=0, description="길이 (X축)")
    w: float = Field(gt=0, description="너비 (Y축)")
    t: float = Field(gt=0, description="두께 (Z축)")
    qty: int = Field(default=1, ge=1, description="보유 수량")

class PartIn(BaseModel):
    id: str = Field(description="부품 고유 ID")
    l: float = Field(gt=0, description="길이 (X축)")
    w: float = Field(gt=0, description="너비 (Y축)")
    t: float = Field(gt=0, description="두께 (Z축)")
    qty: int = Field(ge=1, description="필요 수량")
    lock_z: bool = Field(default=True, description="True이면 Z축(두께) 방향 고정")
    allow_xy_rotation: bool = Field(default=True, description="XY 평면 90도 회전 허용")
    priority: int = Field(default=0, description="배치 우선순위")

    @model_validator(mode="after")
    def validate_dims_vs_stock(self) -> "PartIn":
        if self.l <= 0 or self.w <= 0 or self.t <= 0:
            raise ValueError("부품 치수는 모두 양수여야 합니다.")
        return self

class OptimizeRequest(BaseModel):
    settings: SettingsIn = Field(default_factory=SettingsIn)
    stocks: List[StockIn] = Field(min_length=1, description="원장 목록")
    parts: List[PartIn] = Field(min_length=1, description="부품 목록")

    @model_validator(mode="after")
    def validate_part_fits_any_stock(self) -> "OptimizeRequest":
        for part in self.parts:
            fits = False
            for stock in self.stocks:
                if part.lock_z and abs(part.t - stock.t) > 1e-6:
                    continue
                if part.l <= stock.l and part.w <= stock.w:
                    fits = True; break
                if part.allow_xy_rotation and part.w <= stock.l and part.l <= stock.w:
                    fits = True; break
                if not part.lock_z:
                    fits = True; break
            if not fits:
                raise ValueError(
                    f"Part '{part.id}' ({part.l}×{part.w}×{part.t})가 "
                    "어떤 Stock에도 맞지 않습니다. 치수나 lock_z 설정을 확인하세요."
                )
        return self


# ══════════════════════════════════════════════════════════════════
# 3. Pydantic Response Models
# ══════════════════════════════════════════════════════════════════

class DimsOut(BaseModel):
    l: float; w: float; t: float; volume: float

class OriginOut(BaseModel):
    x: float; y: float; z: float

class CutRecordOut(BaseModel):
    cut_id: str; axis: str; position: float; kerf: float; parent_node_id: str

class PlacedPartOut(BaseModel):
    node_id: str; stock_id: str; part_id: str
    placed_dims: DimsOut
    origin: OriginOut
    cut_history: List[CutRecordOut]
    depth: int

class StockSummaryOut(BaseModel):
    stock_id: str
    original_dims: DimsOut
    usable_dims: DimsOut
    placed_count: int
    placed_volume: float
    usable_volume: float
    efficiency_pct: float

class OptimizeResponse(BaseModel):
    placements: List[PlacedPartOut]
    unplaced: Dict[str, int]
    stock_summaries: List[StockSummaryOut]
    stats: Dict[str, Any]
    # [v3 추가] 백엔드의 상세 정보(카메라 중심점, 실패 사유, 패킹 모드)를 프론트엔드로 전달
    failures: List[Dict[str, Any]] = Field(default_factory=list)
    stock_centers: List[Dict[str, Any]] = Field(default_factory=list)
    mode: str = Field(default="2D")

class ErrorResponse(BaseModel):
    error: str; detail: Optional[str] = None; error_code: str


# ══════════════════════════════════════════════════════════════════
# 4. 도메인 변환 헬퍼
# ══════════════════════════════════════════════════════════════════

def _build_engine_settings(s: SettingsIn) -> EngineSettings:
    return EngineSettings(
        kerf=s.kerf,
        trimming=TrimmingMargins(x=s.trimming.x, y=s.trimming.y, z=s.trimming.z),
        optimization_goal=OptimizationGoal[s.optimization_goal],
    )

def _build_stocks(stocks_in: List[StockIn], trimming: TrimmingMargins) -> List[Stock]:
    return [Stock(id=s.id, dims=Dims(l=s.l, w=s.w, t=s.t), qty=s.qty, trimming=trimming) for s in stocks_in]

def _build_parts(parts_in: List[PartIn]) -> List[Part]:
    return [Part(id=p.id, dims=Dims(l=p.l, w=p.w, t=p.t), qty=p.qty, lock_z=p.lock_z, allow_xy_rotation=p.allow_xy_rotation, priority=p.priority) for p in parts_in]

def _dims_to_out(d: Dims) -> DimsOut:
    return DimsOut(l=d.l, w=d.w, t=d.t, volume=d.volume)

def _node_to_placed_out(node: Node) -> PlacedPartOut:
    assert node.placed_part is not None and node.placed_part_dims is not None
    history = [
        CutRecordOut(cut_id=cut.cut_id, axis=cut.axis.value, position=cut.position, kerf=cut.kerf, parent_node_id=cut.parent_node_id)
        for cut in node.collect_cut_history()
    ]
    return PlacedPartOut(
        node_id=node.node_id, stock_id=node.stock_id or "unknown",
        part_id=node.placed_part.id, placed_dims=_dims_to_out(node.placed_part_dims),
        origin=OriginOut(x=node.origin.x, y=node.origin.y, z=node.origin.z),
        cut_history=history, depth=node.depth,
    )

def _build_stock_summaries(occupied_nodes: List[Node], stocks: List[Stock]) -> List[StockSummaryOut]:
    by_stock: Dict[str, List[Node]] = {}
    for node in occupied_nodes:
        sid = node.stock_id or "unknown"
        by_stock.setdefault(sid, []).append(node)

    stock_map = {s.id: s for s in stocks}
    summaries = []

    for sid, nodes in by_stock.items():
        stock = stock_map.get(sid)
        if stock is None: continue
        placed_vol = sum(n.placed_part_dims.volume for n in nodes if n.placed_part_dims)
        usable_vol = stock.usable_volume
        efficiency = (placed_vol / usable_vol * 100) if usable_vol > 0 else 0.0

        summaries.append(StockSummaryOut(
            stock_id=sid, original_dims=_dims_to_out(stock.dims), usable_dims=_dims_to_out(stock.usable_dims),
            placed_count=len(nodes), placed_volume=round(placed_vol, 4), usable_volume=round(usable_vol, 4),
            efficiency_pct=round(efficiency, 2),
        ))
    return sorted(summaries, key=lambda s: s.stock_id)


# ══════════════════════════════════════════════════════════════════
# 5. CPU-bound 작업을 동기 함수로 래핑
# ══════════════════════════════════════════════════════════════════

def _run_packing(settings: EngineSettings, stocks: List[Stock], parts: List[Part]) -> Any:
    """
    [v3 변경점]
    pack_parts()는 이제 단순히 Node 리스트가 아니라 PackResult 객체를 통째로 반환합니다.
    잔여 수량 계산도 엔진 내부에서 다 알아서 처리하므로 그대로 리턴합니다.
    """
    return pack_parts(settings=settings, stocks=stocks, parts=parts)


# ══════════════════════════════════════════════════════════════════
# 6. FastAPI 앱 초기화 및 예외 핸들러
# ══════════════════════════════════════════════════════════════════

app = FastAPI(title="3D Guillotine Cutting Optimization Engine")

app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

@app.exception_handler(InvalidCutError)
async def invalid_cut_handler(request: Request, exc: InvalidCutError) -> JSONResponse:
    return JSONResponse(status_code=400, content=ErrorResponse(error="Invalid cut parameters", detail=str(exc), error_code="INVALID_CUT").model_dump())

@app.exception_handler(CuttingError)
async def cutting_error_handler(request: Request, exc: CuttingError) -> JSONResponse:
    return JSONResponse(status_code=422, content=ErrorResponse(error="Cutting engine error", detail=str(exc), error_code="CUTTING_ERROR").model_dump())

@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    return JSONResponse(status_code=400, content=ErrorResponse(error="Invalid input value", detail=str(exc), error_code="VALUE_ERROR").model_dump())

@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.exception("Unexpected error")
    return JSONResponse(status_code=500, content=ErrorResponse(error="Internal server error", detail="서버 내부 오류", error_code="INTERNAL_ERROR").model_dump())


# ══════════════════════════════════════════════════════════════════
# 7. 엔드포인트
# ══════════════════════════════════════════════════════════════════

@app.post("/optimize", response_model=OptimizeResponse)
async def optimize(body: OptimizeRequest) -> OptimizeResponse:
    t_start = time.perf_counter()

    try:
        engine_settings = _build_engine_settings(body.settings)
        stocks = _build_stocks(body.stocks, engine_settings.trimming)
        parts = _build_parts(body.parts)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"입력 데이터 변환 오류: {e}")

    try:
        # [v3 수정] pack_result는 이제 튜플이 아닌 PackResult 객체입니다.
        pack_result = await run_in_threadpool(
            functools.partial(_run_packing, engine_settings, stocks, parts)
        )
        occupied_nodes = pack_result.occupied
        unplaced = pack_result.unplaced
    except (InvalidCutError, CuttingError):
        raise
    except Exception as e:
        logger.exception("pack_parts() failed")
        raise HTTPException(status_code=500, detail=f"최적화 연산 중 오류: {e}")

    elapsed = time.perf_counter() - t_start

    placements = [_node_to_placed_out(n) for n in occupied_nodes]
    stock_summaries = _build_stock_summaries(occupied_nodes, stocks)

    total_placed_vol = sum(p.placed_dims.volume for p in placements)
    total_usable_vol = sum(s.usable_volume * s.qty for s in stocks)
    overall_efficiency = (total_placed_vol / total_usable_vol * 100) if total_usable_vol > 0 else 0.0

    stats = {
        "total_placed": len(occupied_nodes),
        "total_unplaced_types": len(unplaced),
        "total_placed_volume": round(total_placed_vol, 4),
        "total_usable_volume": round(total_usable_vol, 4),
        "overall_efficiency_pct": round(overall_efficiency, 2),
        "stocks_used": len({n.stock_id for n in occupied_nodes}),
        "processing_time_sec": round(elapsed, 4),
    }

    # [v3 추가] 엔진에서 보내주는 추가 데이터(실패 사유, 중앙점, 모드)를 함께 반환합니다.
    return OptimizeResponse(
        placements=placements,
        unplaced=unplaced,
        stock_summaries=stock_summaries,
        stats=stats,
        failures=[f.to_dict() for f in pack_result.failures],
        stock_centers=[c.to_dict() for c in pack_result.stock_centers],
        mode=pack_result.mode.value
    )
