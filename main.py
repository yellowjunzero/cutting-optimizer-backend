"""
main.py — 3D Guillotine Cutting Optimization Engine: FastAPI Server
===================================================================

실행 방법:
    pip install fastapi uvicorn pydantic
    uvicorn main:app --reload --host 0.0.0.0 --port 8000

API 문서:
    http://localhost:8000/docs   (Swagger UI)
    http://localhost:8000/redoc  (ReDoc)

레이어 구조:
    [Client JSON]
        → Pydantic Request Model (입력 검증)
        → run_in_threadpool (이벤트 루프 블로킹 방지)
        → pack_parts() (CPU-bound 최적화 연산)
        → List[Node] → Pydantic Response Model 변환
        → [Client JSON]
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
#    클라이언트 → 서버 방향의 입력 데이터 검증
# ══════════════════════════════════════════════════════════════════

class TrimmingIn(BaseModel):
    """[C3] 축별 초기 테두리 여백 입력."""
    x: float = Field(default=0.0, ge=0, description="X축 양단 여백 (단위: mm 등)")
    y: float = Field(default=0.0, ge=0, description="Y축 양단 여백")
    z: float = Field(default=0.0, ge=0, description="Z축 양단 여백")


class SettingsIn(BaseModel):
    """최적화 엔진 전역 설정."""
    kerf: float = Field(
        default=3.0,
        ge=0,
        description="절단 시 소모되는 톱날 두께 (loss)",
    )
    trimming: TrimmingIn = Field(
        default_factory=TrimmingIn,
        description="원장 테두리 여백",
    )
    optimization_goal: str = Field(
        default="MINIMIZE_WASTE",
        description="최적화 목표: MINIMIZE_WASTE | MINIMIZE_SHEETS | MINIMIZE_CUTS",
    )

    @field_validator("optimization_goal")
    @classmethod
    def validate_goal(cls, v: str) -> str:
        allowed = {g.name for g in OptimizationGoal}
        if v.upper() not in allowed:
            raise ValueError(f"optimization_goal은 {allowed} 중 하나여야 합니다.")
        return v.upper()

    model_config = {
        "json_schema_extra": {
            "example": {
                "kerf": 3.0,
                "trimming": {"x": 10, "y": 10, "z": 0},
                "optimization_goal": "MINIMIZE_WASTE",
            }
        }
    }


class StockIn(BaseModel):
    """원장(원자재) 입력 모델."""
    id: str = Field(description="원장 고유 ID (예: 'S1')")
    l: float = Field(gt=0, description="길이 (X축)")
    w: float = Field(gt=0, description="너비 (Y축)")
    t: float = Field(gt=0, description="두께 (Z축)")
    qty: int = Field(default=1, ge=1, description="보유 수량")

    model_config = {
        "json_schema_extra": {
            "example": {"id": "S1", "l": 2440, "w": 1220, "t": 18, "qty": 5}
        }
    }


class PartIn(BaseModel):
    """절단할 부품 입력 모델."""
    id: str = Field(description="부품 고유 ID (예: 'P1')")
    l: float = Field(gt=0, description="길이 (X축)")
    w: float = Field(gt=0, description="너비 (Y축)")
    t: float = Field(gt=0, description="두께 (Z축)")
    qty: int = Field(ge=1, description="필요 수량")
    lock_z: bool = Field(
        default=True,
        description="[C4] True이면 Z축(두께) 방향 고정",
    )
    allow_xy_rotation: bool = Field(
        default=True,
        description="[C4] lock_z=True일 때 XY 평면 90도 회전 허용",
    )
    priority: int = Field(
        default=0,
        description="배치 우선순위 (높을수록 먼저 배치 시도)",
    )

    @model_validator(mode="after")
    def validate_dims_vs_stock(self) -> "PartIn":
        if self.l <= 0 or self.w <= 0 or self.t <= 0:
            raise ValueError("부품 치수는 모두 양수여야 합니다.")
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "id": "P1",
                "l": 600,
                "w": 400,
                "t": 18,
                "qty": 10,
                "lock_z": True,
                "allow_xy_rotation": True,
                "priority": 0,
            }
        }
    }


class OptimizeRequest(BaseModel):
    """
    POST /optimize 전체 요청 바디.

    프론트엔드 개발자용 Swagger 예시 데이터가
    json_schema_extra에 포함되어 있어 /docs에서 바로 확인 가능합니다.
    """
    settings: SettingsIn = Field(default_factory=SettingsIn)
    stocks: List[StockIn] = Field(min_length=1, description="원장 목록 (최소 1개)")
    parts: List[PartIn] = Field(min_length=1, description="부품 목록 (최소 1개)")

    @model_validator(mode="after")
    def validate_part_fits_any_stock(self) -> "OptimizeRequest":
        """
        각 Part가 최소 하나의 Stock에 들어갈 수 있는지 사전 검증.
        (trimming 제외한 단순 치수 비교 — 조기 실패 유도)
        """
        for part in self.parts:
            fits = False
            for stock in self.stocks:
                # lock_z: t == stock.t 여야 함
                if part.lock_z and abs(part.t - stock.t) > 1e-6:
                    continue
                # 가장 느슨한 배치: l,w가 stock l,w 이내
                if part.l <= stock.l and part.w <= stock.w:
                    fits = True
                    break
                if part.allow_xy_rotation and part.w <= stock.l and part.l <= stock.w:
                    fits = True
                    break
                if not part.lock_z:
                    fits = True  # 3D 자유 회전 — 세밀한 체크는 엔진에 위임
                    break
            if not fits:
                raise ValueError(
                    f"Part '{part.id}' ({part.l}×{part.w}×{part.t})가 "
                    "어떤 Stock에도 맞지 않습니다. 치수나 lock_z 설정을 확인하세요."
                )
        return self

    model_config = {
        "json_schema_extra": {
            "example": {
                "settings": {
                    "kerf": 3.0,
                    "trimming": {"x": 10, "y": 10, "z": 0},
                    "optimization_goal": "MINIMIZE_WASTE",
                },
                "stocks": [
                    {"id": "S1", "l": 2440, "w": 1220, "t": 18, "qty": 5}
                ],
                "parts": [
                    {
                        "id": "P1",
                        "l": 600,
                        "w": 400,
                        "t": 18,
                        "qty": 10,
                        "lock_z": True,
                        "allow_xy_rotation": True,
                        "priority": 0,
                    },
                    {
                        "id": "P2",
                        "l": 300,
                        "w": 200,
                        "t": 18,
                        "qty": 8,
                        "lock_z": True,
                        "allow_xy_rotation": True,
                        "priority": 1,
                    },
                    {
                        "id": "P3",
                        "l": 800,
                        "w": 600,
                        "t": 18,
                        "qty": 3,
                        "lock_z": True,
                        "allow_xy_rotation": False,
                        "priority": 2,
                    },
                ],
            }
        }
    }


# ══════════════════════════════════════════════════════════════════
# 3. Pydantic Response Models
#    서버 → 클라이언트 방향의 출력 직렬화
# ══════════════════════════════════════════════════════════════════

class DimsOut(BaseModel):
    """치수 출력."""
    l: float
    w: float
    t: float
    volume: float


class OriginOut(BaseModel):
    """원장 기준 절대 좌표."""
    x: float
    y: float
    z: float


class CutRecordOut(BaseModel):
    """
    단일 Guillotine Cut 기록 출력.
    절단 이력(Cut History) 구성 요소입니다.
    """
    cut_id: str = Field(description="절단 고유 ID")
    axis: str = Field(description="절단 축: X | Y | Z")
    position: float = Field(description="로컬 좌표 기준 절단 위치")
    kerf: float = Field(description="이 절단의 톱날 두께 손실")
    parent_node_id: str = Field(description="이 절단이 수행된 부모 노드 ID")


class PlacedPartOut(BaseModel):
    """
    하나의 배치 결과.
    어떤 원장에, 어떤 좌표·크기로, 어떤 절단 순서로 배치됐는지를 포함합니다.
    """
    node_id: str = Field(description="최종 배치 노드 ID")
    stock_id: str = Field(description="이 부품이 배치된 원장 ID")

    part_id: str = Field(description="배치된 부품 ID")
    placed_dims: DimsOut = Field(description="실제 배치된 방향의 치수 (회전 반영)")
    origin: OriginOut = Field(description="원장 기준 절대 좌표 (trimming 오프셋 포함)")

    cut_history: List[CutRecordOut] = Field(
        description=(
            "이 노드를 만들기까지 수행된 절단 이력 (루트→리프 순서). "
            "각 Cut의 axis + position으로 CNC/실제 절단 지시서 생성 가능."
        )
    )
    depth: int = Field(description="트리에서의 깊이 (절단 횟수)")


class StockSummaryOut(BaseModel):
    """원장 한 장의 사용 요약."""
    stock_id: str
    original_dims: DimsOut
    usable_dims: DimsOut
    placed_count: int = Field(description="이 원장에 배치된 부품 수")
    placed_volume: float
    usable_volume: float
    efficiency_pct: float = Field(description="배치 부피 / 사용 가능 부피 × 100")


class OptimizeResponse(BaseModel):
    """
    POST /optimize 전체 응답.

    placements: 배치된 모든 부품과 절단 이력
    unplaced   : 배치되지 못한 부품 (id → 남은 수량)
    summary    : 원장별 효율 요약
    stats      : 전체 통계
    """
    placements: List[PlacedPartOut]
    unplaced: Dict[str, int] = Field(
        description="배치 실패 부품 (id → 남은 수량). 비어있으면 전량 배치 성공."
    )
    stock_summaries: List[StockSummaryOut]
    stats: Dict[str, Any] = Field(description="전체 통계 (총 배치 수, 효율, 처리 시간 등)")


class ErrorResponse(BaseModel):
    """에러 응답 공통 스키마."""
    error: str
    detail: Optional[str] = None
    error_code: str


# ══════════════════════════════════════════════════════════════════
# 4. 도메인 변환 헬퍼 (Request → Core 객체, Core → Response)
# ══════════════════════════════════════════════════════════════════

def _build_engine_settings(s: SettingsIn) -> EngineSettings:
    return EngineSettings(
        kerf=s.kerf,
        trimming=TrimmingMargins(x=s.trimming.x, y=s.trimming.y, z=s.trimming.z),
        optimization_goal=OptimizationGoal[s.optimization_goal],
    )


def _build_stocks(stocks_in: List[StockIn], trimming: TrimmingMargins) -> List[Stock]:
    return [
        Stock(
            id=s.id,
            dims=Dims(l=s.l, w=s.w, t=s.t),
            qty=s.qty,
            trimming=trimming,
        )
        for s in stocks_in
    ]


def _build_parts(parts_in: List[PartIn]) -> List[Part]:
    return [
        Part(
            id=p.id,
            dims=Dims(l=p.l, w=p.w, t=p.t),
            qty=p.qty,
            lock_z=p.lock_z,
            allow_xy_rotation=p.allow_xy_rotation,
            priority=p.priority,
        )
        for p in parts_in
    ]


def _dims_to_out(d: Dims) -> DimsOut:
    return DimsOut(l=d.l, w=d.w, t=d.t, volume=d.volume)


def _node_to_placed_out(node: Node) -> PlacedPartOut:
    """OCCUPIED Node 하나를 PlacedPartOut으로 변환합니다."""
    assert node.placed_part is not None
    assert node.placed_part_dims is not None

    # 절단 이력: core의 collect_cut_history() 활용
    history = [
        CutRecordOut(
            cut_id=cut.cut_id,
            axis=cut.axis.value,
            position=cut.position,
            kerf=cut.kerf,
            parent_node_id=cut.parent_node_id,
        )
        for cut in node.collect_cut_history()
    ]

    return PlacedPartOut(
        node_id=node.node_id,
        stock_id=node.stock_id or "unknown",
        part_id=node.placed_part.id,
        placed_dims=_dims_to_out(node.placed_part_dims),
        origin=OriginOut(x=node.origin.x, y=node.origin.y, z=node.origin.z),
        cut_history=history,
        depth=node.depth,
    )


def _build_stock_summaries(
    occupied_nodes: List[Node],
    stocks: List[Stock],
) -> List[StockSummaryOut]:
    """원장별 배치 요약을 집계합니다."""
    # stock_id → 배치 노드 목록
    by_stock: Dict[str, List[Node]] = {}
    for node in occupied_nodes:
        sid = node.stock_id or "unknown"
        by_stock.setdefault(sid, []).append(node)

    stock_map = {s.id: s for s in stocks}
    summaries = []

    for sid, nodes in by_stock.items():
        stock = stock_map.get(sid)
        if stock is None:
            continue

        placed_vol = sum(
            n.placed_part_dims.volume for n in nodes if n.placed_part_dims
        )
        usable_vol = stock.usable_volume
        efficiency = (placed_vol / usable_vol * 100) if usable_vol > 0 else 0.0

        summaries.append(
            StockSummaryOut(
                stock_id=sid,
                original_dims=_dims_to_out(stock.dims),
                usable_dims=_dims_to_out(stock.usable_dims),
                placed_count=len(nodes),
                placed_volume=round(placed_vol, 4),
                usable_volume=round(usable_vol, 4),
                efficiency_pct=round(efficiency, 2),
            )
        )

    return sorted(summaries, key=lambda s: s.stock_id)


# ══════════════════════════════════════════════════════════════════
# 5. CPU-bound 작업을 동기 함수로 래핑
#    (run_in_threadpool에 전달할 동기 callable)
# ══════════════════════════════════════════════════════════════════

def _run_packing(
    settings: EngineSettings,
    stocks: List[Stock],
    parts: List[Part],
) -> tuple[List[Node], Dict[str, int]]:
    """
    pack_parts()를 실행하고 (occupied_nodes, remaining) 튜플을 반환합니다.

    이 함수는 동기(sync)이며, run_in_threadpool()을 통해
    별도 스레드에서 실행되어 FastAPI 이벤트 루프를 블로킹하지 않습니다.
    """
    # remaining 추적을 위해 dict 복사
    remaining: Dict[str, int] = {p.id: p.qty for p in parts}

    occupied = pack_parts(settings=settings, stocks=stocks, parts=parts)

    # 실제 배치된 수량을 차감하여 잔여량 계산
    for node in occupied:
        if node.placed_part:
            remaining[node.placed_part.id] = max(
                0, remaining.get(node.placed_part.id, 0) - 1
            )

    return occupied, {pid: qty for pid, qty in remaining.items() if qty > 0}


# ══════════════════════════════════════════════════════════════════
# 6. FastAPI 앱 초기화
# ══════════════════════════════════════════════════════════════════

app = FastAPI(
    title="3D Guillotine Cutting Optimization Engine",
    description=(
        "목재, 철강, 스폰지 등 다양한 재질의 3D 자재 재단을 최적화하는 API.\n\n"
        "**핵심 제약**\n"
        "- Strict Guillotine Cut (완전 관통 절단만 허용)\n"
        "- Kerf Deduction (절단면마다 톱날 두께만큼 손실)\n"
        "- Initial Trimming Margins (원장 테두리 여백)\n"
        "- Orientation Lock (Z축 고정 + XY 회전만 허용)\n\n"
        "**비동기 처리**: CPU-bound 최적화 연산은 ThreadPool에서 실행됩니다."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS 설정 (모바일 앱 및 웹 프론트엔드 지원)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 프로덕션에서는 구체적 도메인으로 제한
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ══════════════════════════════════════════════════════════════════
# 7. 전역 예외 핸들러
# ══════════════════════════════════════════════════════════════════

@app.exception_handler(InvalidCutError)
async def invalid_cut_handler(request: Request, exc: InvalidCutError) -> JSONResponse:
    """
    [C1][C2] Guillotine Cut 물리 제약 위반 시 400 반환.
    (kerf 설정 오류, position 범위 초과 등)
    """
    logger.warning("InvalidCutError: %s", str(exc))
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Invalid cut parameters",
            detail=str(exc),
            error_code="INVALID_CUT",
        ).model_dump(),
    )


@app.exception_handler(CuttingError)
async def cutting_error_handler(request: Request, exc: CuttingError) -> JSONResponse:
    """기타 절단 엔진 오류 시 422 반환."""
    logger.error("CuttingError: %s", str(exc))
    return JSONResponse(
        status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
        content=ErrorResponse(
            error="Cutting engine error",
            detail=str(exc),
            error_code="CUTTING_ERROR",
        ).model_dump(),
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
    """EngineSettings, Dims 생성 시 값 오류."""
    logger.warning("ValueError: %s", str(exc))
    return JSONResponse(
        status_code=status.HTTP_400_BAD_REQUEST,
        content=ErrorResponse(
            error="Invalid input value",
            detail=str(exc),
            error_code="VALUE_ERROR",
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
    """예상치 못한 서버 오류 시 500 반환 (스택 트레이스 숨김)."""
    logger.exception("Unexpected error on %s", request.url)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=ErrorResponse(
            error="Internal server error",
            detail="서버 내부 오류가 발생했습니다. 관리자에게 문의하세요.",
            error_code="INTERNAL_ERROR",
        ).model_dump(),
    )


# ══════════════════════════════════════════════════════════════════
# 8. 엔드포인트
# ══════════════════════════════════════════════════════════════════

@app.get(
    "/health",
    tags=["System"],
    summary="서버 상태 확인",
    response_description="서버가 정상 동작 중이면 200 OK",
)
async def health_check() -> Dict[str, str]:
    """헬스 체크 엔드포인트. 로드 밸런서 / k8s probe에 사용합니다."""
    return {"status": "ok", "engine": "3D Guillotine Cutting v1.0.0"}


@app.post(
    "/optimize",
    response_model=OptimizeResponse,
    status_code=status.HTTP_200_OK,
    tags=["Optimization"],
    summary="3D 자재 재단 최적화 실행",
    response_description="배치 결과, 절단 이력, 원장별 효율 통계",
    responses={
        400: {"model": ErrorResponse, "description": "입력값 오류 또는 절단 제약 위반"},
        422: {"model": ErrorResponse, "description": "엔진 처리 오류"},
        500: {"model": ErrorResponse, "description": "서버 내부 오류"},
    },
)
async def optimize(body: OptimizeRequest) -> OptimizeResponse:
    """
    ## 3D Guillotine Cutting 최적화 API

    주어진 **원장(Stock)** 목록과 **부품(Part)** 목록을 받아,
    4가지 물리 제약을 준수하는 최적 재단 계획을 반환합니다.

    ### 물리 제약 (자동 적용)
    - **[C1] Strict Guillotine Cut**: 모든 절단은 완전 관통 평면 절단
    - **[C2] Kerf Deduction**: 절단면마다 `settings.kerf`만큼 손실
    - **[C3] Trimming Margins**: 원장 테두리 `settings.trimming`만큼 제거
    - **[C4] Orientation Lock**: `lock_z=true`이면 Z축 고정, XY 평면 회전만 허용

    ### 응답에서 확인할 수 있는 정보
    - `placements[].origin`: 각 부품의 원장 기준 절대 좌표
    - `placements[].placed_dims`: 실제 배치된 방향의 치수 (회전 반영)
    - `placements[].cut_history`: 루트→리프 순서의 절단 이력 (CNC 지시서 생성 가능)
    - `stock_summaries[].efficiency_pct`: 원장별 공간 효율 (%)
    - `unplaced`: 배치 실패 부품 (Stock 부족 시)

    ### 비동기 처리
    CPU-intensive 최적화 연산은 `run_in_threadpool`을 통해
    별도 스레드에서 실행되어 서버 이벤트 루프를 블로킹하지 않습니다.
    """
    t_start = time.perf_counter()
    logger.info(
        "Optimize request: %d stocks, %d part types",
        len(body.stocks),
        len(body.parts),
    )

    # ── 도메인 객체 변환 ─────────────────────────────────────────
    try:
        engine_settings = _build_engine_settings(body.settings)
        stocks = _build_stocks(body.stocks, engine_settings.trimming)
        parts = _build_parts(body.parts)
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"입력 데이터 변환 오류: {e}",
        ) from e

    # ── CPU-bound 작업을 ThreadPool에서 비동기 실행 ──────────────
    #
    #   run_in_threadpool(fn, *args)은 FastAPI가 제공하는 유틸리티로,
    #   동기 함수를 starlette의 ThreadPoolExecutor에서 실행합니다.
    #   → asyncio 이벤트 루프가 다른 요청을 계속 처리할 수 있습니다.
    #
    try:
        occupied_nodes, unplaced = await run_in_threadpool(
            functools.partial(_run_packing, engine_settings, stocks, parts)
        )
    except (InvalidCutError, CuttingError):
        raise  # 전역 핸들러에게 위임
    except Exception as e:
        logger.exception("pack_parts() failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"최적화 연산 중 오류: {e}",
        ) from e

    elapsed = time.perf_counter() - t_start
    logger.info(
        "Optimize done: %d placed, %d unplaced types, %.3fs",
        len(occupied_nodes),
        len(unplaced),
        elapsed,
    )

    # ── 응답 조립 ────────────────────────────────────────────────
    placements = [_node_to_placed_out(n) for n in occupied_nodes]
    stock_summaries = _build_stock_summaries(occupied_nodes, stocks)

    total_placed_vol = sum(p.placed_dims.volume for p in placements)
    # [수정] 합판 1장 부피가 아니라, 사용한 전체 장수(qty)를 곱해줌
    total_usable_vol = sum(s.usable_volume * s.qty for s in stocks)
    overall_efficiency = (
        (total_placed_vol / total_usable_vol * 100) if total_usable_vol > 0 else 0.0
    )

    stats: Dict[str, Any] = {
        "total_placed":          len(occupied_nodes),
        "total_unplaced_types":  len(unplaced),
        "total_placed_volume":   round(total_placed_vol, 4),
        "total_usable_volume":   round(total_usable_vol, 4),
        "overall_efficiency_pct": round(overall_efficiency, 2),
        "stocks_used":           len({n.stock_id for n in occupied_nodes}),
        "processing_time_sec":   round(elapsed, 4),
    }

    return OptimizeResponse(
        placements=placements,
        unplaced=unplaced,
        stock_summaries=stock_summaries,
        stats=stats,
    )