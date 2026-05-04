"""POST /backtest — trigger a backtest run."""
from __future__ import annotations
from fastapi import APIRouter

from src.api.schemas import BacktestRequest

router = APIRouter()


@router.post("/backtest")
def run_backtest(req: BacktestRequest):
    """Run rolling-window backtests. Returns summary metrics."""
    return {
        "status": "scheduled",
        "message": (
            "Backtest harness available in src/backtesting/. "
            "Phase 1 will wire this endpoint to a trained ensemble."
        ),
        "config": req.model_dump(),
    }
