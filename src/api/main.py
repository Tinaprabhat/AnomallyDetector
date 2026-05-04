"""
AnomalyDetector v2 — FastAPI service entry point.

Run with: uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
"""
from __future__ import annotations
from fastapi import FastAPI

from src.api.routers import detect, explain, backtest, knowledge_base, feedback
from src.utils.logging import configure_logging

configure_logging()

app = FastAPI(
    title="AnomalyDetector v2",
    description=(
        "Production-grade fraud detection prototype with LLM-augmented explanation. "
        "4-model ML ensemble + statistical validation + 4-collection RAG + tiered routing."
    ),
    version="2.0.0",
)

app.include_router(detect.router, tags=["detect"])
app.include_router(explain.router, tags=["explain"])
app.include_router(backtest.router, tags=["backtest"])
app.include_router(knowledge_base.router, tags=["knowledge_base"])
app.include_router(feedback.router, tags=["feedback"])


@app.get("/")
def root():
    return {
        "service": "AnomalyDetector v2",
        "version": "2.0.0",
        "endpoints": ["/detect", "/explain/{tx_id}", "/backtest",
                      "/knowledge_base", "/feedback"],
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok"}
