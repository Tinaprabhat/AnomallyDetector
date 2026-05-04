"""Lightweight monitoring — latency, throughput, error tracking."""
from __future__ import annotations
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class StageTiming:
    """Latency for a single pipeline stage."""
    stage: str
    duration_ms: float
    timestamp: float = field(default_factory=time.time)


@dataclass
class PipelineMetrics:
    """Aggregate metrics for one pipeline run."""
    transaction_id: str
    stages: List[StageTiming] = field(default_factory=list)
    error: str | None = None
    tier: str | None = None

    def total_ms(self) -> float:
        return sum(s.duration_ms for s in self.stages)

    def to_dict(self) -> Dict:
        return {
            "transaction_id": self.transaction_id,
            "total_ms": self.total_ms(),
            "tier": self.tier,
            "error": self.error,
            "stages": {s.stage: s.duration_ms for s in self.stages},
        }


@contextmanager
def time_stage(metrics: PipelineMetrics, stage_name: str):
    """Context manager — measure a stage and append it to metrics."""
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        metrics.stages.append(StageTiming(stage=stage_name, duration_ms=elapsed_ms))
