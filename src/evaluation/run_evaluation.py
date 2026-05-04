"""
Evaluation runner — runs all rubrics and produces a JSON report.

Usage:
    python -m src.evaluation.run_evaluation
"""
from __future__ import annotations
import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Dict

from src.evaluation.eval_cases import EVAL_CASES
from src.evaluation.routing_rubric import evaluate_routing
from src.utils.config import load_default_config, resolve_path
from src.utils.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)


def run_full_evaluation() -> Dict:
    """Run all available rubrics and return a combined report."""
    report = {
        "timestamp": datetime.utcnow().isoformat(),
        "n_eval_cases": len(EVAL_CASES),
        "tier_distribution": {
            tier: sum(1 for c in EVAL_CASES if c.expected_tier == tier)
            for tier in ["PASS", "HIGH", "MEDIUM", "LOW", "AMBIGUOUS"]
        },
    }

    # Routing rubric
    logger.info("evaluation_routing_start")
    routing_result = evaluate_routing()
    report["routing"] = {
        "total": routing_result.total,
        "correct": routing_result.correct,
        "accuracy": routing_result.accuracy,
        "per_tier_accuracy": routing_result.per_tier_accuracy,
        "confusion": routing_result.confusion,
        "n_mismatches": len(routing_result.mismatches),
        "mismatches": routing_result.mismatches,
    }
    logger.info("evaluation_routing_done", accuracy=routing_result.accuracy)

    # LLM rubric — placeholder until trained pipeline + LLM are wired
    report["llm_rubric"] = {
        "status": "not_run",
        "note": (
            "LLM rubric requires trained models + Mistral API key + Ollama. "
            "Run `python -m src.evaluation.run_evaluation` after training to populate."
        ),
    }

    return report


def save_report(report: Dict) -> Path:
    cfg = load_default_config()
    out_dir = resolve_path(cfg["paths"]["reports"]) / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"eval_report_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    return out_path


if __name__ == "__main__":
    report = run_full_evaluation()
    out_path = save_report(report)
    print(f"Evaluation report saved to: {out_path}")
    print(json.dumps({k: v for k, v in report.items() if k != "routing"}, indent=2))
    print(f"\nRouting accuracy: {report['routing']['accuracy']:.2%}")
    print(f"Mismatches: {report['routing']['n_mismatches']}")
