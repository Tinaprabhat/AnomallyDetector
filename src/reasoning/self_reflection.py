"""
LLM Self-Reflection — generates personal_notes after a confirmed case.

Per locked design (your insight): when an analyst confirms a flagged case,
the LLM studies the full case context (tx, ML evidence, what RAG retrieved,
what LLM originally said, analyst notes) and extracts a structured "key lesson"
for use in future LLM calls.

The note is saved in personal_notes; the full case is saved in case_history.
"""
from __future__ import annotations
import json
from typing import Any, Dict, Optional

from src.reasoning.llm_abstraction import LLMProvider, parse_json_response
from src.reasoning.schemas import SelfReflectionOutput
from src.utils.config import load_prompts_config
from src.utils.logging import get_logger

logger = get_logger(__name__)


def reflect_on_confirmed_case(
    provider: LLMProvider,
    transaction_id: str,
    typology: str,
    ml_evidence: Dict[str, Any],
    original_rag_summary: str,
    original_explanation: str,
    analyst_notes: str,
) -> Optional[SelfReflectionOutput]:
    """
    Run LLM self-reflection on a confirmed case.
    Returns a SelfReflectionOutput, or None if reflection failed.
    """
    prompts = load_prompts_config()
    user_prompt = prompts["self_reflection_prompt"].format(
        transaction_id=transaction_id,
        typology=typology,
        ml_evidence=json.dumps(ml_evidence, default=str),
        original_rag=original_rag_summary,
        original_explanation=original_explanation,
        analyst_notes=analyst_notes,
    )

    resp = provider.complete(
        system=prompts["system_prompt"],
        user=user_prompt,
        max_tokens=512,
        temperature=0.2,
    )
    payload = parse_json_response(resp.text)
    if not payload:
        logger.warning("self_reflection_no_json", transaction_id=transaction_id)
        return None

    try:
        return SelfReflectionOutput(
            topic=str(payload.get("topic", "general")),
            key_lesson=str(payload.get("key_lesson", "")),
            indicators_to_watch=list(payload.get("indicators_to_watch") or []),
            related_typologies=list(payload.get("related_typologies") or []),
        )
    except Exception as e:
        logger.warning("self_reflection_validation_failed", error=str(e))
        return None
