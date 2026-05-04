"""
Provider-agnostic LLM abstraction.

Per locked decision:
- Tier 2: Ollama qwen2.5:1.5b (local SLM)
- Tier 3 primary: Mistral Small 3.1
- Tier 3 fallback 1: Mistral Medium 3
- Tier 3 fallback 2: Ollama qwen2.5:3b

This module provides one interface (.complete) that all providers implement,
with automatic fallback chains.
"""
from __future__ import annotations
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from src.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    text: str
    provider: str
    model: str
    error: Optional[str] = None


class LLMProvider(ABC):
    name: str = "abstract"
    model: str = "abstract"

    @abstractmethod
    def complete(self, system: str, user: str, max_tokens: int = 1024,
                 temperature: float = 0.1) -> LLMResponse:
        ...


class FallbackChain:
    """Try providers in order; return first success."""

    def __init__(self, providers: List[LLMProvider]):
        if not providers:
            raise ValueError("FallbackChain requires at least one provider")
        self.providers = providers

    def complete(self, system: str, user: str, max_tokens: int = 1024,
                 temperature: float = 0.1) -> LLMResponse:
        last_error = None
        for p in self.providers:
            try:
                resp = p.complete(system, user, max_tokens, temperature)
                if resp.error is None and resp.text:
                    return resp
                last_error = resp.error
            except Exception as e:
                last_error = str(e)
                logger.warning("provider_failed", provider=p.name, error=last_error)
                continue
        return LLMResponse(
            text="",
            provider="fallback_chain",
            model="none",
            error=f"All providers failed; last_error={last_error}",
        )


def parse_json_response(text: str) -> Optional[dict]:
    """Robust JSON parsing — strips common wrappers like ```json fences."""
    if not text:
        return None
    s = text.strip()
    # Strip markdown fences
    if s.startswith("```"):
        s = s.split("```", 2)[1]
        if s.lstrip().startswith("json"):
            s = s.lstrip()[4:]
        s = s.strip()
    if s.endswith("```"):
        s = s.rsplit("```", 1)[0].strip()

    # Try direct parse
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass

    # Try extracting the first {...} block
    start = s.find("{")
    end = s.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(s[start: end + 1])
        except json.JSONDecodeError:
            return None
    return None
