"""
Mistral API client — Tier 3 primary + fallback 1.

Per locked decision: Mistral Small 3.1 primary, Mistral Medium 3 fallback.
Free tier: 1B tokens/month.
"""
from __future__ import annotations
import os
from typing import Optional

import httpx

from src.reasoning.llm_abstraction import LLMProvider, LLMResponse
from src.utils.logging import get_logger

logger = get_logger(__name__)


class MistralProvider(LLMProvider):
    """Talks to Mistral's chat completions API."""

    BASE_URL = "https://api.mistral.ai/v1/chat/completions"

    def __init__(self, model: str = "mistral-small-latest",
                 api_key: Optional[str] = None, timeout_s: float = 30.0):
        self.model = model
        self.api_key = api_key or os.environ.get("MISTRAL_API_KEY")
        self.timeout_s = timeout_s
        self.name = f"mistral:{model}"

    def complete(self, system: str, user: str, max_tokens: int = 1024,
                 temperature: float = 0.1) -> LLMResponse:
        if not self.api_key:
            return LLMResponse(
                text="", provider="mistral", model=self.model,
                error="MISTRAL_API_KEY not set",
            )
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                r = client.post(self.BASE_URL, headers=headers, json=payload)
                r.raise_for_status()
                data = r.json()
                text = data["choices"][0]["message"]["content"]
                return LLMResponse(text=text, provider="mistral", model=self.model)
        except Exception as e:
            logger.warning("mistral_call_failed", model=self.model, error=str(e))
            return LLMResponse(text="", provider="mistral", model=self.model, error=str(e))
