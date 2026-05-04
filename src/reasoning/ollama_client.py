"""
Ollama client — local SLM/LLM provider.

Used for Tier 2 (qwen2.5:1.5b) and as Tier 3 fallback (qwen2.5:3b).
Runs entirely on CPU. No API key required.
"""
from __future__ import annotations
import os
from typing import Optional

import httpx

from src.reasoning.llm_abstraction import LLMProvider, LLMResponse
from src.utils.logging import get_logger

logger = get_logger(__name__)


class OllamaProvider(LLMProvider):
    """Talks to a local Ollama server via HTTP."""

    def __init__(self, model: str = "qwen2.5:1.5b", base_url: Optional[str] = None,
                 timeout_s: float = 60.0):
        self.model = model
        self.base_url = base_url or os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
        self.timeout_s = timeout_s
        self.name = f"ollama:{model}"

    def complete(self, system: str, user: str, max_tokens: int = 1024,
                 temperature: float = 0.1) -> LLMResponse:
        url = f"{self.base_url.rstrip('/')}/api/chat"
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        try:
            with httpx.Client(timeout=self.timeout_s) as client:
                r = client.post(url, json=payload)
                r.raise_for_status()
                data = r.json()
                text = (data.get("message") or {}).get("content", "") or data.get("response", "")
                return LLMResponse(text=text, provider="ollama", model=self.model)
        except Exception as e:
            logger.warning("ollama_call_failed", model=self.model, error=str(e))
            return LLMResponse(text="", provider="ollama", model=self.model, error=str(e))
