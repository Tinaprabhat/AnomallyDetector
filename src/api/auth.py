"""Optional API-key auth — disabled by default for local dev."""
from __future__ import annotations
import os
from fastapi import Header, HTTPException


def require_api_key(x_api_key: str | None = Header(default=None)) -> None:
    """If ANOMALY_API_KEY env var is set, require it as X-API-Key header."""
    expected = os.environ.get("ANOMALY_API_KEY")
    if expected and x_api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")
