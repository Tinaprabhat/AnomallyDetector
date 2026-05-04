"""Configuration loader — single source of truth for all config files."""
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict
import os

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"


def load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_default_config() -> Dict[str, Any]:
    return load_yaml(CONFIG_DIR / "default.yaml")


def load_routing_config() -> Dict[str, Any]:
    return load_yaml(CONFIG_DIR / "routing.yaml")


def load_ensemble_config() -> Dict[str, Any]:
    return load_yaml(CONFIG_DIR / "ensemble_weights.yaml")


def load_prompts_config() -> Dict[str, Any]:
    return load_yaml(CONFIG_DIR / "prompts.yaml")


def get_env(key: str, default: str | None = None) -> str | None:
    return os.environ.get(key, default)


def project_root() -> Path:
    return PROJECT_ROOT


def resolve_path(relative: str | Path) -> Path:
    """Resolve a relative path to absolute, anchored at project root."""
    p = Path(relative)
    if p.is_absolute():
        return p
    return PROJECT_ROOT / p
