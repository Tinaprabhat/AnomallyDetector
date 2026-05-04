"""Shared pytest fixtures for unit tests."""
from __future__ import annotations
import sys
from pathlib import Path

# Make src importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pytest

from src.ingestion.elliptic_loader import EllipticLoader, EllipticData


@pytest.fixture(scope="session")
def synthetic_data() -> EllipticData:
    """Synthetic Elliptic-shaped dataset for tests."""
    loader = EllipticLoader(root="nonexistent")
    return loader.load_synthetic(n_transactions=500, seed=42)


@pytest.fixture(scope="session")
def small_synthetic_data() -> EllipticData:
    """Smaller synthetic dataset for fast tests."""
    loader = EllipticLoader(root="nonexistent")
    return loader.load_synthetic(n_transactions=100, seed=7)


@pytest.fixture
def random_features():
    """Random feature matrix for tests that need raw arrays."""
    rng = np.random.default_rng(42)
    return rng.standard_normal((200, 50))


@pytest.fixture
def random_labels():
    """Imbalanced labels matching fraud distribution."""
    rng = np.random.default_rng(42)
    return rng.choice([0, 1], size=200, p=[0.95, 0.05])
