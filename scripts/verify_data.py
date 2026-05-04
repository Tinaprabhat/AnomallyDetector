"""
Verify that data is properly placed and loadable.

Usage:  python -m scripts.verify_data
"""
from __future__ import annotations
from pathlib import Path

from src.ingestion.elliptic_loader import EllipticLoader
from src.utils.config import load_default_config, resolve_path


def main() -> int:
    cfg = load_default_config()
    data_dir = resolve_path(cfg["paths"]["data_raw"])
    print(f"Checking data at: {data_dir}")

    expected = [
        "elliptic_txs_features.csv",
        "elliptic_txs_classes.csv",
        "elliptic_txs_edgelist.csv",
    ]
    missing = [f for f in expected if not (data_dir / f).exists()]
    if missing:
        print(f"❌ Missing files: {missing}")
        print()
        print("Run ONE of:")
        print("  Real data:      Download from https://www.kaggle.com/datasets/ellipticco/elliptic-data-set")
        print("  Synthetic demo: python -m scripts.generate_synthetic_data")
        return 1

    is_synthetic = (data_dir / ".synthetic").exists()
    label = "SYNTHETIC" if is_synthetic else "REAL"
    print(f"  Mode: {label}")

    print("  Loading...")
    try:
        data = EllipticLoader(root=data_dir).load()
    except Exception as e:
        print(f"❌ Load failed: {e}")
        return 1

    n_illicit = int((data.classes["label"] == 1).sum())
    n_licit = int((data.classes["label"] == 0).sum())
    n_unknown = int((data.classes["label"] == -1).sum())

    print(f"✓ Data loaded successfully:")
    print(f"  • Transactions:  {len(data.features)}")
    print(f"  • Features dim:  {data.features.shape[1] - 1}")  # exclude time_step
    print(f"  • Time steps:    {data.time_steps.min()}-{data.time_steps.max()}")
    print(f"  • Illicit:       {n_illicit}")
    print(f"  • Licit:         {n_licit}")
    print(f"  • Unknown:       {n_unknown}")
    print(f"  • Edges:         {len(data.edges)}")
    print(f"  • Graph nodes:   {data.graph.number_of_nodes()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
