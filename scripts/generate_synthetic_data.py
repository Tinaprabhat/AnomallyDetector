"""
Generate synthetic Elliptic-shaped CSVs for demos and quick-start.

Same shape as real data:
- 165 features per transaction
- Time steps 1-49
- ~2% illicit, ~20% licit, ~78% unknown (matches Elliptic distribution)

Output goes to data/raw/elliptic_bitcoin_dataset/ — same path as real data,
so all downstream scripts work identically.

Usage:
    python -m scripts.generate_synthetic_data
    python -m scripts.generate_synthetic_data --n-tx 5000
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.utils.config import resolve_path, load_default_config


def generate(n_transactions: int, seed: int, out_dir: Path) -> None:
    rng = np.random.default_rng(seed)
    n_features = 165

    out_dir.mkdir(parents=True, exist_ok=True)

    # Transaction IDs as numeric (Elliptic uses string-numerical IDs)
    tx_ids = [str(100000 + i) for i in range(n_transactions)]
    time_steps = rng.integers(1, 50, size=n_transactions)

    # Features: ~Gaussian, but inject signal — illicit transactions get a shift on a few feats
    labels_raw = rng.choice([0, 1, 2], size=n_transactions, p=[0.78, 0.20, 0.02])
    # 0 = unknown (encoded as "unknown"), 1 = licit, 2 = illicit (Elliptic original encoding)

    feats = rng.standard_normal((n_transactions, n_features)).astype(np.float32)
    illicit_mask = labels_raw == 2
    # Make illicit transactions slightly shifted on first 8 features (so detectors have signal)
    feats[illicit_mask, :8] += rng.normal(2.0, 0.5, size=(int(illicit_mask.sum()), 8))

    # Features CSV: txId, time_step, feat_0..feat_164  (no header — matches Elliptic)
    feat_df = pd.DataFrame(
        np.column_stack([np.array(tx_ids, dtype=object),
                         time_steps.astype(int),
                         feats]),
    )
    features_path = out_dir / "elliptic_txs_features.csv"
    feat_df.to_csv(features_path, header=False, index=False)

    # Classes CSV: txId, label  (with header — Elliptic v1 format)
    label_str = np.where(labels_raw == 0, "unknown",
                np.where(labels_raw == 1, "1", "2"))
    cls_df = pd.DataFrame({"txId": tx_ids, "class": label_str})
    classes_path = out_dir / "elliptic_txs_classes.csv"
    cls_df.to_csv(classes_path, index=False)

    # Edges CSV: txId1, txId2 (with header)
    n_edges = n_transactions * 2
    src = rng.choice(tx_ids, size=n_edges)
    dst = rng.choice(tx_ids, size=n_edges)
    edge_df = pd.DataFrame({"txId1": src, "txId2": dst})
    edge_df = edge_df[edge_df["txId1"] != edge_df["txId2"]].reset_index(drop=True)
    edges_path = out_dir / "elliptic_txs_edgelist.csv"
    edge_df.to_csv(edges_path, index=False)

    # Marker so other scripts can detect synthetic data
    (out_dir / ".synthetic").write_text(
        f"synthetic data\nn_transactions={n_transactions}\nseed={seed}\n"
    )

    print(f"✓ Synthetic Elliptic-shaped data written to: {out_dir}")
    print(f"  • {features_path.name}: {features_path.stat().st_size // 1024} KB")
    print(f"  • {classes_path.name}: {classes_path.stat().st_size // 1024} KB")
    print(f"  • {edges_path.name}: {edges_path.stat().st_size // 1024} KB")
    print(f"  • Transactions: {n_transactions}")
    print(f"  • Illicit:  {int(illicit_mask.sum())} (~{illicit_mask.mean()*100:.1f}%)")
    print(f"  • Licit:    {int((labels_raw == 1).sum())}")
    print(f"  • Unknown:  {int((labels_raw == 0).sum())}")
    print()
    print("Next step:  python -m scripts.bootstrap")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic Elliptic-shaped data")
    parser.add_argument("--n-tx", type=int, default=2000,
                        help="Number of transactions (default 2000)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = load_default_config()
    out_dir = resolve_path(cfg["paths"]["data_raw"])
    generate(args.n_tx, args.seed, out_dir)


if __name__ == "__main__":
    main()
