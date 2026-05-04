"""
Bootstrap — one command sets up the entire Phase 0 pipeline.

What it does:
  1. Verifies data is in place
  2. Loads + temporal-splits Elliptic
  3. Trains the CPU-friendly portion of the ensemble
  4. Saves trained artifacts
  5. Seeds the 4 RAG collections (typology + regulatory + bootstrapped case_history)
  6. Runs the routing rubric to validate the system
  7. Prints next steps

Usage:
    python -m scripts.bootstrap
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np

from src.evaluation.routing_rubric import evaluate_routing
from src.features.tabular import fit_tabular_pipeline, transform_tabular
from src.ingestion.elliptic_loader import EllipticLoader
from src.ingestion.temporal_splitter import TemporalSplitter
from src.knowledge.chromadb_client import ChromaDBClient
from src.knowledge.collections.case_history import bootstrap_from_elliptic
from src.knowledge.collections.regulatory import seed_regulatory
from src.knowledge.collections.typology_library import seed_typology_library
from src.pipeline import save_artifacts, train_pipeline
from src.utils.config import load_default_config, resolve_path
from src.utils.logging import configure_logging, get_logger

configure_logging("INFO")
logger = get_logger(__name__)


def banner(title: str) -> None:
    print()
    print("━" * 60)
    print(f"  {title}")
    print("━" * 60)


def main() -> int:
    cfg = load_default_config()
    data_dir = resolve_path(cfg["paths"]["data_raw"])

    banner("STEP 1/6  Verify data")
    expected = ["elliptic_txs_features.csv", "elliptic_txs_classes.csv",
                "elliptic_txs_edgelist.csv"]
    missing = [f for f in expected if not (data_dir / f).exists()]
    if missing:
        print(f"❌ Missing files in {data_dir}: {missing}")
        print("\nRun ONE of:")
        print("  • python -m scripts.generate_synthetic_data    (quick demo)")
        print("  • Download from Kaggle (see DATA_SETUP.md)")
        return 1
    is_synthetic = (data_dir / ".synthetic").exists()
    print(f"✓ Data found in {data_dir} ({'SYNTHETIC' if is_synthetic else 'REAL'})")

    banner("STEP 2/6  Load + temporal split")
    data = EllipticLoader(root=data_dir).load()
    splitter = TemporalSplitter()
    split = splitter.split(data)
    print(f"✓ Loaded {len(data.features)} transactions")
    print(f"  Train [t={split.train_range}]: {len(split.train_tx_ids)} samples")
    print(f"  Val   [t={split.val_range}]:   {len(split.val_tx_ids)} samples")
    print(f"  Test  [t={split.test_range}]:  {len(split.test_tx_ids)} samples")

    X_train_df, y_train = TemporalSplitter.get_features_and_labels(data, split.train_tx_ids)
    X_val_df, y_val = TemporalSplitter.get_features_and_labels(data, split.val_tx_ids)

    if len(X_train_df) == 0 or len(X_val_df) == 0:
        print("❌ Empty train or validation split — not enough labeled data.")
        return 1

    banner("STEP 3/6  Train CPU detectors")
    pipeline_x = fit_tabular_pipeline(X_train_df)
    X_train = transform_tabular(pipeline_x, X_train_df)
    X_val = transform_tabular(pipeline_x, X_val_df)
    feature_names = list(X_train_df.columns)
    print(f"  Feature dim: {X_train.shape[1]}")
    print(f"  Training Isolation Forest ...", flush=True)
    artifacts = train_pipeline(X_train, y_train, X_val, y_val, feature_names)

    # Auto-configure ensemble weights for whichever detectors are available.
    # This makes the demo "just work" even with only IF installed.
    from src.detection.ensemble import EnsembleWeights
    n = len(artifacts.available_models)
    weights_kwargs = {
        "w_xgboost": 1.0 if "xgboost" in artifacts.available_models else 0.0,
        "w_graphsage": 1.0 if "graphsage" in artifacts.available_models else 0.0,
        "w_isolation_forest": 1.0 if "isolation_forest" in artifacts.available_models else 0.0,
        "w_autoencoder": 1.0 if "autoencoder" in artifacts.available_models else 0.0,
        # Inactive detectors get unreachable thresholds so they never flag
        "t_xgboost": 0.5 if "xgboost" in artifacts.available_models else 0.99,
        "t_graphsage": 0.5 if "graphsage" in artifacts.available_models else 0.99,
        "t_isolation_forest": 0.6 if "isolation_forest" in artifacts.available_models else 0.99,
        "t_autoencoder": 0.6 if "autoencoder" in artifacts.available_models else 0.99,
    }
    artifacts.ensemble_weights = EnsembleWeights(**weights_kwargs)
    print(f"✓ Trained: {', '.join(artifacts.available_models)}")

    artifact_path = save_artifacts(artifacts)
    print(f"✓ Saved to: {artifact_path.relative_to(resolve_path('.'))}")

    banner("STEP 4/6  Seed RAG knowledge base")
    chroma = ChromaDBClient(persist_directory=resolve_path(cfg["paths"]["chromadb"]))
    n_typo = seed_typology_library(chroma)
    n_reg = seed_regulatory(chroma)
    # Bootstrap case_history with up to 50 illicit transactions from training
    label_lookup = data.classes.set_index('txId')['label']
    illicit_ids = [
        str(tx) for tx in split.train_tx_ids
        if label_lookup.get(str(tx)) == 1
    ][:50]
    print(f"  Found {len(illicit_ids)} illicit tx IDs for case_history bootstrap")
    n_cases = bootstrap_from_elliptic(chroma, illicit_ids)

    print(f"✓ Knowledge base seeded:")
    print(f"  typology_library:   {n_typo}")
    print(f"  regulatory:         {n_reg}")
    print(f"  case_history:       {n_cases} (bootstrapped)")
    print(f"  personal_notes:     0  (grows from confirmed cases)")

    banner("STEP 5/6  Validate routing logic on 50 eval cases")
    eval_result = evaluate_routing()
    pct = eval_result.accuracy * 100
    print(f"  Total: {eval_result.total}, correct: {eval_result.correct}, "
          f"accuracy: {pct:.1f}%")
    print(f"  Per-tier: {eval_result.per_tier_accuracy}")
    if eval_result.accuracy < 1.0:
        print(f"  Mismatches: {len(eval_result.mismatches)}")

    banner("STEP 6/6  Done")
    print("✓ Bootstrap complete.")
    print()
    print("Next steps:")
    print("  1. Run the API:        uvicorn src.api.main:app --reload")
    print("  2. Run the UI:         streamlit run src/ui/app.py")
    print("  3. Optional: training the GraphSAGE model in Colab "
          "(see notebooks/02_train_gnn.ipynb)")
    print()

    # Save a small summary file for the UI's home page
    summary = {
        "is_synthetic": is_synthetic,
        "trained_at": artifacts.trained_at,
        "n_train_samples": artifacts.n_train_samples,
        "available_models": artifacts.available_models,
        "rag_counts": {
            "typology_library": n_typo,
            "regulatory": n_reg,
            "case_history": n_cases,
            "personal_notes": 0,
        },
        "routing_accuracy_pct": pct,
    }
    summary_path = resolve_path("artifacts/bootstrap_summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
