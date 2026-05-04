# Data Setup Guide

The Elliptic Bitcoin Dataset is **~600MB** — too large to ship in a code repo.
Follow ONE of the two paths below.

---

## Option A — Download Real Elliptic Dataset (Recommended)

1. Create a free account at https://www.kaggle.com (if you don't have one).
2. Visit the dataset page:
   **https://www.kaggle.com/datasets/ellipticco/elliptic-data-set**
3. Click **Download** (~600MB zip).
4. Extract the zip.
5. Place the three CSVs at the path Claude expects:

```
AnomalyDetector_v2/
└── data/
    └── raw/
        └── elliptic_bitcoin_dataset/
            ├── elliptic_txs_features.csv
            ├── elliptic_txs_classes.csv
            └── elliptic_txs_edgelist.csv
```

That's it. Every script, the API, and the UI will pick this up automatically because the path is wired in `config/default.yaml`:

```yaml
paths:
  data_raw: data/raw/elliptic_bitcoin_dataset
```

To verify:

```powershell
python -m scripts.verify_data
```

---

## Option B — Generate Synthetic Data (For Demo / Quick Start)

If you want to see the full pipeline working before downloading the real dataset, run:

```powershell
python -m scripts.generate_synthetic_data
```

This creates synthetic Elliptic-shaped CSVs at the correct path. Same shape as real data (165 features, time steps 1-49, ~2% illicit). **Useful for:** UI walkthroughs, smoke-testing the pipeline, presentation demos. **Not useful for:** publishable metrics, since the data is random.

The synthetic flag is automatically detected — every script knows whether you're running on real or synthetic data and labels reports accordingly.

---

## Initial Bootstrap (One Command)

After data is in place (real or synthetic), run:

```powershell
python -m scripts.bootstrap
```

This:
1. Verifies the data is readable
2. Seeds the 4 RAG collections (typology library, regulatory, etc.)
3. Trains the 3 CPU-friendly detectors (Isolation Forest, Autoencoder, XGBoost)
4. Saves trained models to `models/`
5. Computes ensemble weights via Optuna
6. Bootstraps `case_history` from known illicit transactions
7. Runs the evaluation rubric (50 cases, expected 100%)
8. Reports done

Total time: ~3-5 minutes on CPU for synthetic, ~10-15 minutes for real Elliptic.

---

## Why Not Just Ship The Data?

Three reasons:
- **GitHub blocks files >100MB** (the features CSV alone is ~250MB)
- **License compliance** — Elliptic's terms require users get the data from Kaggle
- **Reproducibility** — anyone with internet can get the same data, no zip-mangling risk

This matches the v1 pattern (your README said the same).
