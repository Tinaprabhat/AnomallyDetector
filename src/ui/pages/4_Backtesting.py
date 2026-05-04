"""Backtesting page — run rolling-window backtests and see results."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import streamlit as st

from src.backtesting.temporal_backtest import (
    BacktestWindow, TemporalBacktester, parse_windows,
)
from src.detection.isolation_forest import IFConfig, IsolationForestDetector
from src.features.tabular import fit_tabular_pipeline, transform_tabular
from src.ingestion.elliptic_loader import EllipticLoader
from src.ingestion.temporal_splitter import TemporalSplitter
from src.utils.config import load_default_config, resolve_path

st.title("4️⃣ Backtesting")
st.caption("Rolling-window evaluation across the 49 Elliptic time steps. "
           "No future leakage — train always precedes test.")

cfg = load_default_config()

# --- Window display --------------------------------------------------------

windows_cfg = cfg["backtesting"]["rolling_windows"]
df_windows = pd.DataFrame([
    {"#": i + 1,
     "train_start": w["train"][0], "train_end": w["train"][1],
     "test_start": w["test"][0], "test_end": w["test"][1]}
    for i, w in enumerate(windows_cfg)
])
st.subheader("Rolling windows")
st.dataframe(df_windows, use_container_width=True, hide_index=True)

c1, c2, _ = st.columns([1, 1, 2])
cost_fn = c1.number_input("Cost of false negative", value=100.0, min_value=1.0)
cost_fp = c2.number_input("Cost of false positive", value=1.0, min_value=0.1)

# --- Run -------------------------------------------------------------------

if st.button("Run backtest", type="primary"):
    with st.spinner("Loading data..."):
        try:
            data = EllipticLoader(root=resolve_path(cfg["paths"]["data_raw"])).load()
        except FileNotFoundError:
            st.error("No data found. Run `python -m scripts.generate_synthetic_data` "
                     "or place real Elliptic CSVs.")
            st.stop()

    time_steps = data.time_steps.to_dict()
    windows = parse_windows(windows_cfg)

    def train_score(train_ids: np.ndarray, test_ids: np.ndarray):
        """Train Isolation Forest on this train slice, score the test slice."""
        X_train_df, _ = TemporalSplitter.get_features_and_labels(data, train_ids)
        X_test_df, y_test = TemporalSplitter.get_features_and_labels(data, test_ids)
        if len(X_train_df) == 0 or len(X_test_df) == 0:
            return np.array([0]), np.array([0.5])
        pipeline = fit_tabular_pipeline(X_train_df)
        X_train = transform_tabular(pipeline, X_train_df)
        X_test = transform_tabular(pipeline, X_test_df)
        det = IsolationForestDetector(IFConfig(n_estimators=80)).fit(X_train, X_val=X_train)
        scores = det.score(X_test)
        return y_test, scores

    bt = TemporalBacktester(time_steps=time_steps, cost_fn=cost_fn, cost_fp=cost_fp)
    progress = st.progress(0.0, "Running windows...")
    report = bt.run(windows, train_score)
    progress.progress(1.0, "Done")

    st.subheader("Per-window metrics")
    rows = []
    for w in report.windows:
        m = w.metrics
        rows.append({
            "window": w.window.label(),
            "n_train": w.n_train, "n_test": w.n_test,
            "PR-AUC": round(m.pr_auc, 3),
            "ROC-AUC": round(m.roc_auc, 3),
            "P@5%": round(m.precision_at_5pct, 3),
            "P@10%": round(m.precision_at_10pct, 3),
            "F1*": round(m.f1_at_optimal_threshold, 3),
            "MCC*": round(m.mcc_at_optimal_threshold, 3),
            "cost": round(w.cost_breakdown.total_cost, 1) if w.cost_breakdown else None,
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    st.subheader("Summary")
    st.json(report.summary)

    # PR-AUC chart across windows
    if any(w.metrics.pr_auc > 0 for w in report.windows):
        st.subheader("PR-AUC across windows (drift indicator)")
        st.line_chart(df.set_index("window")["PR-AUC"])
