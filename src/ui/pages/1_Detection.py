"""Detection page — submit a transaction, see the full pipeline output."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import streamlit as st

from src.feedback.review_queue import ReviewQueue
from src.ingestion.elliptic_loader import EllipticLoader
from src.ingestion.temporal_splitter import TemporalSplitter
from src.knowledge.chromadb_client import ChromaDBClient
from src.pipeline import detect_transaction, load_artifacts
from src.utils.config import load_default_config, resolve_path

st.title("1️⃣ Detection")
st.caption("Submit a transaction — runs the full ML + statistical + RAG + reasoning pipeline.")

cfg = load_default_config()
artifacts = load_artifacts()

if artifacts is None:
    st.error("❌ Pipeline not yet bootstrapped. Run `python -m scripts.bootstrap` first.")
    st.stop()

st.success(f"✓ Pipeline ready — models: {', '.join(artifacts.available_models)}")
expected_dim = len(artifacts.feature_names)


# --- Helpers ---------------------------------------------------------------

@st.cache_data(show_spinner=False)
def _load_test_pool() -> dict:
    """Load test-split tx_ids + feature rows for the 'pick a test transaction' picker."""
    try:
        data = EllipticLoader(root=resolve_path(cfg["paths"]["data_raw"])).load()
    except Exception:
        return {"available": False}
    split = TemporalSplitter().split(data)
    test_ids = [str(t) for t in split.test_tx_ids[:200]]
    rows = {}
    for tx in test_ids:
        if tx in data.features.index:
            row = data.features.loc[tx].drop("time_step").values.tolist()
            ts = int(data.features.loc[tx, "time_step"])
            label = int(data.classes.set_index("txId").loc[tx, "label"])
            rows[tx] = {"features": row, "time_step": ts, "label": label}
    return {"available": True, "rows": rows}


pool = _load_test_pool()

# --- Input section ---------------------------------------------------------

st.subheader("Choose input")
mode = st.radio(
    "Source",
    options=["Pick a test transaction", "Random feature vector", "Manual entry"],
    horizontal=True,
)

if mode == "Pick a test transaction":
    if not pool["available"]:
        st.warning("Real data not loaded. Run `python -m scripts.generate_synthetic_data` "
                   "first or download Elliptic data.")
        st.stop()
    tx_options = list(pool["rows"].keys())
    if not tx_options:
        st.warning("No test transactions available.")
        st.stop()
    tx_id = st.selectbox("Test transaction", tx_options)
    row = pool["rows"][tx_id]
    raw_features = row["features"]
    time_step = row["time_step"]
    label_str = {1: "illicit", 0: "licit", -1: "unknown"}[row["label"]]
    st.caption(f"Ground-truth label: **{label_str}** • time_step={time_step}")

elif mode == "Random feature vector":
    seed = st.number_input("Seed", min_value=0, value=42)
    rng = np.random.default_rng(int(seed))
    raw_features = rng.standard_normal(expected_dim).tolist()
    tx_id = f"random_{int(seed)}"
    time_step = int(st.slider("Time step", 1, 49, 42))

else:  # Manual entry
    tx_id = st.text_input("Transaction ID", value="tx_demo_001")
    time_step = int(st.number_input("Time step", min_value=1, max_value=49, value=42))
    raw_str = st.text_area(
        f"Raw features (comma-separated, {expected_dim} floats)",
        value=", ".join(["0.0"] * expected_dim),
        height=120,
    )
    try:
        raw_features = [float(x.strip()) for x in raw_str.split(",")]
    except ValueError:
        st.error("Could not parse features as floats.")
        st.stop()

# --- Run pipeline ----------------------------------------------------------

if st.button("Run detection", type="primary"):
    with st.spinner("Running pipeline..."):
        chroma = ChromaDBClient(persist_directory=resolve_path(cfg["paths"]["chromadb"]))
        queue = ReviewQueue(db_path=resolve_path(cfg["paths"]["audit_db"]))
        result = detect_transaction(
            transaction_id=tx_id,
            raw_features=raw_features,
            time_step=time_step,
            artifacts=artifacts,
            chroma_client=chroma,
            review_queue=queue,
        )

    # --- Display result -----------------------------------------------------
    tier = result.confidence_tier
    tier_color = {
        "PASS": "🟢", "HIGH": "🔴", "MEDIUM": "🟠",
        "LOW": "🟡", "AMBIGUOUS": "🟣",
    }.get(tier, "⚪")
    st.markdown(f"### {tier_color} **{result.verdict}** — tier **{tier}**")

    cols = st.columns(4)
    cols[0].metric("Ensemble score", f"{result.ml_evidence['ensemble_score']:.3f}")
    cols[1].metric("Agreement", f"{int(result.ml_evidence['agreement_count'])}/4")
    cols[2].metric("Mahalanobis D²",
                   f"{result.statistical_evidence['mahalanobis_distance']:.2f}")
    cols[3].metric("Percentile",
                   f"{result.statistical_evidence['ensemble_percentile']:.1f}")

    st.subheader("Per-detector scores")
    st.bar_chart(
        {
            "score": {
                "XGBoost": result.ml_evidence["p_xgb"],
                "GraphSAGE": result.ml_evidence["p_gnn"],
                "Isolation Forest": result.ml_evidence["s_iso"],
                "Autoencoder": result.ml_evidence["e_ae"],
            }
        }
    )

    st.subheader("Narrative explanation")
    st.info(result.narrative_explanation)

    cols2 = st.columns(2)
    with cols2[0]:
        st.subheader("Typology match")
        st.json(result.typology_match)
    with cols2[1]:
        st.subheader("Recommended action")
        st.write(f"**{result.recommended_action}**")
        if result.rag_citations:
            st.caption("RAG citations:")
            st.write(", ".join(result.rag_citations))

    with st.expander("Full audit trail"):
        st.json(result.audit_trail)
        st.subheader("Top SHAP-style features")
        st.write(result.statistical_evidence["shap_top_features"])

    if tier in ("MEDIUM", "LOW", "AMBIGUOUS"):
        st.success("Case added to review queue → check the **Review Queue** page.")
