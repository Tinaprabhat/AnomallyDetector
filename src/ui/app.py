"""
AnomalyDetector v2 — Streamlit Dashboard.

Run with: streamlit run src/ui/app.py
"""
import json
import sys
from pathlib import Path

# Make src importable when run directly
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.knowledge.chromadb_client import ChromaDBClient
from src.pipeline import load_artifacts
from src.utils.config import load_default_config, resolve_path

st.set_page_config(
    page_title="AnomalyDetector v2",
    page_icon="🔍",
    layout="wide",
)

cfg = load_default_config()


def _summary_path() -> Path:
    return resolve_path("artifacts/bootstrap_summary.json")


def render_home() -> None:
    st.title("🔍 AnomalyDetector v2")
    st.caption("Production-grade fraud detection prototype with LLM-augmented explanation")

    artifacts = load_artifacts()
    summary_file = _summary_path()
    bootstrap_done = summary_file.exists() and artifacts is not None

    if not bootstrap_done:
        st.warning("⚠️  Pipeline not yet bootstrapped. See setup steps below.")
    else:
        st.success("✓ Pipeline ready")

    cols = st.columns(4)
    cols[0].metric(
        "Pipeline trained",
        "Yes" if artifacts else "No",
        delta=", ".join(artifacts.available_models) if artifacts else None,
        delta_color="off",
    )

    chroma = ChromaDBClient(persist_directory=resolve_path(cfg["paths"]["chromadb"]))
    cols[1].metric("Typologies", chroma.count("typology_library"))
    cols[2].metric("Case history", chroma.count("case_history"))
    cols[3].metric("Personal notes", chroma.count("personal_notes"))

    st.divider()

    st.subheader("Quick start")
    st.markdown(
        """
1. **Get data** — see `DATA_SETUP.md`.
   For a quick demo: `python -m scripts.generate_synthetic_data`

2. **Bootstrap the pipeline** (one command):
   ```
   python -m scripts.bootstrap
   ```
   This loads + splits the data, trains CPU-friendly detectors, seeds the 4 RAG collections,
   and runs the routing rubric (expects 100% on the 50 eval cases).

3. **Use the dashboard:**
   - **1 Detection** — submit transactions and view full pipeline output
   - **2 Review Queue** — process LOW/AMBIGUOUS cases (confirm or reject)
   - **3 Knowledge Base** — browse + search the 4 RAG collections, seed if needed
   - **4 Backtesting** — run rolling-window evaluation across 49 time steps
   - **5 Personal Notes** — view/edit LLM-distilled meta-lessons
"""
    )

    st.divider()
    st.subheader("System architecture (locked design)")
    st.markdown(
        """
- **Detection:** 4-model ensemble — XGBoost · GraphSAGE · Isolation Forest · Autoencoder
- **Validation:** Mahalanobis distance · SHAP top-K · Ensemble agreement
- **Routing:** 3-tier — Template / Ollama SLM / Mistral LLM
- **RAG:** 4 collections — typology_library · case_history · regulatory · personal_notes
- **LLM autonomy:** Level 2 (synth + classify within bounded typology library)
- **Hardware:** CPU-only inference; optional Colab for GraphSAGE training
"""
    )

    if bootstrap_done:
        try:
            data = json.loads(summary_file.read_text())
            with st.expander("Bootstrap summary"):
                st.json(data)
        except Exception:
            pass

    with st.sidebar:
        st.header("Status")
        if artifacts:
            st.success("Pipeline: ready")
            st.caption(f"Trained: {artifacts.trained_at}")
            st.caption(f"Samples: {artifacts.n_train_samples}")
            st.caption(f"Models: {', '.join(artifacts.available_models)}")
        else:
            st.warning("Pipeline: not trained")
            st.caption("Run `python -m scripts.bootstrap`")


render_home()
