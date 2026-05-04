"""Review Queue page — analyst processes LOW/AMBIGUOUS cases."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.feedback.analyst_actions import AnalystAction, AnalystOrchestrator
from src.feedback.rag_curator import FalsePositiveLog, RAGCurator
from src.feedback.review_queue import ReviewQueue
from src.knowledge.chromadb_client import ChromaDBClient
from src.knowledge.collections.typology_library import SEED_TYPOLOGIES
from src.utils.config import load_default_config, resolve_path

st.title("2️⃣ Review Queue")
st.caption("Process LOW / MEDIUM / AMBIGUOUS cases — confirm or reject. "
           "Confirmed cases enter case_history (and trigger LLM self-reflection if Mistral/Ollama configured).")

cfg = load_default_config()
queue = ReviewQueue(db_path=resolve_path(cfg["paths"]["audit_db"]))
chroma = ChromaDBClient(persist_directory=resolve_path(cfg["paths"]["chromadb"]))
fp_log = FalsePositiveLog(db_path=resolve_path("artifacts/false_positives.db"))
curator = RAGCurator(chroma_client=chroma, fp_log=fp_log, llm_provider=None)
orchestrator = AnalystOrchestrator(queue=queue, curator=curator)

# --- Stats -----------------------------------------------------------------
stats = queue.stats()
cols = st.columns(3)
cols[0].metric("Pending", stats.get("pending", 0))
cols[1].metric("Confirmed", stats.get("confirmed", 0))
cols[2].metric("Rejected", stats.get("rejected", 0))

st.divider()
st.subheader("Pending cases")

items = queue.list_pending(limit=50)
if not items:
    st.info("Queue is empty. Submit transactions on the **Detection** page to populate it.")
    st.stop()

# Typology dropdown options for analyst
typology_options = ["(no_match)"] + sorted(t.name for t in SEED_TYPOLOGIES)

for item in items:
    payload = item.detection_payload
    title = (
        f"**{item.transaction_id}**  ·  tier=`{item.tier}`  ·  "
        f"score={payload.get('ml_evidence', {}).get('ensemble_score', 0):.3f}"
    )
    with st.expander(title):
        narrative = payload.get("narrative_explanation", "")
        if narrative:
            st.info(narrative)

        col_evidence, col_decision = st.columns([2, 1])
        with col_evidence:
            st.subheader("Evidence")
            st.write("**ML scores:**")
            st.json(payload.get("ml_evidence", {}))
            st.write("**Statistical evidence:**")
            st.json(payload.get("statistical_evidence", {}))
            if payload.get("rag_citations"):
                st.caption(f"RAG citations: {', '.join(payload['rag_citations'])}")

        with col_decision:
            st.subheader("Decision")
            typology = st.selectbox(
                "Typology", typology_options, key=f"typo_{item.transaction_id}",
            )
            notes = st.text_area("Notes", key=f"note_{item.transaction_id}", height=120)

            c1, c2 = st.columns(2)
            if c1.button("✅ Confirm", key=f"ok_{item.transaction_id}",
                         use_container_width=True):
                action = AnalystAction(
                    transaction_id=item.transaction_id,
                    decision="confirmed",
                    typology_assigned=(typology if typology != "(no_match)" else "no_match"),
                    notes=notes,
                )
                result = orchestrator.apply(action)
                if result.get("ok"):
                    st.success(f"Confirmed → {result.get('curated', {})}")
                    st.rerun()
                else:
                    st.error(f"Failed: {result}")

            if c2.button("❌ Reject", key=f"no_{item.transaction_id}",
                         use_container_width=True):
                action = AnalystAction(
                    transaction_id=item.transaction_id,
                    decision="rejected",
                    typology_assigned="",
                    notes=notes,
                )
                result = orchestrator.apply(action)
                if result.get("ok"):
                    st.success("Rejected — logged to false-positive log "
                               "(NOT added to RAG by design).")
                    st.rerun()
                else:
                    st.error(f"Failed: {result}")
