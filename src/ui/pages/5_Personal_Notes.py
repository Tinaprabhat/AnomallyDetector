"""Personal Notes page — view, search, manually add, or delete LLM-generated reflective notes."""
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.knowledge.chromadb_client import ChromaDBClient
from src.knowledge.collections.personal_notes import (
    COLLECTION_NAME, PersonalNote, add_note,
)
from src.utils.config import load_default_config, resolve_path

st.title("5️⃣ Personal Notes")
st.caption(
    "LLM-distilled reflective notes (auto-generated after analyst-confirmed cases). "
    "You can also manually add notes here."
)

cfg = load_default_config()
chroma = ChromaDBClient(persist_directory=resolve_path(cfg["paths"]["chromadb"]))
n_notes = chroma.count(COLLECTION_NAME)
st.metric("Total notes", n_notes)

st.markdown(
    """
**About this collection:**
- Notes are **LLM-generated** after a human-confirmed case (per locked design decision).
- The LLM studies the full case context and extracts a *meta-lesson* for future reasoning.
- These notes get retrieved on every flagged case (weight 1.2×) to help the LLM apply past learnings.
- You can edit (re-add with same id) or delete notes at any time.
"""
)

st.divider()

st.subheader("🔎 Search notes")
q = st.text_input("Query", value="false positive arbitrage")
k = st.slider("Top K", 1, 10, 5)
if st.button("Search"):
    docs = chroma.query(collection_name=COLLECTION_NAME, query_text=q, n_results=k)
    if not docs:
        st.info("No notes found yet. Notes are generated automatically when "
                "analysts confirm cases (Tier 2/3 with Mistral/Ollama configured), "
                "or you can add one manually below.")
    else:
        for d in docs:
            with st.expander(f"{d.id}  ·  similarity {int(d.similarity * 100)}%"):
                st.write(d.text)
                st.json(d.metadata)

st.divider()

with st.expander("✍️ Add a note manually"):
    topic = st.text_input("Topic", value="rapid_layering_detection")
    lesson = st.text_area(
        "Key lesson (1-2 sentences)",
        value="High fan-out + rapid sequential timing strongly indicates peel-chain laundering.",
    )
    indicators = st.text_input(
        "Indicators (comma separated)",
        value="fan_out>10, time_delta<60s, decreasing_amounts",
    )
    related = st.text_input("Related typologies (comma separated)",
                            value="peel_chain, layering")
    if st.button("Save note"):
        note = PersonalNote(
            id=f"NOTE-MANUAL-{int(datetime.utcnow().timestamp())}",
            source_case_id="manual",
            created_at=datetime.utcnow().isoformat(),
            topic=topic,
            key_lesson=lesson,
            indicators_to_watch=[s.strip() for s in indicators.split(",") if s.strip()],
            related_typologies=[s.strip() for s in related.split(",") if s.strip()],
        )
        add_note(chroma, note)
        st.success(f"Saved as `{note.id}`")
        st.rerun()
