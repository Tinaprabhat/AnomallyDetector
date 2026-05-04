"""Knowledge Base page — browse + search + seed the 4 RAG collections."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.knowledge.chromadb_client import ChromaDBClient
from src.knowledge.collections.regulatory import seed_regulatory
from src.knowledge.collections.typology_library import seed_typology_library
from src.utils.config import load_default_config, resolve_path

st.title("3️⃣ Knowledge Base")
st.caption("Browse and search the 4 RAG collections.")

cfg = load_default_config()
chroma = ChromaDBClient(persist_directory=resolve_path(cfg["paths"]["chromadb"]))

collections = ["typology_library", "case_history", "regulatory", "personal_notes"]
counts = {c: chroma.count(c) for c in collections}

cols = st.columns(4)
for c, name in zip(cols, collections):
    c.metric(name, counts[name])

# Seed action — useful when KB is empty
if counts["typology_library"] == 0 or counts["regulatory"] == 0:
    st.warning("Some collections are empty.")
    if st.button("Seed typology + regulatory (idempotent)"):
        n_t = seed_typology_library(chroma)
        n_r = seed_regulatory(chroma)
        st.success(f"Seeded — typology_library: +{n_t}, regulatory: +{n_r}")
        st.rerun()

st.divider()
st.subheader("🔎 Search")

c1, c2, c3 = st.columns([2, 4, 1])
collection = c1.selectbox("Collection", collections)
q = c2.text_input("Query", value="rapid layering across multiple addresses")
k = c3.slider("Top K", 1, 10, 5)

if st.button("Search"):
    docs = chroma.query(collection_name=collection, query_text=q, n_results=k)
    if not docs:
        st.warning("No results — collection may be empty.")
    else:
        st.caption(f"Found {len(docs)} results")
        for d in docs:
            sim_pct = int(d.similarity * 100)
            with st.expander(f"{d.id}  ·  similarity {sim_pct}%"):
                st.write(d.text)
                st.json(d.metadata)
