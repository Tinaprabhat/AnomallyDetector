"""
Shared helpers for Tier 2 (SLM) and Tier 3 (LLM) explainers.
"""
from __future__ import annotations
from typing import Dict, List, Optional

from src.knowledge.chromadb_client import RetrievedDoc


def format_rag_context(docs: List[RetrievedDoc]) -> Dict[str, str]:
    """Format retrieved docs into named sections for the prompt template."""
    sections = {
        "typology_options": [],
        "case_history": [],
        "regulations": [],
        "personal_notes": [],
    }
    for d in docs:
        if d.collection == "typology_library":
            sections["typology_options"].append(
                f"  - id={d.id} name={d.metadata.get('name')} (sim={d.similarity:.2f}): {d.text}"
            )
        elif d.collection == "case_history":
            sections["case_history"].append(
                f"  - id={d.id} typology={d.metadata.get('typology_assigned')} "
                f"(sim={d.similarity:.2f}): {d.text}"
            )
        elif d.collection == "regulatory":
            sections["regulations"].append(
                f"  - id={d.id} source={d.metadata.get('source')} (sim={d.similarity:.2f}): {d.text}"
            )
        elif d.collection == "personal_notes":
            sections["personal_notes"].append(
                f"  - id={d.id} topic={d.metadata.get('topic')} (sim={d.similarity:.2f}): {d.text}"
            )

    return {
        "typology_options": "\n".join(sections["typology_options"]) or "  (none)",
        "case_history": "\n".join(sections["case_history"]) or "  (none)",
        "regulations": "\n".join(sections["regulations"]) or "  (none)",
        "personal_notes": "\n".join(sections["personal_notes"]) or "  (none)",
    }
