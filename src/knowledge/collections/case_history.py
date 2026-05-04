"""
Case History — confirmed past anomalies with full context.

This collection auto-grows over time. Per locked decision, only HUMAN-CONFIRMED
cases enter (analyst confirms via review queue → curated entry added here).

Cold-start: bootstrap with ~500 known illicit Elliptic transactions.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from src.knowledge.chromadb_client import ChromaDBClient


COLLECTION_NAME = "case_history"


@dataclass
class CaseEntry:
    id: str
    transaction_id: str
    detected_at: str
    typology_assigned: str
    analyst_decision: str   # "confirmed_fraud" | "confirmed_licit"
    analyst_notes: str
    ml_evidence: Dict = field(default_factory=dict)
    outcome: str = ""

    def to_embedding_text(self) -> str:
        ev = " ".join(f"{k}={v}" for k, v in self.ml_evidence.items())
        return (
            f"{self.typology_assigned} {self.analyst_decision} "
            f"{self.analyst_notes} evidence: {ev}"
        )

    def to_metadata(self) -> Dict:
        return {
            "transaction_id": self.transaction_id,
            "detected_at": self.detected_at,
            "typology_assigned": self.typology_assigned,
            "analyst_decision": self.analyst_decision,
            "outcome": self.outcome,
        }


def add_confirmed_case(client: ChromaDBClient, case: CaseEntry) -> None:
    """Add a single confirmed case to the case_history collection."""
    client.add(
        collection_name=COLLECTION_NAME,
        doc_id=case.id,
        text=case.to_embedding_text(),
        metadata=case.to_metadata(),
    )


def bootstrap_from_elliptic(
    client: ChromaDBClient,
    illicit_tx_ids: List[str],
    max_entries: int = 500,
) -> int:
    """
    Cold-start: bootstrap case_history from known illicit transactions.

    Per locked decision — gives the system memory to draw from on day one.
    Uses batch embedding for speed: one encode() call for all entries.
    """
    now = datetime.utcnow().isoformat()
    cases = [
        CaseEntry(
            id=f"BOOT-{tx}",
            transaction_id=str(tx),
            detected_at=now,
            typology_assigned="unknown_illicit",
            analyst_decision="confirmed_fraud",
            analyst_notes="Bootstrapped from Elliptic illicit label.",
            ml_evidence={"source": "elliptic_label"},
            outcome="historical_label",
        )
        for tx in illicit_tx_ids[:max_entries]
    ]
    if not cases:
        return 0
    try:
        client.add_batch(
            collection_name=COLLECTION_NAME,
            doc_ids=[c.id for c in cases],
            texts=[c.to_embedding_text() for c in cases],
            metadatas=[c.to_metadata() for c in cases],
        )
        return len(cases)
    except Exception:
        return 0
