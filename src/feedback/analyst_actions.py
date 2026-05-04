"""
Analyst actions — high-level operations exposed to the UI/API for handling
items in the review queue.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict

from src.feedback.rag_curator import RAGCurator
from src.feedback.review_queue import ReviewQueue


@dataclass
class AnalystAction:
    transaction_id: str
    decision: str           # "confirmed" | "rejected"
    typology_assigned: str  # used only for confirmed
    notes: str = ""


class AnalystOrchestrator:
    """Wraps queue + curator so a single 'apply' call closes the loop."""

    def __init__(self, queue: ReviewQueue, curator: RAGCurator):
        self.queue = queue
        self.curator = curator

    def apply(self, action: AnalystAction) -> Dict:
        item = self.queue.get(action.transaction_id)
        if item is None:
            return {"ok": False, "error": "transaction not in queue"}

        if action.decision == "confirmed":
            self.queue.resolve(
                action.transaction_id, "confirmed",
                analyst_notes=action.notes, typology_assigned=action.typology_assigned,
            )
            curated = self.curator.on_confirmed(
                transaction_id=action.transaction_id,
                tier=item.tier,
                detection_payload=item.detection_payload,
                analyst_notes=action.notes,
                typology_assigned=action.typology_assigned,
            )
            return {"ok": True, "curated": curated}

        if action.decision == "rejected":
            self.queue.resolve(
                action.transaction_id, "rejected", analyst_notes=action.notes,
            )
            self.curator.on_rejected(
                transaction_id=action.transaction_id,
                analyst_notes=action.notes,
                detection_payload=item.detection_payload,
            )
            return {"ok": True, "logged_as_false_positive": True}

        return {"ok": False, "error": f"unknown decision: {action.decision}"}
