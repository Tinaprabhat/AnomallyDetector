"""
RAG Curator — handles the feedback loop into the knowledge base.

Per locked design:
- ONLY confirmed cases enter case_history.
- After confirmation, LLM self-reflection generates a personal_notes entry.
- Rejected cases go to a separate false-positive log (NOT into RAG).
"""
from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

from src.knowledge.chromadb_client import ChromaDBClient
from src.knowledge.collections.case_history import CaseEntry, add_confirmed_case
from src.knowledge.collections.personal_notes import (
    PersonalNote, add_note, make_note_from_reflection,
)
from src.reasoning.llm_abstraction import LLMProvider
from src.reasoning.self_reflection import reflect_on_confirmed_case
from src.utils.logging import get_logger

logger = get_logger(__name__)


class FalsePositiveLog:
    """SQLite log of rejected cases — used for retraining, NOT for RAG."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS false_positives (
        transaction_id TEXT PRIMARY KEY,
        rejected_at TEXT NOT NULL,
        analyst_notes TEXT,
        detection_payload TEXT
    );
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as c:
            c.executescript(self.SCHEMA)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def log_rejection(self, tx_id: str, analyst_notes: str, payload: Dict) -> None:
        with self._conn() as c:
            c.execute(
                """INSERT OR REPLACE INTO false_positives
                   (transaction_id, rejected_at, analyst_notes, detection_payload)
                   VALUES (?, ?, ?, ?)""",
                (tx_id, datetime.utcnow().isoformat(), analyst_notes, json.dumps(payload)),
            )


class RAGCurator:
    """Orchestrates the feedback loop into RAG."""

    def __init__(
        self,
        chroma_client: ChromaDBClient,
        fp_log: FalsePositiveLog,
        llm_provider: Optional[LLMProvider] = None,
    ):
        self.chroma = chroma_client
        self.fp_log = fp_log
        self.llm_provider = llm_provider

    def on_confirmed(
        self,
        transaction_id: str,
        tier: str,
        detection_payload: Dict,
        analyst_notes: str,
        typology_assigned: str,
    ) -> Dict[str, bool]:
        """
        Handle a confirmed case:
          1. Add to case_history
          2. Run LLM self-reflection (if provider available) → personal_notes
        """
        result = {"case_history_added": False, "personal_note_added": False}

        # 1) case_history
        case_id = f"CASE-{transaction_id}-{int(datetime.utcnow().timestamp())}"
        ml_evidence = detection_payload.get("ml_evidence", {})
        case = CaseEntry(
            id=case_id,
            transaction_id=transaction_id,
            detected_at=datetime.utcnow().isoformat(),
            typology_assigned=typology_assigned or "unspecified",
            analyst_decision="confirmed_fraud",
            analyst_notes=analyst_notes,
            ml_evidence={k: float(v) for k, v in ml_evidence.items()
                         if isinstance(v, (int, float))},
            outcome=f"tier={tier}",
        )
        try:
            add_confirmed_case(self.chroma, case)
            result["case_history_added"] = True
        except Exception as e:
            logger.warning("case_history_add_failed", error=str(e))

        # 2) personal_notes via self-reflection
        if self.llm_provider is None:
            return result

        original_explanation = detection_payload.get("narrative_explanation", "")
        rag_summary = ", ".join(detection_payload.get("rag_citations", []))

        try:
            reflection = reflect_on_confirmed_case(
                provider=self.llm_provider,
                transaction_id=transaction_id,
                typology=typology_assigned,
                ml_evidence=ml_evidence,
                original_rag_summary=rag_summary,
                original_explanation=original_explanation,
                analyst_notes=analyst_notes,
            )
        except Exception as e:
            logger.warning("self_reflection_failed", error=str(e))
            reflection = None

        if reflection is not None:
            note = make_note_from_reflection(case_id, reflection.model_dump())
            try:
                add_note(self.chroma, note)
                result["personal_note_added"] = True
            except Exception as e:
                logger.warning("personal_note_add_failed", error=str(e))

        return result

    def on_rejected(
        self, transaction_id: str, analyst_notes: str, detection_payload: Dict,
    ) -> None:
        """Log rejection — does NOT enter RAG."""
        self.fp_log.log_rejection(transaction_id, analyst_notes, detection_payload)
