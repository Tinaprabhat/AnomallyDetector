"""
Review Queue — SQLite-backed queue for LOW + AMBIGUOUS cases needing human review.
"""
from __future__ import annotations
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class ReviewItem:
    transaction_id: str
    tier: str
    detection_payload: Dict
    status: str = "pending"      # pending | confirmed | rejected
    analyst_notes: str = ""
    typology_assigned: str = ""
    created_at: str = ""
    resolved_at: Optional[str] = None


class ReviewQueue:
    """Persistent queue for cases requiring analyst review."""

    SCHEMA = """
    CREATE TABLE IF NOT EXISTS review_queue (
        transaction_id TEXT PRIMARY KEY,
        tier TEXT NOT NULL,
        detection_payload TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        analyst_notes TEXT DEFAULT '',
        typology_assigned TEXT DEFAULT '',
        created_at TEXT NOT NULL,
        resolved_at TEXT
    );
    CREATE INDEX IF NOT EXISTS idx_review_status ON review_queue(status);
    """

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def _init_db(self) -> None:
        with self._conn() as c:
            c.executescript(self.SCHEMA)

    def enqueue(self, item: ReviewItem) -> None:
        with self._conn() as c:
            c.execute(
                """
                INSERT OR REPLACE INTO review_queue
                (transaction_id, tier, detection_payload, status, analyst_notes,
                 typology_assigned, created_at, resolved_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.transaction_id, item.tier, json.dumps(item.detection_payload),
                    item.status, item.analyst_notes, item.typology_assigned,
                    item.created_at or datetime.utcnow().isoformat(), item.resolved_at,
                ),
            )

    def list_pending(self, limit: int = 100) -> List[ReviewItem]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT * FROM review_queue WHERE status='pending' "
                "ORDER BY created_at ASC LIMIT ?", (limit,),
            ).fetchall()
        return [self._row_to_item(r) for r in rows]

    def get(self, tx_id: str) -> Optional[ReviewItem]:
        with self._conn() as c:
            row = c.execute(
                "SELECT * FROM review_queue WHERE transaction_id=?", (tx_id,)
            ).fetchone()
        return self._row_to_item(row) if row else None

    def resolve(
        self, tx_id: str, status: str, analyst_notes: str = "",
        typology_assigned: str = "",
    ) -> bool:
        if status not in ("confirmed", "rejected"):
            raise ValueError(f"Invalid status: {status}")
        with self._conn() as c:
            cur = c.execute(
                """UPDATE review_queue
                   SET status=?, analyst_notes=?, typology_assigned=?, resolved_at=?
                   WHERE transaction_id=?""",
                (status, analyst_notes, typology_assigned,
                 datetime.utcnow().isoformat(), tx_id),
            )
            return cur.rowcount > 0

    def stats(self) -> Dict[str, int]:
        with self._conn() as c:
            rows = c.execute(
                "SELECT status, COUNT(*) as n FROM review_queue GROUP BY status"
            ).fetchall()
        return {r["status"]: int(r["n"]) for r in rows}

    @staticmethod
    def _row_to_item(row) -> ReviewItem:
        return ReviewItem(
            transaction_id=row["transaction_id"],
            tier=row["tier"],
            detection_payload=json.loads(row["detection_payload"]),
            status=row["status"],
            analyst_notes=row["analyst_notes"] or "",
            typology_assigned=row["typology_assigned"] or "",
            created_at=row["created_at"],
            resolved_at=row["resolved_at"],
        )
