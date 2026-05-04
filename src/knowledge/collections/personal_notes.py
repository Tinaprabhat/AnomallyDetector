"""
Personal Notes — LLM-generated reflective notes after confirmed cases.

Per locked decision (your design insight): LLM studies each confirmed case and
extracts a structured "key lesson" that future LLM calls can use to reason
about similar cases. This is "self-reflection memory" — the system grows
its own meta-knowledge.

Notes can be edited or deleted by the analyst via the UI. The analyst's actual
decision is stored in case_history; this collection holds the *meta-lessons*.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List

from src.knowledge.chromadb_client import ChromaDBClient


COLLECTION_NAME = "personal_notes"


@dataclass
class PersonalNote:
    id: str
    source_case_id: str
    created_at: str
    topic: str
    key_lesson: str
    indicators_to_watch: List[str] = field(default_factory=list)
    related_typologies: List[str] = field(default_factory=list)

    def to_embedding_text(self) -> str:
        return (
            f"{self.topic}: {self.key_lesson} "
            f"watch for: {', '.join(self.indicators_to_watch)} "
            f"applies to: {', '.join(self.related_typologies)}"
        )

    def to_metadata(self) -> Dict:
        return {
            "source_case_id": self.source_case_id,
            "created_at": self.created_at,
            "topic": self.topic,
            "indicators_to_watch": ";".join(self.indicators_to_watch),
            "related_typologies": ";".join(self.related_typologies),
        }


def add_note(client: ChromaDBClient, note: PersonalNote) -> None:
    client.add(
        collection_name=COLLECTION_NAME,
        doc_id=note.id,
        text=note.to_embedding_text(),
        metadata=note.to_metadata(),
    )


def make_note_from_reflection(
    source_case_id: str,
    reflection_payload: Dict,
) -> PersonalNote:
    """Construct a PersonalNote from the LLM's self-reflection JSON output."""
    now = datetime.utcnow().isoformat()
    return PersonalNote(
        id=f"NOTE-{source_case_id}-{int(datetime.utcnow().timestamp())}",
        source_case_id=source_case_id,
        created_at=now,
        topic=str(reflection_payload.get("topic", "general")),
        key_lesson=str(reflection_payload.get("key_lesson", "")),
        indicators_to_watch=list(reflection_payload.get("indicators_to_watch") or []),
        related_typologies=list(reflection_payload.get("related_typologies") or []),
    )
