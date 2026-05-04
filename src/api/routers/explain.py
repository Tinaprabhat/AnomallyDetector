"""GET /explain/{transaction_id} — retrieve a past detection result."""
from __future__ import annotations
from fastapi import APIRouter, HTTPException

router = APIRouter()


@router.get("/explain/{transaction_id}")
def explain(transaction_id: str):
    """Return the stored detection payload for a given transaction id."""
    # Phase 0: stub — Phase 4 will wire to audit DB
    return {
        "transaction_id": transaction_id,
        "status": "not_implemented_in_phase_0",
        "message": "Will retrieve from audit log in Phase 4.",
    }
