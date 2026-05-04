"""
Regulatory Knowledge — curated text from FATF, OFAC, FinCEN.

Static collection. Quarterly batch updates. Used to make explanations
regulator-ready by citing the relevant compliance frame.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

from src.knowledge.chromadb_client import ChromaDBClient


COLLECTION_NAME = "regulatory"


@dataclass
class RegulatoryEntry:
    id: str
    source: str
    topic: str
    text: str
    applies_to: List[str] = field(default_factory=list)

    def to_embedding_text(self) -> str:
        return f"{self.source} {self.topic} {self.text}"

    def to_metadata(self) -> Dict:
        return {
            "source": self.source,
            "topic": self.topic,
            "applies_to": ";".join(self.applies_to),
        }


SEED_REGULATIONS: List[RegulatoryEntry] = [
    RegulatoryEntry(
        id="REG-FATF-R10",
        source="FATF Recommendation 10",
        topic="customer_due_diligence",
        text=(
            "Financial institutions should undertake CDD measures when establishing "
            "business relations and when carrying out occasional transactions above "
            "the designated threshold (USD/EUR 15,000)."
        ),
        applies_to=["all_jurisdictions"],
    ),
    RegulatoryEntry(
        id="REG-FATF-R11",
        source="FATF Recommendation 11",
        topic="record_keeping",
        text=(
            "Financial institutions should maintain all necessary records on transactions, "
            "both domestic and international, for at least five years."
        ),
        applies_to=["all_jurisdictions"],
    ),
    RegulatoryEntry(
        id="REG-FATF-VASP",
        source="FATF Virtual Asset Service Provider Guidance",
        topic="vasp_obligations",
        text=(
            "VASPs must implement Travel Rule requirements: collect and transmit "
            "originator and beneficiary information for crypto transfers above thresholds."
        ),
        applies_to=["all_jurisdictions"],
    ),
    RegulatoryEntry(
        id="REG-OFAC-MIXERS",
        source="OFAC Sanctions — Mixers",
        topic="sanctioned_mixers",
        text=(
            "OFAC has sanctioned mixers including Tornado Cash and Blender.io. "
            "U.S. persons are prohibited from transactions involving sanctioned addresses."
        ),
        applies_to=["us_jurisdiction"],
    ),
    RegulatoryEntry(
        id="REG-FINCEN-RANSOM",
        source="FinCEN Advisory 2020-A006",
        topic="ransomware_payments",
        text=(
            "Financial institutions facilitating ransomware payments may be subject to "
            "OFAC enforcement if payments go to sanctioned cyber actors."
        ),
        applies_to=["us_jurisdiction"],
    ),
    RegulatoryEntry(
        id="REG-FATF-R13",
        source="FATF Recommendation 13",
        topic="correspondent_banking",
        text=(
            "Financial institutions in correspondent relationships should gather sufficient "
            "information about respondent institutions to understand the nature of business."
        ),
        applies_to=["all_jurisdictions"],
    ),
    RegulatoryEntry(
        id="REG-EU-MICA",
        source="EU MiCA Regulation",
        topic="crypto_asset_service_providers",
        text=(
            "EU's Markets in Crypto-Assets (MiCA) imposes licensing, governance, and "
            "consumer-protection obligations on crypto asset service providers."
        ),
        applies_to=["eu_jurisdiction"],
    ),
    RegulatoryEntry(
        id="REG-INDIA-PMLA",
        source="India PMLA — Crypto Inclusion 2023",
        topic="crypto_under_aml",
        text=(
            "India's Prevention of Money Laundering Act now covers virtual digital asset "
            "operators, requiring KYC and reporting to the Financial Intelligence Unit."
        ),
        applies_to=["india_jurisdiction"],
    ),
]


def seed_regulatory(client: ChromaDBClient) -> int:
    """Seed regulatory collection. Idempotent."""
    n = 0
    for r in SEED_REGULATIONS:
        try:
            client.add(
                collection_name=COLLECTION_NAME,
                doc_id=r.id,
                text=r.to_embedding_text(),
                metadata=r.to_metadata(),
            )
            n += 1
        except Exception:
            pass
    return n
