"""
Typology Library — expert-curated fraud typologies.

Static collection. Seeded with ~10 well-known patterns. New typologies added
manually after expert review (NOT auto-added by the LLM — Level 2 autonomy).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List

from src.knowledge.chromadb_client import ChromaDBClient


COLLECTION_NAME = "typology_library"


@dataclass
class Typology:
    id: str
    name: str
    category: str
    description: str
    indicators: List[str]
    graph_signature: str = ""
    typical_features: Dict[str, str] = field(default_factory=dict)
    regulatory_refs: List[str] = field(default_factory=list)

    def to_embedding_text(self) -> str:
        return (
            f"{self.name} {self.category} {self.description} "
            f"indicators: {', '.join(self.indicators)} graph: {self.graph_signature}"
        )

    def to_metadata(self) -> Dict:
        return {
            "name": self.name,
            "category": self.category,
            "indicators": ";".join(self.indicators),
            "graph_signature": self.graph_signature,
            "regulatory_refs": ";".join(self.regulatory_refs),
        }


SEED_TYPOLOGIES: List[Typology] = [
    Typology(
        id="TYPO-001", name="peel_chain", category="layering",
        description=(
            "Funds moved through a long sequence of addresses, with small amounts "
            "peeled off at each step to obscure origin in money laundering."
        ),
        indicators=[
            "high fan-out per address (>10 outputs)",
            "decreasing balance pattern",
            "rapid sequential timing (<1 hour between hops)",
            "majority of value flows to single 'main' chain",
        ],
        graph_signature="linear_chain_with_branches",
        typical_features={"fan_out": "5-30", "time_delta_s": "10-3600", "depth": "5-50"},
        regulatory_refs=["FATF Recommendation 10", "FinCEN Advisory 2019-A003"],
    ),
    Typology(
        id="TYPO-002", name="layering", category="laundering",
        description=(
            "Multiple sequential transactions across diverse addresses to create "
            "complex routing that obscures the source of funds."
        ),
        indicators=[
            "many intermediate hops",
            "transactions that loop or fan back",
            "use of mixers or tumblers in path",
            "diverse amount distributions",
        ],
        graph_signature="dense_subgraph_or_loop",
        typical_features={"hops": "8-100", "unique_addresses": "high"},
        regulatory_refs=["FATF Recommendation 11"],
    ),
    Typology(
        id="TYPO-003", name="smurfing", category="placement",
        description=(
            "Splitting a large amount into many small deposits below reporting "
            "thresholds to evade AML/KYC detection."
        ),
        indicators=[
            "many incoming amounts just under reporting threshold",
            "high fan-in to a single aggregator address",
            "consistent time clustering of deposits",
        ],
        graph_signature="fan_in_to_aggregator",
        typical_features={"fan_in": ">10", "amount": "near reporting threshold"},
        regulatory_refs=["FinCEN BSA reporting threshold"],
    ),
    Typology(
        id="TYPO-004", name="mixing", category="layering",
        description=(
            "Use of cryptocurrency mixers (e.g., Tornado Cash) to obscure the link "
            "between source and destination addresses."
        ),
        indicators=[
            "transaction routes through known mixer addresses",
            "uniform output amounts (mixer signature)",
            "time delays between deposit and withdrawal",
        ],
        graph_signature="passes_through_known_mixer",
        typical_features={"output_amount": "uniform", "withdrawal_delay": ">1h"},
        regulatory_refs=["OFAC Tornado Cash designation 2022", "FATF VASP guidance"],
    ),
    Typology(
        id="TYPO-005", name="mule_account", category="cashout",
        description=(
            "Account used as intermediary to receive illicit funds and forward them, "
            "often controlled by a coerced or unwitting individual."
        ),
        indicators=[
            "rapid pass-through of received funds",
            "minimal legitimate activity history",
            "incoming from multiple unrelated sources",
            "outgoing to small set of consistent destinations",
        ],
        graph_signature="hub_with_short_holding_time",
        typical_features={"holding_time_s": "<3600", "source_diversity": "high"},
        regulatory_refs=["FATF Recommendation 13"],
    ),
    Typology(
        id="TYPO-006", name="rapid_cashout", category="cashout",
        description=(
            "Funds rapidly moved from compromised account or fraudulent receipt to "
            "an exchange or off-ramp before the victim/system can react."
        ),
        indicators=[
            "transfers to known exchange addresses within minutes",
            "transactions outside normal hours",
            "amounts close to account balance",
        ],
        graph_signature="quick_path_to_exchange",
        typical_features={"time_to_exchange": "<600s"},
        regulatory_refs=["FATF Travel Rule"],
    ),
    Typology(
        id="TYPO-007", name="ransomware_payment", category="extortion",
        description=(
            "Bitcoin payment to a ransomware-associated address, often followed by "
            "rapid laundering through multiple hops."
        ),
        indicators=[
            "single large payment to flagged address",
            "downstream rapid layering",
            "payment matching known ransomware demand patterns",
        ],
        graph_signature="single_inflow_then_layering",
        typical_features={"payment_size": "large", "downstream_complexity": "high"},
        regulatory_refs=["OFAC ransomware advisory 2021", "FinCEN advisory 2020-A006"],
    ),
    Typology(
        id="TYPO-008", name="dusting_attack", category="reconnaissance",
        description=(
            "Tiny amounts sent to many addresses to deanonymize wallet ownership "
            "by tracking subsequent consolidation."
        ),
        indicators=[
            "many tiny outputs to unrelated addresses",
            "amounts below typical fee level",
            "from a single source to a large fan-out",
        ],
        graph_signature="single_source_huge_fan_out_tiny_amounts",
        typical_features={"output_amount": "below fee threshold", "fan_out": ">100"},
        regulatory_refs=[],
    ),
    Typology(
        id="TYPO-009", name="darknet_market_settlement", category="illicit_commerce",
        description=(
            "Transactions characteristic of darknet marketplace settlements — "
            "escrow-like patterns and known marketplace addresses."
        ),
        indicators=[
            "interaction with known darknet addresses",
            "escrow-style multi-sig patterns",
            "round-numbered amounts characteristic of marketplaces",
        ],
        graph_signature="escrow_triangle_with_marketplace",
        typical_features={"address_overlap_with_known_dnm": "yes"},
        regulatory_refs=["FATF Virtual Asset guidance"],
    ),
    Typology(
        id="TYPO-010", name="exchange_arbitrage_lookalike", category="false_positive",
        description=(
            "Legitimate cross-exchange arbitrage that superficially resembles "
            "layering due to multi-hop routing and rapid timing."
        ),
        indicators=[
            "round-trip pattern (funds return to origin)",
            "consistent timing across many similar txs",
            "endpoints are known legitimate exchange addresses",
        ],
        graph_signature="round_trip_through_exchanges",
        typical_features={"endpoints": "known_exchanges"},
        regulatory_refs=[],
    ),
]


def seed_typology_library(client: ChromaDBClient) -> int:
    """Seed the typology_library collection. Idempotent."""
    n_added = 0
    for t in SEED_TYPOLOGIES:
        try:
            client.add(
                collection_name=COLLECTION_NAME,
                doc_id=t.id,
                text=t.to_embedding_text(),
                metadata=t.to_metadata(),
            )
            n_added += 1
        except Exception:
            pass
    return n_added
