"""Unit tests for the reasoning layer — explainers, schemas, abstractions."""
from __future__ import annotations
import json

import pytest

from src.knowledge.chromadb_client import RetrievedDoc
from src.reasoning.llm_abstraction import LLMProvider, LLMResponse, FallbackChain, parse_json_response
from src.reasoning.schemas import ExplanationOutput, SelfReflectionOutput
from src.reasoning.template_explainer import explain_with_template


# --- mock provider for deterministic tests ---

class MockProvider(LLMProvider):
    def __init__(self, response_text: str, name: str = "mock", model: str = "mock-1"):
        self.response_text = response_text
        self.name = name
        self.model = model
        self.call_count = 0

    def complete(self, system, user, max_tokens=1024, temperature=0.1) -> LLMResponse:
        self.call_count += 1
        return LLMResponse(text=self.response_text, provider=self.name, model=self.model)


class FailingProvider(LLMProvider):
    name = "failing"
    model = "fail-1"

    def complete(self, system, user, max_tokens=1024, temperature=0.1) -> LLMResponse:
        return LLMResponse(text="", provider="failing", model="fail-1", error="simulated")


class TestSchemas:

    def test_valid_explanation(self):
        out = ExplanationOutput(
            typology_match="peel_chain",
            typology_confidence=0.8,
            supporting_evidence=["fan-out high"],
            contradicting_evidence=[],
            narrative_explanation="Test narrative",
            recommended_action="auto_report",
            rag_citations=["doc_1"],
        )
        assert out.typology_match == "peel_chain"

    def test_invalid_action_rejected(self):
        with pytest.raises(Exception):
            ExplanationOutput(
                typology_match=None,
                typology_confidence=0.5,
                narrative_explanation="x",
                recommended_action="invalid_action",
            )

    def test_confidence_out_of_range_rejected(self):
        with pytest.raises(Exception):
            ExplanationOutput(
                typology_match=None,
                typology_confidence=1.5,
                narrative_explanation="x",
                recommended_action="monitor",
            )

    def test_self_reflection(self):
        r = SelfReflectionOutput(
            topic="peel_chains",
            key_lesson="Watch for rapid sequential outputs",
            indicators_to_watch=["fan_out", "time_delta"],
            related_typologies=["peel_chain"],
        )
        assert r.topic == "peel_chains"


class TestJSONParsing:

    def test_parses_clean_json(self):
        text = '{"typology_match": "peel_chain", "typology_confidence": 0.8}'
        result = parse_json_response(text)
        assert result["typology_match"] == "peel_chain"

    def test_strips_markdown_fences(self):
        text = '```json\n{"a": 1}\n```'
        result = parse_json_response(text)
        assert result == {"a": 1}

    def test_extracts_json_from_preamble(self):
        text = 'Here is the analysis: {"a": 2} hope this helps'
        result = parse_json_response(text)
        assert result == {"a": 2}

    def test_returns_none_on_invalid(self):
        assert parse_json_response("") is None
        assert parse_json_response("not json at all") is None


class TestFallbackChain:

    def test_first_provider_succeeds(self):
        good = MockProvider('{"ok": true}')
        bad = FailingProvider()
        chain = FallbackChain([good, bad])
        resp = chain.complete("sys", "user")
        assert good.call_count == 1
        assert resp.text == '{"ok": true}'

    def test_falls_back_to_second(self):
        bad = FailingProvider()
        good = MockProvider('{"recovered": true}')
        chain = FallbackChain([bad, good])
        resp = chain.complete("sys", "user")
        assert resp.text == '{"recovered": true}'

    def test_all_fail_returns_error(self):
        chain = FallbackChain([FailingProvider(), FailingProvider()])
        resp = chain.complete("sys", "user")
        assert resp.error is not None


class TestTemplateExplainer:

    def test_template_returns_valid_explanation(self):
        rag_docs = [
            RetrievedDoc(
                id="TYPO-001", text="peel chain", collection="typology_library",
                metadata={"name": "peel_chain"}, similarity=0.85,
            ),
        ]
        out = explain_with_template(
            transaction_id="tx_test",
            ensemble_score=0.92,
            agreement_count=4,
            top_features=[("feat_47", 0.5)],
            rag_docs=rag_docs,
        )
        assert isinstance(out, ExplanationOutput)
        assert out.typology_match == "peel_chain"
        assert out.recommended_action == "auto_report"
        assert "TYPO-001" in out.rag_citations

    def test_template_with_no_rag_match(self):
        out = explain_with_template(
            transaction_id="tx_test",
            ensemble_score=0.92,
            agreement_count=4,
            top_features=[],
            rag_docs=[],
        )
        assert out.typology_match == "no_match"
