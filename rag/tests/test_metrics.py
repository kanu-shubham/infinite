"""
Unit tests for the metrics layer — mocks all Anthropic API calls.

Tests cover:
- FaithfulnessEvaluator claim extraction and verification logic
- RelevanceEvaluator precision, recall, and answer relevance
- EvaluationResult report formatting
- Edge cases: empty answers, no retrieval results
"""

import sys
import os
import json
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.documents import Chunk, RetrievalResult
from src.generation.generator import RAGResponse, Citation
from src.metrics.faithfulness import FaithfulnessEvaluator, FaithfulnessResult
from src.metrics.relevance import RelevanceEvaluator, RelevanceResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chunk(content: str) -> Chunk:
    c = Chunk(content=content, doc_id="d1", doc_title="Test Doc",
              doc_source="test_source", start_char=0,
              end_char=len(content), chunk_index=0)
    return c


def _make_response(
    query="What is BERT?",
    answer="BERT is a transformer model pre-trained on masked language modeling.",
    chunks=None,
) -> RAGResponse:
    chunks = chunks or [_make_chunk("BERT is a transformer-based model by Google.")]
    results = [RetrievalResult(chunk=c, score=0.8) for c in chunks]
    return RAGResponse(
        query=query,
        answer=answer,
        citations=[],
        retrieval_results=results,
        pattern="naive",
    )


def _mock_message(text: str):
    """Return a minimal Anthropic API response mock."""
    content = MagicMock()
    content.text = text
    msg = MagicMock()
    msg.content = [content]
    return msg


# ---------------------------------------------------------------------------
# Faithfulness tests
# ---------------------------------------------------------------------------

class TestFaithfulnessEvaluator(unittest.TestCase):

    def _make_evaluator(self, extract_response: str, verify_responses: list):
        """Create a FaithfulnessEvaluator with mocked API calls."""
        evaluator = FaithfulnessEvaluator()
        call_count = [0]
        verify_idx = [0]

        def mock_create(**kwargs):
            system = kwargs.get("system", "")
            # First call is extraction, subsequent calls are verification
            if "Extract every atomic" in system:
                return _mock_message(extract_response)
            else:
                resp = verify_responses[verify_idx[0] % len(verify_responses)]
                verify_idx[0] += 1
                return _mock_message(resp)

        evaluator._client = MagicMock()
        evaluator._client.messages.create.side_effect = mock_create
        return evaluator

    def test_all_claims_supported_gives_score_one(self):
        claims = json.dumps(["BERT uses transformers.", "BERT was made by Google."])
        evaluator = self._make_evaluator(claims, ["YES", "YES"])
        response = _make_response()
        result = evaluator.evaluate(response)
        self.assertAlmostEqual(result.score, 1.0)
        self.assertEqual(result.supported_claims, 2)
        self.assertEqual(result.unsupported, [])

    def test_no_claims_supported_gives_score_zero(self):
        claims = json.dumps(["BERT invented reinforcement learning.", "BERT uses CNNs."])
        evaluator = self._make_evaluator(claims, ["NO", "NO"])
        response = _make_response()
        result = evaluator.evaluate(response)
        self.assertAlmostEqual(result.score, 0.0)
        self.assertEqual(result.supported_claims, 0)
        self.assertEqual(len(result.unsupported), 2)

    def test_partial_support_gives_correct_fraction(self):
        claims = json.dumps(["Claim A", "Claim B", "Claim C", "Claim D"])
        evaluator = self._make_evaluator(claims, ["YES", "YES", "NO", "NO"])
        response = _make_response()
        result = evaluator.evaluate(response)
        self.assertAlmostEqual(result.score, 0.5)
        self.assertEqual(result.supported_claims, 2)
        self.assertEqual(result.total_claims, 4)

    def test_no_retrieval_results_gives_score_zero(self):
        evaluator = FaithfulnessEvaluator()
        evaluator._client = MagicMock()  # should not be called
        response = RAGResponse(
            query="q", answer="a", citations=[], retrieval_results=[], pattern="naive"
        )
        result = evaluator.evaluate(response)
        self.assertAlmostEqual(result.score, 0.0)
        evaluator._client.messages.create.assert_not_called()

    def test_empty_claims_fallback(self):
        """If extraction returns no parseable claims, score is 1.0 (no claims to violate)."""
        evaluator = self._make_evaluator("[]", [])
        response = _make_response()
        result = evaluator.evaluate(response)
        self.assertAlmostEqual(result.score, 1.0)
        self.assertEqual(result.total_claims, 0)

    def test_str_representation(self):
        result = FaithfulnessResult(score=0.75, total_claims=4,
                                    supported_claims=3, unsupported=["X"])
        s = str(result)
        self.assertIn("0.75", s)
        self.assertIn("3/4", s)


# ---------------------------------------------------------------------------
# Relevance tests
# ---------------------------------------------------------------------------

class TestRelevanceEvaluator(unittest.TestCase):

    def _make_evaluator(self, precision_responses, recall_response, relevance_response):
        evaluator = RelevanceEvaluator()
        call_count = [0]

        def mock_create(**kwargs):
            system = kwargs.get("system", "")
            if "aspects" in system.lower() or "SCORE:" in system:
                return _mock_message(recall_response)
            elif "0 =" in system or "scale of 0" in system:
                return _mock_message(relevance_response)
            else:
                # precision binary checks
                idx = call_count[0] % len(precision_responses)
                call_count[0] += 1
                return _mock_message(precision_responses[idx])

        evaluator._client = MagicMock()
        evaluator._client.messages.create.side_effect = mock_create
        return evaluator

    def test_perfect_precision(self):
        evaluator = self._make_evaluator(
            precision_responses=["YES", "YES"],
            recall_response="ASPECTS: a,b\nCOVERED: a,b\nSCORE: 2/2",
            relevance_response="5",
        )
        chunks = [_make_chunk("BERT text"), _make_chunk("Transformer text")]
        response = _make_response(chunks=chunks)
        result = evaluator.evaluate(response)
        self.assertAlmostEqual(result.context_precision, 1.0)

    def test_zero_precision(self):
        evaluator = self._make_evaluator(
            precision_responses=["NO", "NO"],
            recall_response="ASPECTS: a\nCOVERED:\nSCORE: 0/1",
            relevance_response="0",
        )
        chunks = [_make_chunk("irrelevant text 1"), _make_chunk("irrelevant text 2")]
        response = _make_response(chunks=chunks)
        result = evaluator.evaluate(response)
        self.assertAlmostEqual(result.context_precision, 0.0)

    def test_perfect_relevance_score(self):
        evaluator = self._make_evaluator(
            precision_responses=["YES"],
            recall_response="ASPECTS: x\nCOVERED: x\nSCORE: 1/1",
            relevance_response="5",
        )
        response = _make_response()
        result = evaluator.evaluate(response)
        self.assertAlmostEqual(result.answer_relevance, 1.0)

    def test_zero_relevance_score(self):
        evaluator = self._make_evaluator(
            precision_responses=["NO"],
            recall_response="ASPECTS: x\nCOVERED:\nSCORE: 0/1",
            relevance_response="0",
        )
        response = _make_response()
        result = evaluator.evaluate(response)
        self.assertAlmostEqual(result.answer_relevance, 0.0)

    def test_overall_is_harmonic_mean(self):
        result = RelevanceResult(
            context_precision=1.0,
            context_recall=1.0,
            answer_relevance=1.0,
        )
        self.assertAlmostEqual(result.overall, 1.0)

    def test_overall_penalises_zero_components(self):
        result = RelevanceResult(
            context_precision=1.0,
            context_recall=0.0,
            answer_relevance=1.0,
        )
        self.assertAlmostEqual(result.overall, 0.0)

    def test_str_representation(self):
        result = RelevanceResult(
            context_precision=0.8,
            context_recall=0.6,
            answer_relevance=0.9,
        )
        s = str(result)
        self.assertIn("0.80", s)
        self.assertIn("0.60", s)
        self.assertIn("0.90", s)


# ---------------------------------------------------------------------------
# EvaluationResult report test
# ---------------------------------------------------------------------------

class TestEvaluationReport(unittest.TestCase):

    def test_report_contains_key_sections(self):
        from src.pipeline import EvaluationResult
        response = _make_response()
        faith = FaithfulnessResult(score=0.9, total_claims=5,
                                   supported_claims=4, unsupported=["X"])
        rel = RelevanceResult(context_precision=0.8, context_recall=0.7,
                              answer_relevance=0.85)
        ev = EvaluationResult(response=response, faithfulness=faith, relevance=rel)
        report = ev.report()
        self.assertIn("NAIVE RAG", report)
        self.assertIn("Faithfulness", report)
        self.assertIn("Relevance", report)
        self.assertIn("0.90", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
