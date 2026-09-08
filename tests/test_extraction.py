"""
Phase 2 exit gate — extraction + grounding check tests.
The grounding verification is deterministic and must not require a live LLM.
"""
import pytest
from extraction import verify_grounding


def test_exact_match():
    assert verify_grounding("Revenue was INR 420 million.", "Revenue was INR 420 million.") is True


def test_case_and_whitespace_normalization():
    assert verify_grounding("revenue  was inr 420 million", "Revenue was INR 420 million.") is True


def test_missing_quote_returns_false():
    assert verify_grounding("This sentence is not in the page.", "Completely different content here.") is False


def test_empty_quote_returns_false():
    assert verify_grounding("", "Some page content.") is False


def test_fuzzy_match_minor_ocr_noise():
    # One character transposition — should still pass at default threshold.
    assert verify_grounding("Revenue was INR 420 milion.", "Revenue was INR 420 million.") is True


def test_grounding_caps_confidence():
    """An ungrounded quote must cap confidence at 0.3."""
    from extraction import _apply_grounding_check
    from schemas import ExtractedFact

    fact = ExtractedFact(
        entity="Acme Corp",
        attribute="Q1 Revenue",
        value="500 Cr",
        source_quote="This quote does not appear anywhere in the page text at all.",
        confidence=0.9,
    )
    result = _apply_grounding_check(fact, "The page says something completely different.")
    assert result.grounding_verified is False
    assert result.confidence <= 0.3
