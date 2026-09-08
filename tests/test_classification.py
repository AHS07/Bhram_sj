"""
Phase 5 exit gate — classification schema validation tests.
Mocks the LLM call so provider credentials are not required.
"""
import pytest
from unittest.mock import patch
from schemas import ClassificationResult


def test_valid_labels_parse():
    for label in ("corroborated", "contradicted", "reconciled", "unrelated"):
        result = ClassificationResult(label=label, reasoning="test reasoning")
        assert result.label == label


def test_invalid_label_raises():
    with pytest.raises(Exception):
        ClassificationResult(label="unsupported_label", reasoning="x")


def test_classify_fact_pair_returns_result():
    from classification import classify_fact_pair

    mock_response = {"label": "corroborated", "reasoning": "Both facts state the same revenue figure."}

    with patch("classification.call_json", return_value=mock_response):
        result = classify_fact_pair(
            {"entity": "Acme", "attribute": "Revenue", "value": "100 Cr"},
            {"entity": "Acme", "attribute": "Revenue", "value": "1 billion"},
            "Page A text",
            "Page B text",
        )
    assert result.label == "corroborated"
    assert "revenue" in result.reasoning.lower()


def test_quote_window_centers_on_quote():
    from classification import _quote_window
    long_prefix = "A" * 1500
    target_quote = "Revenue reached INR 500 Crore in FY24."
    long_suffix = "B" * 1500
    full_text = f"{long_prefix} {target_quote} {long_suffix}"

    # Naive slice from index 0 misses the quote completely
    assert target_quote not in full_text[:400]

    # Context window centered on quote captures the quote
    window = _quote_window(full_text, target_quote, window=400)
    assert target_quote in window


def test_quote_window_fallback_when_not_found():
    from classification import _quote_window
    full_text = "Start of page. " + ("Middle. " * 50)
    window = _quote_window(full_text, "Missing Quote", window=50)
    assert window.startswith("Start of page.")
