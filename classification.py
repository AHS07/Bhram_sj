"""
Single classification call: given two candidate-matched facts plus their
parent page text, decide the relationship. One prompt, one decision point —
deliberately not three separate detection mechanisms.
"""
from schemas import ClassificationResult
from llm_client import call_json, TEXT_MODEL


CLASSIFICATION_SYSTEM_PROMPT = """You compare two facts extracted from different pages/documents and decide their relationship. Use the page context to check for differences in time period, unit, currency, or scope.

Labels:
- corroborated: same entity, attribute, and value (phrasing differences allowed)
- contradicted: same entity and attribute, genuinely conflicting value, no explanation in context
- reconciled: apparent conflict explained by a contextual difference (time, unit, scope, currency)
- unrelated: the embedding match was a false positive

Respond with ONLY this JSON, nothing else: {"label": "corroborated|contradicted|reconciled|unrelated", "reasoning": "one sentence"}
"""


def _quote_window(page_text: str, source_quote: str, window: int = 600) -> str:
    """Return a context window centred on source_quote within page_text.

    Instead of blindly slicing the first N characters (which misses facts
    extracted from the middle or bottom of the page), we locate the quote's
    offset and extract an asymmetric window around it.  Falls back to the
    first `window` chars when the quote cannot be found.
    """
    if not source_quote or not page_text:
        return page_text[:window]

    # Try direct lower-case substring match first for exact character offset.
    lower_page = page_text.lower()
    lower_quote = source_quote.strip().lower()
    pos = lower_page.find(lower_quote) if lower_quote else -1

    if pos != -1:
        quote_len = len(source_quote)
    else:
        # Fall back to whitespace-normalised search
        import re
        _ws = re.compile(r"\s+")
        norm_page = _ws.sub(" ", page_text).lower()
        norm_quote = _ws.sub(" ", source_quote).lower()
        pos = norm_page.find(norm_quote)
        quote_len = len(norm_quote)

    if pos == -1:
        # Quote not found verbatim — fall back to prefix window.
        return page_text[:window]

    # Centre the window: 40 % before the quote, 60 % after (reasoning cues
    # usually follow the quoted sentence).
    pre = max(0, pos - int(window * 0.4))
    post = min(len(page_text), pos + quote_len + int(window * 0.6))
    return page_text[pre:post]


def classify_fact_pair(
    fact_a: dict, fact_b: dict, page_a_text: str, page_b_text: str
) -> ClassificationResult:
    """Classify the relationship between two facts.

    Page context is extracted as a window *around* each fact's source_quote
    rather than a blind prefix slice, so evidence in the middle or bottom of
    a page reaches the classifier.
    """
    ctx_a = _quote_window(page_a_text, fact_a.get("source_quote", ""))
    ctx_b = _quote_window(page_b_text, fact_b.get("source_quote", ""))

    user_content = (
        f"FACT A: entity={fact_a.get('entity')!r} attribute={fact_a.get('attribute')!r} "
        f"value={fact_a.get('value')!r} context={fact_a.get('context')!r}\n"
        f"PAGE A (around source quote):\n{ctx_a}\n\n"
        f"FACT B: entity={fact_b.get('entity')!r} attribute={fact_b.get('attribute')!r} "
        f"value={fact_b.get('value')!r} context={fact_b.get('context')!r}\n"
        f"PAGE B (around source quote):\n{ctx_b}"
    )
    raw = call_json(CLASSIFICATION_SYSTEM_PROMPT, user_content, model=TEXT_MODEL, max_tokens=512)
    result = ClassificationResult(**raw)
    # Hard-cap reasoning at 300 chars for compact storage.
    result.reasoning = result.reasoning[:300]
    return result
