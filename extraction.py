"""Per-page fact extraction into the ExtractedFact schema."""
import difflib
import re
from schemas import PageExtractionResult, ExtractedFact
from llm_client import call_json, TEXT_MODEL

EXTRACTION_SYSTEM_PROMPT = """You extract atomic facts from a document page.

Rules:
- Always resolve entity names to their most formal and complete corporate or structural form found anywhere on this page.
- Standardize attributes into concise, noun-heavy keys (e.g. "Q3 2024 Revenue" not "Revenue recorded in the third quarter").
- Extract only the top 10 most critical facts on this page. Do not transcribe every table cell.
- Every fact must include an exact source_quote grounding it in the page text.
- If a value has a time period, currency, unit, or scope qualifier, put it in "context", not "value".

Respond as JSON: {"facts": [{"entity":..., "attribute":..., "value":..., "context":..., "source_quote":..., "confidence":...}]}
"""

_WS_RE = re.compile(r"\s+")
_MD_RE = re.compile(r"(\*{1,2}|_{1,2}|`)")  # strip bold/italic/code markers


def _normalize(s: str) -> str:
    # Strip markdown emphasis markers before whitespace normalization so that
    # quotes copied verbatim from vision-parsed Markdown tables (which wrap
    # values in ** or *) still match the underlying plain-text token.
    s = _MD_RE.sub("", s or "")
    return _WS_RE.sub(" ", s).strip().lower()


def verify_grounding(source_quote: str, page_text: str, fuzzy_threshold: float = 0.85) -> bool:
    """Deterministic check that the fact's source_quote actually appears in
    the page it claims to come from. Whitespace/case differences are
    tolerated (OCR/markdown reflow); a quote with no real match in the page
    text is a strong hallucination signal that an LLM self-rating won't
    reliably catch on its own."""
    quote_norm = _normalize(source_quote)
    page_norm = _normalize(page_text)
    if not quote_norm:
        return False
    if quote_norm in page_norm:
        return True
    # Fuzzy fallback for minor reflow/OCR noise: slide a same-length window
    # over the page text and take the best match ratio.
    window = len(quote_norm)
    if window == 0 or len(page_norm) < window:
        return difflib.SequenceMatcher(None, quote_norm, page_norm).ratio() >= fuzzy_threshold
    best = 0.0
    step = max(window // 4, 1)
    for start in range(0, len(page_norm) - window + 1, step):
        chunk = page_norm[start:start + window]
        ratio = difflib.SequenceMatcher(None, quote_norm, chunk).ratio()
        if ratio > best:
            best = ratio
            if best >= fuzzy_threshold:
                break
    return best >= fuzzy_threshold


def _apply_grounding_check(fact: ExtractedFact, page_text: str) -> ExtractedFact:
    verified = verify_grounding(fact.source_quote, page_text)
    fact.grounding_verified = verified
    if not verified:
        # Don't trust the LLM's self-reported confidence if the quote isn't
        # actually findable in the source page — cap it hard rather than
        # silently keeping whatever number the model guessed.
        fact.confidence = min(fact.confidence, 0.3)
    return fact


def extract_facts_from_page(page_markdown: str, global_summary: str) -> PageExtractionResult:
    user_content = f"Document summary: {global_summary}\n\nPage content:\n{page_markdown}"
    try:
        # max_tokens=2000: sufficient for up to 10 facts with quotes in non-thinking mode.
        raw = call_json(EXTRACTION_SYSTEM_PROMPT, user_content, model=TEXT_MODEL, max_tokens=2000)
        result = PageExtractionResult(**raw)
        result.facts = [_apply_grounding_check(f, page_markdown) for f in result.facts]
        return result
    except Exception as e:
        # Surfaces as the "extraction failure" required case — caller logs
        # page_id + raw text rather than silently dropping the page.
        raise ValueError(f"extraction failed: {e}")
