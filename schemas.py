"""Pydantic schemas used as LLM structured-output targets (not DB tables)."""
from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class ExtractedFact(BaseModel):
    entity: str = Field(..., description="Subject, resolved to its most formal/complete form found in the text")
    attribute: str = Field(..., description="Concise, noun-heavy key, e.g. 'Q3 2024 Revenue'")
    value: str
    context: str = Field(default="", description="Scope/time/unit qualifiers, e.g. 'FY2024, European region'")
    source_quote: str = Field(..., description="Exact sentence or table cell the fact was grounded in")
    confidence: float = Field(default=0.5, ge=0.0, le=1.0, description="LLM self-reported extraction clarity")
    # Computed after extraction (not LLM-provided) — see extraction.py::verify_grounding.
    # None until the post-processing pass runs.
    grounding_verified: Optional[bool] = Field(
        default=None,
        description="Deterministic check: does source_quote actually appear in the page text?",
    )


class PageExtractionResult(BaseModel):
    facts: List[ExtractedFact] = Field(default_factory=list, description="Top 10 most critical facts on this page")


class ClassificationResult(BaseModel):
    label: Literal["corroborated", "contradicted", "reconciled", "unrelated"]
    reasoning: str = Field(..., description="One or two sentences, referencing both facts' context")
