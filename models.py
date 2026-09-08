"""SQLModel schema: Document -> Page -> Fact hierarchy, plus cross-fact relations.

Indexes on foreign-key columns eliminate full-table scans on every API call
that joins across tables (_enrich_fact, /relations resolution).
"""
from typing import Optional
from sqlmodel import SQLModel, Field, Index
from sqlalchemy import Column, String


class Document(SQLModel, table=True):
    doc_id:         str = Field(primary_key=True)
    filename:       str
    content_hash:   str = Field(default="", index=True)  # SHA-256 of PDF bytes for dedup
    global_summary: str = ""


class Page(SQLModel, table=True):
    __table_args__ = (
        Index("ix_page_doc_id", "doc_id"),
    )
    page_id:     str = Field(primary_key=True)
    doc_id:      str = Field(foreign_key="document.doc_id")
    page_number: int
    raw_text:    str = ""  # markdown, tables preserved — used for reconciliation lookups


class Fact(SQLModel, table=True):
    __table_args__ = (
        Index("ix_fact_page_id", "page_id"),
    )
    fact_id:           str   = Field(primary_key=True)
    page_id:           str   = Field(foreign_key="page.page_id")
    entity:            str
    attribute:         str
    value:             str
    context:           str   = ""
    source_quote:      str   = ""
    confidence:        float = 0.5
    # Deterministic check (extraction.py::verify_grounding), not LLM self-report.
    # False means source_quote couldn't be matched back to the page text —
    # surface these distinctly in the UI rather than trusting confidence alone.
    grounding_verified: bool = True


class FactRelation(SQLModel, table=True):
    """Cross-document relationship between two facts, produced by the
    classification step (corroborated / contradicted / reconciled / unrelated)."""
    __table_args__ = (
        Index("ix_factrelation_fact_id_a", "fact_id_a"),
        Index("ix_factrelation_fact_id_b", "fact_id_b"),
    )
    relation_id: str = Field(primary_key=True)
    fact_id_a:   str = Field(foreign_key="fact.fact_id")
    fact_id_b:   str = Field(foreign_key="fact.fact_id")
    label:       str
    reasoning:   str = ""
