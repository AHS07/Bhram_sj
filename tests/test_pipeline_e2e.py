"""
Phase 3/6 exit gate — end-to-end pipeline smoke test.
Mocks all LLM calls; validates that the LangGraph pipeline wires up correctly
and that Document/Page/Fact rows land in SQLite after a run.
"""
import asyncio
import pytest
import pymupdf as fitz
from unittest.mock import patch
from sqlmodel import select


@pytest.fixture
def synthetic_pdf(tmp_path):
    pdf_path = tmp_path / "test.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Acme Corp reported Q1 revenue of INR 500 crore for FY2025.")
    doc.save(str(pdf_path))
    doc.close()
    return str(pdf_path)


@pytest.fixture
def in_memory_db(tmp_path, monkeypatch):
    """Redirect the SQLite engine to a temp file so tests don't touch bhram.db."""
    import db as db_module
    import models  # Register table models in SQLModel.metadata before create_all
    from sqlmodel import create_engine, SQLModel, Session
    from contextlib import contextmanager

    test_engine = create_engine(f"sqlite:///{tmp_path}/test.db")
    SQLModel.metadata.create_all(test_engine)

    @contextmanager
    def _get_session():
        with Session(test_engine) as s:
            yield s

    monkeypatch.setattr(db_module, "engine", test_engine)
    monkeypatch.setattr(db_module, "get_session", _get_session)
    return test_engine


def test_pipeline_persists_facts(synthetic_pdf, in_memory_db, monkeypatch):
    mock_summary = {"summary": "Test document about Acme Corp FY2025 financials."}
    mock_facts = {
        "facts": [{
            "entity": "Acme Corp",
            "attribute": "Q1 FY2025 Revenue",
            "value": "INR 500 crore",
            "context": "Q1 FY2025",
            "source_quote": "Acme Corp reported Q1 revenue of INR 500 crore for FY2025.",
            "confidence": 0.95,
        }]
    }

    import canonicalization
    monkeypatch.setattr(canonicalization, "index_fact", lambda *a, **kw: None)

    with patch("pipeline.call_json", return_value=mock_summary), \
         patch("extraction.call_json", return_value=mock_facts):

        from pipeline import build_pipeline
        pipeline = build_pipeline()
        result = asyncio.run(pipeline.ainvoke({
            "pdf_path": synthetic_pdf,
            "filename": "test.pdf",
            "content_hash": "test-hash-abc123",
        }))

    assert len(result["facts"]) == 1
    assert result["errors"] == []

    from models import Fact, Document
    from sqlmodel import Session
    with Session(in_memory_db) as session:
        facts = session.exec(select(Fact)).all()
        docs = session.exec(select(Document)).all()
        assert len(docs) == 1
        assert len(facts) == 1
        assert facts[0].entity == "Acme Corp"
