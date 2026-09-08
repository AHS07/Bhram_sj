"""
LangGraph orchestration for a single PDF's ingestion. Each node calls a
standalone function defined elsewhere (ingestion.py, extraction.py) — the
graph wiring itself never contains extraction/parsing logic, so a LangGraph
bug can't masquerade as an extraction-correctness bug.
"""
import asyncio
import hashlib
import uuid
from typing import TypedDict, List
import pymupdf as fitz  # PyMuPDF 1.28+
from langgraph.graph import StateGraph, END

from ingestion import parse_page, parse_page_native_fallback
from extraction import extract_facts_from_page
from llm_client import call_json, TEXT_MODEL
from db import get_session
from models import Document, Page, Fact
from canonicalization import index_fact

CONCURRENCY_LIMIT = 10  # DeepSeek handles 10 concurrent requests comfortably


class PipelineState(TypedDict):
    pdf_path: str
    filename: str
    content_hash: str   # SHA-256 of PDF bytes for deduplication
    doc_id: str
    global_summary: str
    page_texts: List[str]
    facts: List[dict]
    errors: List[dict]


def _pdf_sha256(pdf_path: str) -> str:
    """Content hash of the PDF file for deduplication."""
    h = hashlib.sha256()
    with open(pdf_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


async def ingest_node(state: PipelineState) -> PipelineState:
    """Generates the global document summary, injected into every page's
    extraction prompt to preserve macro-context across page boundaries.

    Fix: If page 0 is an image-only cover (empty native text), we try the
    next few pages for a text-bearing page rather than sending an empty
    string to the summariser.
    """
    first_page_text = ""
    with fitz.open(state["pdf_path"]) as doc:
        total_pages = len(doc)
        # Walk pages until we find one with substantive native text.
        # Image-only cover pages (e.g. IMF report p0) return empty strings.
        for pn in range(min(5, total_pages)):
            candidate = doc[pn].get_text()[:2000].strip()
            if len(candidate) >= 100:
                first_page_text = candidate
                break
        if not first_page_text:
            # All first 5 pages are image-only — fall back to vision on p0
            first_page_text = await asyncio.to_thread(
                parse_page_native_fallback, state["pdf_path"], 0
            )

    summary_raw = call_json(
        "Summarize this document content in 1-2 sentences: what kind of "
        "document is this, what entity/company does it concern, what time period. "
        'Respond as JSON: {"summary": "..."}',
        first_page_text[:2000],
        model=TEXT_MODEL,
        max_tokens=512,
    )
    state["global_summary"] = summary_raw.get("summary", "")
    state["doc_id"] = str(uuid.uuid4())
    return state


async def parse_node(state: PipelineState) -> PipelineState:
    with fitz.open(state["pdf_path"]) as doc:
        page_count = len(doc)

    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)

    async def parse_one(page_number: int) -> str:
        async with semaphore:
            return await asyncio.to_thread(parse_page, state["pdf_path"], page_number)

    state["page_texts"] = await asyncio.gather(*[parse_one(i) for i in range(page_count)])
    return state


async def extract_node(state: PipelineState) -> PipelineState:
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    facts: List[dict] = []
    errors: List[dict] = []

    async def extract_one(page_number: int, page_text: str):
        async with semaphore:
            try:
                result = await asyncio.to_thread(
                    extract_facts_from_page, page_text, state["global_summary"]
                )
                for f in result.facts:
                    facts.append({"page_number": page_number, **f.model_dump()})
            except Exception as e:
                errors.append({
                    "page_number": page_number,
                    "error": str(e),
                    "raw_text": page_text[:500],
                })

    await asyncio.gather(*[
        extract_one(i, text) for i, text in enumerate(state["page_texts"])
    ])
    state["facts"] = facts
    state["errors"] = errors
    return state


def persist_node(state: PipelineState) -> PipelineState:
    """Write Document → Page → Fact rows to SQLite, then index into Chroma.

    Fix: Chroma indexing now happens AFTER session.commit() so that a DB
    rollback doesn't leave Chroma out of sync with SQLite. If Chroma
    indexing fails after a successful commit, the fact remains in SQLite
    and a warning is logged — the vector store can be rebuilt from SQLite
    if needed.
    """
    import logging
    logger = logging.getLogger("bhram.persist")

    facts_to_index: List[tuple] = []  # (fact_id, entity, attribute)

    with get_session() as session:
        document = Document(
            doc_id=state["doc_id"], filename=state["filename"],
            content_hash=state.get("content_hash", ""),
            global_summary=state["global_summary"],
        )
        session.add(document)
        # Flush document first so Page rows can reference it via FK.
        session.flush()

        page_ids = {}
        for page_number, page_text in enumerate(state["page_texts"]):
            page_id = str(uuid.uuid4())
            page_ids[page_number] = page_id
            session.add(Page(
                page_id=page_id, doc_id=state["doc_id"],
                page_number=page_number, raw_text=page_text,
            ))

        # Flush pages to DB before inserting facts — required because FK
        # enforcement (PRAGMA foreign_keys=ON) means fact rows must reference
        # an already-persisted page row, not just one queued in the same flush.
        session.flush()

        for f in state["facts"]:
            fact_id = str(uuid.uuid4())
            page_id = page_ids[f["page_number"]]
            session.add(Fact(
                fact_id=fact_id, page_id=page_id, entity=f["entity"],
                attribute=f["attribute"], value=f["value"],
                context=f.get("context", ""), source_quote=f["source_quote"],
                confidence=f.get("confidence", 0.5),
                grounding_verified=f.get("grounding_verified") if f.get("grounding_verified") is not None else True,
            ))
            facts_to_index.append((fact_id, f["entity"], f["attribute"]))

        # Commit first — only index into Chroma after the relational write succeeds.
        session.commit()

    # Chroma indexing after commit — decoupled from the SQLite transaction.
    for fact_id, entity, attribute in facts_to_index:
        try:
            index_fact(fact_id, entity, attribute)
        except Exception as e:
            logger.warning("Chroma index_fact failed for %s: %s", fact_id, e)

    return state


def build_pipeline():
    graph = StateGraph(PipelineState)
    graph.add_node("ingest", ingest_node)
    graph.add_node("parse", parse_node)
    graph.add_node("extract", extract_node)
    graph.add_node("persist", persist_node)
    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "parse")
    graph.add_edge("parse", "extract")
    graph.add_edge("extract", "persist")
    graph.add_edge("persist", END)
    return graph.compile()
