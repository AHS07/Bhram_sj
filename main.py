"""
FastAPI app. Serves the Bhram Review UI (upload + evidence browser) as a
static mount at /ui, on top of the JSON API. /docs still works for raw
inspection if you prefer Swagger.
"""
# Load .env before any module that reads os.environ at import time (llm_client.py).
from dotenv import load_dotenv
load_dotenv()

import asyncio
import hashlib
import os
import uuid
from typing import Optional

from fastapi import FastAPI, UploadFile, BackgroundTasks, Query, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from sqlmodel import select

from db import init_db, get_session
from models import Document, Page, Fact, FactRelation
from pipeline import build_pipeline

app = FastAPI(title="Bhram — Fact Knowledge Layer")
pipeline = build_pipeline()

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# ── In-process reconciliation job store ──────────────────────────────────────
# Maps job_id → {"status": "running"|"done"|"error", "errors": [...], "count": int}
# A simple dict is fine here — reconcile runs are rare and serial.
_reconcile_jobs: dict[str, dict] = {}


@app.on_event("startup")
def on_startup():
    init_db()
    os.makedirs(STATIC_DIR, exist_ok=True)


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/ui")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return RedirectResponse(url="/ui/favicon.ico")


# ── Upload ────────────────────────────────────────────────────────────────────

def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


@app.post("/upload")
async def upload_pdf(file: UploadFile):
    """Upload a PDF and run the full ingestion pipeline.

    Fix: uses asyncio-friendly file I/O instead of blocking shutil.copyfileobj
    so the event loop is never blocked during large-file reads.

    Deduplication: if a PDF with the same SHA-256 content hash already exists
    in the database, the upload is rejected with 409 Conflict rather than
    silently creating duplicate facts.
    """
    # Read file bytes asynchronously — non-blocking.
    raw_bytes = await file.read()
    content_hash = _sha256_bytes(raw_bytes)

    # Deduplication check.
    with get_session() as session:
        existing = session.exec(
            select(Document).where(Document.content_hash == content_hash)
        ).first()
        if existing:
            raise HTTPException(
                status_code=409,
                detail=f"Document already ingested: '{existing.filename}' (same content hash). "
                       "Delete it first if you want to re-process.",
            )

    # Write to a temp file (non-blocking via run_in_executor).
    import tempfile
    loop = asyncio.get_event_loop()

    def _write_tmp() -> str:
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(raw_bytes)
            return tmp.name

    tmp_path = await loop.run_in_executor(None, _write_tmp)

    try:
        result = await pipeline.ainvoke({
            "pdf_path":      tmp_path,
            "filename":      file.filename,
            "content_hash":  content_hash,
        })
        return {
            "doc_id":           result["doc_id"],
            "facts_extracted":  len(result["facts"]),
            "extraction_errors": result["errors"],
        }
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


# ── Documents ─────────────────────────────────────────────────────────────────

@app.get("/documents")
def list_documents():
    with get_session() as session:
        return session.exec(select(Document)).all()


# ── Facts (paginated) ─────────────────────────────────────────────────────────

def _enrich_fact(fact: Fact, pages: dict, docs: dict) -> dict:
    page = pages.get(fact.page_id)
    doc  = docs.get(page.doc_id) if page else None
    return {
        **fact.model_dump(),
        "page_number": page.page_number if page else None,
        "filename":    doc.filename    if doc  else None,
    }


@app.get("/facts")
def list_facts(
    limit:  int = Query(default=1000, ge=1, le=10000, description="Max facts to return"),
    offset: int = Query(default=0,    ge=0,           description="Pagination offset"),
):
    """List extracted facts with pagination to avoid loading all facts into memory.

    Use limit/offset for pagination. Default page size is 1000.
    """
    with get_session() as session:
        facts = session.exec(select(Fact).offset(offset).limit(limit)).all()
        page_ids = {f.page_id for f in facts}
        doc_ids_needed: set[str] = set()

        pages: dict[str, Page] = {}
        for p in session.exec(select(Page).where(Page.page_id.in_(page_ids))).all():
            pages[p.page_id] = p
            doc_ids_needed.add(p.doc_id)

        docs: dict[str, Document] = {}
        for d in session.exec(select(Document).where(Document.doc_id.in_(doc_ids_needed))).all():
            docs[d.doc_id] = d

        return [_enrich_fact(f, pages, docs) for f in facts]


# ── Reconciliation (async + job polling) ─────────────────────────────────────

def _run_reconcile(job_id: str) -> None:
    """Executed in a BackgroundTask so POST /reconcile returns immediately."""
    import logging, traceback
    from reconcile import reconcile_all
    logger = logging.getLogger("bhram.reconcile_job")
    _reconcile_jobs[job_id]["status"] = "running"
    try:
        errors = reconcile_all()
        with get_session() as session:
            count = len(session.exec(select(FactRelation)).all())
        _reconcile_jobs[job_id].update(
            status="done", errors=errors, relation_count=count
        )
    except Exception as e:
        tb = traceback.format_exc()
        logger.error("reconcile job %s failed:\n%s", job_id, tb)
        _reconcile_jobs[job_id].update(status="error", detail=str(e))


@app.post("/reconcile")
def run_reconciliation(background_tasks: BackgroundTasks, sync: bool = Query(default=False, description="Run synchronously if True")):
    """Kick off reconciliation as a background job and return a job_id immediately.

    Poll GET /reconcile/{job_id} for status. This prevents HTTP 504 gateway
    timeouts on large fact stores where reconciliation takes minutes.

    The endpoint is still safe to call repeatedly — the background job itself
    skips already-classified pairs via seen_pairs.
    """
    if sync:
        from reconcile import reconcile_all
        errors = reconcile_all()
        return {"status": "done", "classification_errors": errors}

    job_id = str(uuid.uuid4())
    _reconcile_jobs[job_id] = {"status": "queued", "errors": [], "relation_count": 0}
    background_tasks.add_task(_run_reconcile, job_id)
    return {"job_id": job_id, "status": "queued"}


@app.get("/reconcile/{job_id}")
def reconcile_status(job_id: str):
    """Poll reconciliation job status."""
    job = _reconcile_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    return {"job_id": job_id, **job}


# ── Relations (paginated) ─────────────────────────────────────────────────────

@app.get("/relations")
def list_relations(
    limit:        int           = Query(default=1000, ge=1,  le=10000),
    offset:       int           = Query(default=0,    ge=0),
    filter_label: Optional[str] = Query(default=None, description="corroborated|contradicted|reconciled|unrelated"),
):
    """Returns each relation with both facts' full detail resolved.

    Paginated to avoid loading all relations into memory. Filter by label
    to retrieve only the subset needed by the UI filter ribbon.
    """
    with get_session() as session:
        q = select(FactRelation)
        if filter_label:
            q = q.where(FactRelation.label == filter_label)
        relations = session.exec(q.offset(offset).limit(limit)).all()

        fact_ids = {r.fact_id_a for r in relations} | {r.fact_id_b for r in relations}
        facts: dict[str, Fact] = {}
        for f in session.exec(select(Fact).where(Fact.fact_id.in_(fact_ids))).all():
            facts[f.fact_id] = f

        page_ids = {f.page_id for f in facts.values()}
        pages: dict[str, Page] = {}
        for p in session.exec(select(Page).where(Page.page_id.in_(page_ids))).all():
            pages[p.page_id] = p

        doc_ids = {p.doc_id for p in pages.values()}
        docs: dict[str, Document] = {}
        for d in session.exec(select(Document).where(Document.doc_id.in_(doc_ids))).all():
            docs[d.doc_id] = d

        def resolve(fact_id: str):
            f = facts.get(fact_id)
            return _enrich_fact(f, pages, docs) if f else None

        return [
            {
                "relation_id": rel.relation_id,
                "label":       rel.label,
                "reasoning":   rel.reasoning,
                "fact_a":      resolve(rel.fact_id_a),
                "fact_b":      resolve(rel.fact_id_b),
            }
            for rel in relations
        ]


# ── Static UI (mounted last so it never shadows API routes) ──────────────────
app.mount("/ui", StaticFiles(directory=STATIC_DIR, html=True), name="ui")
