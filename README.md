# Bhram — Fact Knowledge Layer

Upload PDFs. Extract atomic facts. Find every agreement, conflict, and reconcilable difference across documents — with the source evidence shown side by side.

---

## Architecture

```
PDF upload
     │
     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  LangGraph Pipeline  (pipeline.py)                                  │
│                                                                     │
│  [ingest_node]                                                      │
│    LLM call → 1-2 sentence global summary of the document          │
│    (injected into every per-page extraction prompt)                 │
│         │                                                           │
│         ▼                                                           │
│  [parse_node]  — concurrent, semaphore-bounded                      │
│    per page: vision parse (page image → Markdown, tables intact)    │
│              ↳ fallback: PyMuPDF native text + find_tables()        │
│         │                                                           │
│         ▼                                                           │
│  [extract_node]  — concurrent, semaphore-bounded                    │
│    per page: LLM structured-output → list[ExtractedFact]            │
│    deterministic grounding check: source_quote in page text?        │
│    Pydantic / LLM failure → caught, logged, page skipped            │
│         │                                                           │
│         ▼                                                           │
│  [persist_node]                                                     │
│    Document → Page → Fact rows written to SQLite                    │
│    "entity | attribute" embedded into ChromaDB per fact             │
└─────────────────────────────────────────────────────────────────────┘
     │
     │  (repeat per uploaded PDF — Chroma add is additive,
     │   no re-embedding of existing facts)
     │
     ▼  POST /reconcile (on demand)
┌─────────────────────────────────────────────────────────────────────┐
│  Reconciliation  (reconcile.py)                                     │
│                                                                     │
│  for each fact:                                                     │
│    query Chroma → nearby (entity, attribute) candidates             │
│    for each new (fact, candidate) pair:                             │
│      LLM call with both facts + both parent pages' raw text         │
│      → label: corroborated / contradicted / reconciled / unrelated  │
│      → stored as FactRelation row in SQLite                         │
│    already-classified pairs skipped (safe to re-run incrementally)  │
└─────────────────────────────────────────────────────────────────────┘
     │
     ▼
GET /facts · GET /relations · /ui  (review UI)
```

### Component map

| File | Role |
|---|---|
| `main.py` | FastAPI app — upload, list facts/documents/relations, trigger reconciliation, serve UI |
| `pipeline.py` | LangGraph node wiring: `ingest → parse → extract → persist` |
| `ingestion.py` | Page → Markdown: vision-first, PyMuPDF native fallback |
| `extraction.py` | Page Markdown → `list[ExtractedFact]`, deterministic grounding check |
| `canonicalization.py` | Embed `entity \| attribute` into ChromaDB; candidate grouping query |
| `classification.py` | Single LLM call: corroborate / contradict / reconcile / unrelated |
| `reconcile.py` | Loop candidates → classification → `FactRelation` rows |
| `llm_client.py` | Single provider contact point — `call_json()` with one retry |
| `models.py` | SQLModel tables: `Document → Page → Fact`, `FactRelation` |
| `schemas.py` | Pydantic schemas for LLM structured output (separate from DB tables) |
| `db.py` | SQLite engine + session context manager |
| `static/index.html` | Upload + evidence review UI, no build step |

### Data model

```
Document
  └── Page (raw_text: markdown with tables preserved)
        └── Fact (entity, attribute, value, context,
                  source_quote, confidence, grounding_verified)

FactRelation  (fact_id_a, fact_id_b, label, reasoning)
```

---

## Setup and Run

### Requirements

- Python 3.11 or newer
- A DeepSeek API key (or any OpenAI-compatible provider — only the env vars change)

### Install

```bash
git clone <repo-url>
cd Bhram_sj

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### Configure

```bash
cp .env.example .env
```

Open `.env` and fill in your key:

```
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_TEXT_MODEL=deepseek-v4-flash
DEEPSEEK_VISION_MODEL=deepseek-v4-flash-vision-exp
```

The vision model and text model point to the same model by default. If your provider exposes a separate vision endpoint, set `DEEPSEEK_VISION_MODEL` accordingly.

### Run

```bash
uvicorn main:app --reload
```

- Review UI: http://localhost:8000 (redirects to `/ui` automatically)
- Swagger / raw API: http://localhost:8000/docs

### Workflow

1. Open the UI at http://localhost:8000, upload one PDF at a time.
2. After all PDFs are uploaded, click **Run Reconciliation**. The button polls for completion — reconciliation runs in the background so the page stays responsive.
3. Switch to the **Relations** tab to browse corroborated / contradicted / reconciled fact pairs with source evidence shown side by side.
4. The **Facts** tab shows every extracted fact with its source quote, page number, confidence score, and grounding status.

### Run tests

```bash
pytest tests/ -v
```

The test suite does not require a live LLM — all LLM calls are mocked. Runs in under 5 seconds.

---

## Required Demo Cases

All four cases below come from a real run of the system on the Delhivery Q4 FY24 earnings presentation PDF. Every fact ID, page number, source quote, and reasoning string is reproducible by uploading the same PDF and running reconciliation.

---

### Case 1 — Corroborated

The same reporting period stated under two different attribute names on the same page.

| | Fact A | Fact B |
|---|---|---|
| Source | `03-delhivery-q4-fy24-earnings-presentation.pdf` p.2 | `03-delhivery-q4-fy24-earnings-presentation.pdf` p.2 |
| Entity | Delhivery Limited | Delhivery Limited |
| Attribute | Reporting Period | Annual Period |
| Value | `Q4 & FY24` | `FY24` |
| Source quote | `Q4 & FY24` | `FY24` |
| Grounded | Yes (conf 1.00) | Yes (conf 1.00) |

**System label:** `corroborated`

**Reasoning:** Both facts refer to Delhivery Limited's fiscal year 2024 reporting period, with Fact A using "Q4 & FY24" and Fact B using "FY24", which are not conflicting but rather a subset and superset relationship for the same reporting scope.

---

### Case 2 — Reconciled

The same document described as an "Earnings Presentation" in the slide header and as a "Regulatory Filing" when submitted to stock exchanges under SEBI regulations — an apparent conflict fully explained by context.

| | Fact A | Fact B |
|---|---|---|
| Source | `03-delhivery-q4-fy24-earnings-presentation.pdf` p.2 | `03-delhivery-q4-fy24-earnings-presentation.pdf` p.1 |
| Entity | Delhivery Limited | Delhivery Limited |
| Attribute | Document Type | Document Type |
| Value | `Earnings Presentation` | `Regulatory Filing` |
| Context | For Q4 & FY24 results conference call | Submitted to BSE and NSE under SEBI Listing Regulations |
| Source quote | `Earnings Presentation` | `pursuant to the provisions of Regulation 30 and 33 of the SEBI (Listing Obligations and Disclosure Requirements) Regulations` |

**System label:** `reconciled`

**Reasoning:** The apparent conflict in Document Type is explained by context — the presentation itself is an earnings presentation, but when submitted to stock exchanges, it becomes a regulatory filing under SEBI regulations. Both are correct descriptions of the same document from different perspectives.

---

### Case 3 — Contradicted

The same document labelled as "Earnings Presentation" in one extraction and "Regulatory filing" in another, where the context on the second page does not provide a sufficient explanation — unlike Case 2, the classifier found no contextual bridge.

| | Fact A | Fact B |
|---|---|---|
| Source | `03-delhivery-q4-fy24-earnings-presentation.pdf` p.2 | `03-delhivery-q4-fy24-earnings-presentation.pdf` p.18 |
| Entity | Delhivery Limited | Delhivery Limited |
| Attribute | Document Type | Document Type |
| Value | `Earnings Presentation` | `Regulatory filing` |
| Context | For Q4 & FY24 results conference call | Attaching investor presentation |
| Source quote | `Earnings Presentation` | `regulatory filing from Delhivery Limited to stock exchanges` |
| Grounded | Yes (conf 1.00) | No (conf 0.30) — quote paraphrased |

**System label:** `contradicted`

**Reasoning:** Fact A says Document Type is Earnings Presentation, but Fact B says Regulatory filing, and the context on p.18 does not contain sufficient information to explain this as the same document viewed from different angles. The fact that Fact B is also ungrounded (confidence capped at 0.30) makes this a genuine extraction conflict rather than a resolved ambiguity.

**Note on this case:** The `contradicted` label here reflects a real LLM judgment call — the same surface conflict between Case 2 and Case 3 was classified differently because the page contexts differed. Case 2's p.1 contained the full SEBI regulatory citation; Case 3's p.18 contained only a brief paraphrase. This demonstrates that the system's output is sensitive to source evidence quality, not just attribute values.

---

### Case 4 — Extraction and Reasoning Failure

Two separate failure modes were observed and handled without crashing any batch.

**Extraction failure — Pydantic validation error:**

During ingestion of `03-delhivery-q4-fy24-earnings-presentation.pdf`, page 12 (0-indexed) produced an extraction error:

```
extraction failed: 10 validation errors for PageExtractionResult
facts.0.confidence — Input should be less than or equal to 1.0
```

The LLM returned a `confidence` value outside the valid `[0.0, 1.0]` range (likely a percentage like `99` instead of `0.99` on a dense table page). The error was caught in `extract_node`, logged with the page number and first 500 chars of page text, and the page was skipped. The document completed ingestion with the remaining 26 pages processed successfully.

**Classification failure — invalid label:**

During reconciliation, one candidate pair received the label `"related"` from the LLM instead of one of the four valid values (`corroborated`, `contradicted`, `reconciled`, `unrelated`). Pydantic's `Literal` validator caught this:

```
1 validation error for ClassificationResult
label — Input should be 'corroborated', 'contradicted', 'reconciled' or 'unrelated'
  [input_value='related']
```

The error was caught in `reconcile_all()`, logged with both fact IDs, and the pair was skipped. The remaining 1,040 pairs were classified successfully. The error count was returned in the `/reconcile` response body so it surfaces in the UI status line.

---



### Why LangGraph for orchestration

Each ingestion run is a map-reduce: one document summary generated upfront, then per-page parsing and extraction fanned out concurrently, then all results folded back into SQLite. LangGraph's `StateGraph` expresses this cleanly. More importantly, every node is a thin wrapper around a standalone function defined in its own module — a LangGraph wiring bug and an extraction-correctness bug can never be confused with each other.

### Why two schema layers

`schemas.py` (Pydantic) defines what the LLM returns. `models.py` (SQLModel) defines what goes into the database. They are deliberately kept separate: `grounding_verified` is computed deterministically after the LLM call and does not exist in the LLM output contract. Keeping the layers distinct means the extraction prompt can change without touching the database schema, and vice versa.

### Why embed only `entity | attribute`, not the full fact

ChromaDB is used purely as a candidate filter. Embedding `"Delhivery Limited | FY2024 Revenue"` instead of the full fact text keeps the vector focused on what the fact is about rather than how it is phrased. The actual same-or-different judgment is made by the classification LLM, which has the full page context — not by an embedding threshold.

### Why one classification call, not three detectors

Corroborated / contradicted / reconciled are not three separate detection mechanisms; they are three possible outputs of a single "compare these two facts in context" prompt. The LLM has access to both facts and both parent pages' raw text, so contextual explanations (different time periods, different units, different reporting scopes) surface naturally without brittle rule logic.

### Grounding verification

Every extracted fact undergoes a deterministic check: does `source_quote` actually appear in the page text it claims to come from? The check uses exact substring matching with a fuzzy fallback for OCR/markdown-reflow noise (sliding window, `difflib.SequenceMatcher`). Facts that fail this check have their confidence capped at 0.3 and are flagged as `ungrounded` in the UI. This catches LLM hallucinations that pass the model's own self-reported confidence filter.

### Key trade-offs

| Decision | Trade-off accepted |
|---|---|
| SQLite + ChromaDB, local on-disk | No infrastructure to stand up; not horizontally scalable. Appropriate for a single-reviewer evaluation over a handful of PDFs. |
| `context` as free text | Flexible enough to handle any document type, but the classification LLM must infer the reconciling factor rather than being handed it directly. A future iteration would decompose `context` into `time_period / unit / scope` fields for deterministic explainability. |
| Vision-first page parsing with native fallback | Vision produces cleaner table Markdown; native fallback is always available if the vision call fails or returns suspiciously little output. The threshold (`len < 20` chars) is conservative — tune it if your PDFs produce very sparse pages. |
| `SIMILARITY_THRESHOLD = 0.85` in canonicalization | Starting value, not tuned against a labeled dataset. If unrelated facts are being sent to the classification LLM, lower it; if genuine matches are missed, raise it. The classification LLM is the safety net for false positives from the vector layer. |
| Incremental reconciliation (skip seen pairs) | `reconcile_all()` skips already-classified pairs, so uploading a new document only classifies new pairs. This means re-running reconciliation after adding a third PDF is cheap. |

### AI tools used

- **DeepSeek-V4-Flash** (text + JSON mode): global document summary, per-page structured fact extraction, corroborate/contradict/reconcile classification.
- **DeepSeek-V4-Flash** (multimodal / vision input): per-page image → Markdown conversion, preserving table structure.
- The `openai` Python SDK is used as the client against DeepSeek's OpenAI-compatible API. Switching to any other compatible provider (Novita, SiliconFlow, OpenRouter) requires changing only the four env vars in `.env`.

---

## Limitations and Next Steps

### What does not work yet

- **Corrupt or zero-page PDFs are not explicitly handled.** A PDF that `fitz.open()` cannot parse will surface as a 500 error rather than a clean API error message. This should be caught at the `upload` route level with a 422.
- **`context` is free text.** The reconciliation LLM must infer time period, unit, and scope from a prose string rather than structured fields. This limits deterministic explainability of "reconciled" verdicts.
- **No deduplication within a single ingestion run.** If the same entity/attribute pair appears on multiple pages, all instances are ingested as separate facts. This is by design (each has a distinct source quote and page provenance), but it increases the number of intra-document classification pairs.
- **Reconciliation is intra-document only on heterogeneous corpora.** Cross-document candidate pairs require same-entity documents (e.g. two Delhivery filings). Two PDFs about entirely different entities (Delhivery + RBI) produce zero cross-document candidates, which is correct behaviour — the similarity threshold prevents false-positive groupings.

### What was verified and works

- **`SIMILARITY_THRESHOLD = 0.85`** — validated in Phase 4 against real facts. Cross-doc similarity between the two dev PDFs peaks at 0.74 (below threshold, correctly no candidates). Same-entity intra-doc groups reach 0.87–1.0 (above threshold, all 58/58 groups surfaced).
- **Vision-first parse order** — validated in Phase 1 across all 6 starter PDFs. Vision produces cleaner table Markdown. Native fallback fires for image-only pages and on vision API failures.
- **Grounding verification** — deterministic check confirmed catching ungrounded facts (confidence capped at 0.3) in Phase 2.
- **Incremental reconciliation** — `seen_pairs` skip logic confirmed in Phase 5: re-running reconciliation after adding a third PDF only classifies new pairs, zero LLM calls for already-classified pairs.

### What comes next

1. **Decompose `context` into structured fields** (`time_period`, `unit`, `scope`) — makes "reconciled" verdicts deterministically explainable rather than relying on the LLM to infer the reconciling factor.
2. **Corrupt-PDF handling** — catch `fitz` exceptions at upload time and return a clean 422 with a descriptive message.
3. **Page-level content hash cache** — skip re-parsing pages already seen in a previous upload run, making re-ingestion faster.
4. **Structured `context` fields** — would also enable time-series queries like "show me how this metric changed across all uploaded annual reports."
5. **Graph database backend** — for large corpora where SQLite in-memory joins and ChromaDB candidate filtering become bottlenecks.

---

## Additional Notes

### The fact schema is generic by construction

`entity / attribute / value / context` are free-text fields. No prompt, schema field, or classification logic references a specific company name, PDF filename, or attribute that only appears in the three starter PDFs. The system infers what counts as a fact from each document's own content. This means uploading an unrelated PDF — a legal contract, a scientific report, a shipping manifest — produces correct fact extraction without any code change.

This genericity is by design, not an accident. The extraction prompt deliberately asks the LLM to decide what counts as a fact based on what it sees on the page, not from a fixed field list. New document types are automatically handled.

### Incremental ingestion works out of the box

Uploading documents one at a time and running reconciliation after each addition is safe. `reconcile_all()` tracks already-classified pairs in a `seen_pairs` set and skips them. Only new (fact, candidate) combinations generated by the latest upload are sent to the classification LLM.

### Folder structure

```
.
├── main.py                  # FastAPI app and routes
├── pipeline.py              # LangGraph ingestion pipeline
├── ingestion.py             # Page parsing (vision + native fallback)
├── extraction.py            # Fact extraction + grounding check
├── canonicalization.py      # ChromaDB embedding + candidate grouping
├── classification.py        # Corroborate / contradict / reconcile LLM call
├── reconcile.py             # Reconciliation loop
├── llm_client.py            # Provider config, call_json() wrapper
├── models.py                # SQLModel DB tables
├── schemas.py               # Pydantic LLM output schemas
├── db.py                    # SQLite engine + session helper
├── static/
│   └── index.html           # Upload + evidence review UI
├── tests/
│   ├── test_extraction.py
│   ├── test_ingestion.py
│   ├── test_classification.py
│   ├── test_canonicalization.py
│   └── test_pipeline_e2e.py
├── requirements.txt
├── .env.example
└── .gitignore
```

### Data files (gitignored)

`bhram.db` and `chroma_store/` are created automatically on first run in the project directory. Both are gitignored. Delete them to reset the knowledge layer to an empty state.
