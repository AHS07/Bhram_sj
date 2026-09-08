"""
Groups facts across documents by embedding a synthetic 'entity | attribute'
string (not the full fact — avoids diluting the vector with context/quote
noise) into a persistent Chroma collection. This is a CANDIDATE filter only:
the final same/different-entity decision is made by the classification LLM
in classification.py, not by the embedding threshold alone.
"""
import os
import chromadb

# Suppress ChromaDB telemetry — it tries to ping an analytics endpoint that
# doesn't exist in this version, printing a warning on every call. Purely cosmetic.
os.environ["CHROMA_TELEMETRY"] = "false"
os.environ["ANONYMIZED_TELEMETRY"] = "false"

SIMILARITY_THRESHOLD = 0.85  # validated in Phase 4: correct for same-entity documents.
# Cross-doc similarity between the two dev PDFs (Delhivery vs RBI) peaks at 0.74 —
# well below this threshold, which is correct: they cover different entities and
# should not produce spurious candidate pairs. Same-entity documents (e.g. two
# Delhivery filings) produce intra-group similarities of 0.87-1.0.

# n_results ceiling: fetch up to 20 nearest neighbours so that common attributes
# (e.g. "Revenue" appearing across 10+ pages) are not truncated at 5.
_N_RESULTS = 20

_client = chromadb.PersistentClient(path="./chroma_store")
_collection = _client.get_or_create_collection(
    "fact_grouping", metadata={"hnsw:space": "cosine"}
)


def grouping_string(entity: str, attribute: str) -> str:
    return f"{entity} | {attribute}"


def index_fact(fact_id: str, entity: str, attribute: str) -> None:
    _collection.add(ids=[fact_id], documents=[grouping_string(entity, attribute)])


def find_candidate_group(fact_id: str, entity: str, attribute: str, n_results: int = _N_RESULTS) -> list[str]:
    """Returns fact_ids of other facts worth sending to the classification
    LLM alongside this one. Chroma's cosine distance is 1 - similarity.

    n_results defaults to _N_RESULTS (20) so that common attributes across
    multiple documents/pages are not silently truncated at 5.
    """
    # Clamp to collection size — Chroma raises if n_results > collection count.
    count = _collection.count()
    if count == 0:
        return []
    actual_n = min(n_results, count)
    results = _collection.query(
        query_texts=[grouping_string(entity, attribute)],
        n_results=actual_n,
    )
    candidate_ids = results["ids"][0]
    distances = results["distances"][0]
    return [
        fid for fid, dist in zip(candidate_ids, distances)
        if fid != fact_id and (1 - dist) >= SIMILARITY_THRESHOLD
    ]
