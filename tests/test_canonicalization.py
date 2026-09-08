"""
Phase 4 exit gate — canonicalization / candidate grouping tests.
Uses an in-memory Chroma collection to avoid polluting the persistent store.
"""
import pytest
import chromadb
from unittest.mock import patch


import uuid

_ephemeral_client = None

def get_test_client():
    global _ephemeral_client
    if _ephemeral_client is None:
        _ephemeral_client = chromadb.EphemeralClient()
    return _ephemeral_client

@pytest.fixture
def ephemeral_collection():
    client = get_test_client()
    col = client.get_or_create_collection(f"test_grouping_{uuid.uuid4().hex}", metadata={"hnsw:space": "cosine"})
    return col


def test_candidate_returned_for_similar_entity_attribute(ephemeral_collection):
    """Two facts with the same entity/attribute string should come back as
    candidates from the grouping query."""
    import canonicalization as canon

    with patch.object(canon, "_collection", ephemeral_collection):
        canon.index_fact("fact-1", "Delhivery Limited", "FY2024 Revenue")
        canon.index_fact("fact-2", "Delhivery Limited", "FY2024 Revenue")

        candidates = canon.find_candidate_group("fact-1", "Delhivery Limited", "FY2024 Revenue")
        assert "fact-2" in candidates


def test_self_not_returned(ephemeral_collection):
    import canonicalization as canon

    with patch.object(canon, "_collection", ephemeral_collection):
        canon.index_fact("fact-solo", "Acme Corp", "Net Profit")
        candidates = canon.find_candidate_group("fact-solo", "Acme Corp", "Net Profit")
        assert "fact-solo" not in candidates


def test_empty_collection_returns_empty_list(ephemeral_collection):
    import canonicalization as canon

    with patch.object(canon, "_collection", ephemeral_collection):
        candidates = canon.find_candidate_group("fact-solo", "Acme Corp", "Net Profit")
        assert candidates == []
