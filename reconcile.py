"""
For every fact, finds candidate matches via the embedding grouping layer,
then runs the classification LLM on each new pair, storing the result as a
FactRelation. Safe to re-run after each upload — already-classified pairs
are skipped, so this naturally supports incremental ingestion without
rebuilding existing relations.
"""
import logging
import uuid
from pydantic import ValidationError
from sqlmodel import select
from db import get_session
from models import Fact, Page, FactRelation
from canonicalization import find_candidate_group
from classification import classify_fact_pair

logger = logging.getLogger("bhram.reconcile")


def reconcile_all() -> list[dict]:
    """Returns a list of {fact_id_a, fact_id_b, error} for pairs whose
    classification call failed, so a single bad LLM response doesn't
    silently drop the whole batch or crash the request."""
    classification_errors: list[dict] = []

    with get_session() as session:
        facts = session.exec(select(Fact)).all()
        fact_by_id = {f.fact_id: f for f in facts}

        seen_pairs = set()
        for rel in session.exec(select(FactRelation)).all():
            seen_pairs.add(frozenset([rel.fact_id_a, rel.fact_id_b]))

        total = len(facts)
        llm_calls = 0
        for i, fact in enumerate(facts):
            candidates = find_candidate_group(fact.fact_id, fact.entity, fact.attribute)
            for candidate_id in candidates:
                if candidate_id == fact.fact_id:
                    continue

                pair_key = frozenset([fact.fact_id, candidate_id])
                if pair_key in seen_pairs or candidate_id not in fact_by_id:
                    continue
                seen_pairs.add(pair_key)

                other = fact_by_id[candidate_id]
                page_a = session.get(Page, fact.page_id)
                page_b = session.get(Page, other.page_id)

                try:
                    result = classify_fact_pair(
                        fact.model_dump(), other.model_dump(),
                        page_a.raw_text[:2000], page_b.raw_text[:2000],
                    )
                except (ValidationError, ValueError, Exception) as e:
                    # ValidationError: LLM returned a label outside the Literal set,
                    # or malformed JSON shape. ValueError: call_json exhausted retries.
                    # Either way: log it, skip this pair, keep the batch alive.
                    logger.warning(
                        "classification failed for pair (%s, %s): %s",
                        fact.fact_id, candidate_id, e,
                    )
                    classification_errors.append({
                        "fact_id_a": fact.fact_id,
                        "fact_id_b": candidate_id,
                        "error": str(e),
                    })
                    continue

                session.add(FactRelation(
                    relation_id=str(uuid.uuid4()),
                    fact_id_a=fact.fact_id, fact_id_b=other.fact_id,
                    label=result.label, reasoning=result.reasoning,
                ))

        session.commit()

    return classification_errors
