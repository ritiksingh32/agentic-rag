"""
Retrieval service.

Performs hybrid search (Point F.3 in the study guide): queries both the
dense (semantic) and sparse (keyword) vectors in Qdrant, then fuses the
two ranked lists using Reciprocal Rank Fusion (RRF) — Qdrant does this
fusion natively via the "Query API" prefetch + fusion mechanism.

Always filtered by user_id, enforcing per-user data isolation
(multi-tenancy, Point H) at the database query level.

ARCHITECTURE NOTE: BM25 (sparse) needs corpus-wide term statistics to
score a query correctly. Since we don't persist the BM25 model object
itself, we rebuild it at query time from the user's own stored chunks —
this is the correct scope for BM25 anyway, since term rarity/frequency
should be computed over the user's full document collection, not a
single document or an arbitrary subset.
"""

from rank_bm25 import BM25Okapi
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    FusionQuery,
    Fusion,
)

from app.core.qdrant_client import qdrant, COLLECTION_NAME
from app.services.embeddings import embed_dense, _tokenize


def _build_user_bm25(user_id: str):
    """
    Fetches ALL chunk texts belonging to this user and builds a fresh
    BM25 model + vocabulary from them. This recomputes corpus-wide
    statistics at query time, since we don't persist the BM25 model
    itself between ingestion and search.
    """
    scroll_result, _ = qdrant.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=Filter(must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]),
        limit=10000,  # generous cap; fine for a portfolio-scale project
        with_payload=["chunk_text"],
    )

    chunk_texts = [point.payload["chunk_text"] for point in scroll_result]
    if not chunk_texts:
        return None, None

    tokenized_corpus = [_tokenize(t) for t in chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    vocabulary = {term: i for i, term in enumerate(bm25.idf.keys())}

    return bm25, vocabulary


def hybrid_search(query: str, user_id: str, limit: int = 20, doc_id: str | None = None) -> list[dict]:
    """
    Runs hybrid (dense + sparse) search over a user's documents.

    Args:
        query: the user's question
        user_id: restricts results to this user's own documents ONLY
        limit: how many candidates to retrieve before re-ranking
        doc_id: optional — if set, restricts search to ONE specific document

    Returns a list of dicts: [{"chunk_text", "filename", "page_number", "score"}, ...]
    """
    dense_vector = embed_dense([query])[0]

    must_conditions = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    if doc_id:
        must_conditions.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))
    query_filter = Filter(must=must_conditions)

    bm25, vocabulary = _build_user_bm25(user_id)

    prefetch_list = [
        Prefetch(query=dense_vector, using="dense", filter=query_filter, limit=limit),
    ]

    # Only add sparse search if the user actually has chunks to build BM25 from
    if bm25 is not None:
        tokens = _tokenize(query)
        indices, values = [], []
        for term in set(tokens):
            if term in vocabulary:
                idf = bm25.idf.get(term, 0.0)
                if idf > 0:
                    indices.append(vocabulary[term])
                    values.append(float(idf))

        if indices:  # only add sparse prefetch if the query has matchable terms
            prefetch_list.append(
                Prefetch(
                    query={"indices": indices, "values": values},
                    using="sparse",
                    filter=query_filter,
                    limit=limit,
                )
            )

    if len(prefetch_list) > 1:
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            prefetch=prefetch_list,
            query=FusionQuery(fusion=Fusion.RRF),
            limit=limit,
        )
    else:
        # Fall back to dense-only search if sparse couldn't be built
        results = qdrant.query_points(
            collection_name=COLLECTION_NAME,
            query=dense_vector,
            using="dense",
            query_filter=query_filter,
            limit=limit,
        )

    chunks = []
    for point in results.points:
        chunks.append({
            "chunk_text": point.payload["chunk_text"],
            "filename": point.payload["filename"],
            "page_number": point.payload["page_number"],
            "doc_id": point.payload["doc_id"],
            "score": point.score,
        })

    return chunks