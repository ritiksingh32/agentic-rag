"""
Retrieval service.

Performs hybrid search (Point F.3 in the study guide): queries both the
dense (semantic) and sparse (keyword) vectors in Qdrant, then fuses the
two ranked lists using Reciprocal Rank Fusion (RRF) — Qdrant does this
fusion natively via the "Query API" prefetch + fusion mechanism.

Always filtered by user_id, enforcing per-user data isolation
(multi-tenancy, Point H) at the database query level.
"""

from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
    Prefetch,
    FusionQuery,
    Fusion,
)

from app.core.qdrant_client import qdrant, COLLECTION_NAME
from app.services.embeddings import embed_dense, embed_sparse


def hybrid_search(query: str, user_id: str, limit: int = 20, doc_id: str | None = None) -> list[dict]:
    """
    Runs hybrid (dense + sparse) search over a user's documents.

    Args:
        query: the user's question
        user_id: restricts results to this user's own documents ONLY
        limit: how many candidates to retrieve before re-ranking (we
               retrieve more than we need here, since a reranker will
               narrow this down later — see Point F.2)
        doc_id: optional — if set, restricts search to ONE specific
                document (used by the compare_documents tool later)

    Returns a list of dicts: [{"chunk_text", "filename", "page_number", "score"}, ...]
    """
    dense_vector = embed_dense([query])[0]
    sparse_vector = embed_sparse([query])[0]

    must_conditions = [FieldCondition(key="user_id", match=MatchValue(value=user_id))]
    if doc_id:
        must_conditions.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id)))

    query_filter = Filter(must=must_conditions)

    results = qdrant.query_points(
        collection_name=COLLECTION_NAME,
        prefetch=[
            Prefetch(
                query=dense_vector,
                using="dense",
                filter=query_filter,
                limit=limit,
            ),
            Prefetch(
                query={"indices": sparse_vector["indices"], "values": sparse_vector["values"]},
                using="sparse",
                filter=query_filter,
                limit=limit,
            ),
        ],
        query=FusionQuery(fusion=Fusion.RRF),
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