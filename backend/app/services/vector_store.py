"""
Vector store service.

Takes chunks (with their dense + sparse embeddings and metadata) and
stores them as points in the Qdrant 'documents' collection.

Also handles deleting all chunks for a given document — needed for the
delete-then-reinsert pattern (Point H in the study guide) when a user
re-uploads or removes a document.
"""

import uuid

from qdrant_client.models import PointStruct, Filter, FieldCondition, MatchValue

from app.core.qdrant_client import qdrant, COLLECTION_NAME
from app.services.embeddings import embed_chunks


def store_chunks(chunks: list[dict], user_id: str, doc_id: str, filename: str):
    """
    Embeds and stores a list of chunks into Qdrant.

    Each chunk dict is expected to look like:
        {"chunk_index": 0, "page_number": 1, "text": "..."}

    Every point gets tagged with user_id and doc_id in its payload —
    this is what enables per-user filtering (multi-tenancy, Point H)
    and per-document deletion later.
    """
    if not chunks:
        return 0

    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_chunks(texts)  # {"dense": [...], "sparse": [...]}

    points = []
    for i, chunk in enumerate(chunks):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": embeddings["dense"][i],
                    "sparse": embeddings["sparse"][i],
                },
                payload={
                    "user_id": user_id,
                    "doc_id": doc_id,
                    "filename": filename,
                    "page_number": chunk["page_number"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_text": chunk["text"],
                },
            )
        )

    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    return len(points)


def delete_document_chunks(user_id: str, doc_id: str):
    """
    Deletes all chunks belonging to a specific document.

    Used when a document is removed, or before re-ingesting an updated
    version of the same document (the delete-then-reinsert pattern).

    Filters by BOTH user_id and doc_id, as a safety measure — a user
    should never be able to delete another user's chunks, even if a
    doc_id were somehow guessed or reused.
    """
    qdrant.delete(
        collection_name=COLLECTION_NAME,
        points_selector=Filter(
            must=[
                FieldCondition(key="user_id", match=MatchValue(value=user_id)),
                FieldCondition(key="doc_id", match=MatchValue(value=doc_id)),
            ]
        ),
    )