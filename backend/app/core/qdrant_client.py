"""
Qdrant connection and collection setup.

This module:
1. Creates a single shared Qdrant client, connected to your cloud cluster.
2. Defines the collection schema for hybrid search (dense + sparse vectors).
3. Provides a function to create the collection if it doesn't already exist.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PayloadSchemaType,
)

from app.core.config import settings

COLLECTION_NAME = "documents"

# The dense embedding model we'll use is BAAI/bge-base-en-v1.5,
# which outputs 768-dimensional vectors. We hardcode this here because
# Qdrant needs to know the exact vector size when the collection is created.
DENSE_VECTOR_SIZE = 768


# One shared client instance, reused across the whole app.
# e.g.  from app.core.qdrant_client import qdrant
qdrant = QdrantClient(
    url=settings.QDRANT_URL,
    api_key=settings.QDRANT_API_KEY,
)


def ensure_collection_exists():
    """
    Creates the 'documents' collection if it doesn't already exist.
    Safe to call every time the app starts — it checks first, so it
    won't error or wipe data if the collection is already there.
    """
    existing_collections = [c.name for c in qdrant.get_collections().collections]

    if COLLECTION_NAME not in existing_collections:
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config={
                # The named dense vector — semantic / meaning-based search
                "dense": VectorParams(
                    size=DENSE_VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            },
            sparse_vectors_config={
                # The named sparse vector — keyword-based search (for hybrid search)
                "sparse": SparseVectorParams(
                    index=SparseIndexParams(),
                ),
            },
        )
        print(f"Created collection '{COLLECTION_NAME}' with dense + sparse vectors.")
    else:
        print(f"Collection '{COLLECTION_NAME}' already exists — skipping creation.")

    _ensure_payload_indexes()


def _ensure_payload_indexes():
    """
    Creates indexes on payload fields we filter by (user_id, doc_id).

    Unlike vectors (auto-indexed via HNSW, see Point C in the study
    guide), Qdrant requires an EXPLICIT index on any payload field used
    in a filter — otherwise filtering fails with a 400 Bad Request.
    Safe to call repeatedly: creating an index that already exists is
    a no-op, not an error.
    """
    for field_name in ["user_id", "doc_id"]:
        qdrant.create_payload_index(
            collection_name=COLLECTION_NAME,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
        )
    print("Payload indexes for 'user_id' and 'doc_id' ensured.")