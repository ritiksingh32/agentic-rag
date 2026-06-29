"""
Embeddings service.

Generates both dense and sparse vectors for text chunks, enabling
hybrid search (Point F.3 in the RAG study guide):

- Dense vector  -> semantic meaning, via sentence-transformers
                   (BAAI/bge-base-en-v1.5, 768 dimensions, cosine similarity)
- Sparse vector -> keyword-style weighting, via FastEmbed's BM25-style
                   sparse model, for catching exact terms dense search misses

Both models are loaded ONCE at module import time and reused for every
call — loading a model is slow (downloads + initializes weights), so we
never want to do it per-request.
"""

from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding

# Dense embedding model — loaded once, reused everywhere.
# First run downloads the model (a few hundred MB); cached after that.
dense_model = SentenceTransformer("BAAI/bge-base-en-v1.5")

# Sparse embedding model — BM25-style keyword weighting, also loaded once.
sparse_model = SparseTextEmbedding(model_name="Qdrant/bm25")


def embed_dense(texts: list[str]) -> list[list[float]]:
    """
    Generates dense (semantic) embeddings for a list of texts.

    Returns a list of 768-number vectors, one per input text.
    """
    embeddings = dense_model.encode(texts, normalize_embeddings=True)
    return embeddings.tolist()


def embed_sparse(texts: list[str]) -> list[dict]:
    """
    Generates sparse (keyword-weighted) embeddings for a list of texts.

    Returns a list of dicts shaped like:
        {"indices": [12, 87, 203, ...], "values": [0.81, 0.45, ...]}
    which is the format Qdrant expects for sparse vectors.
    """
    results = list(sparse_model.embed(texts))

    sparse_vectors = []
    for result in results:
        sparse_vectors.append({
            "indices": result.indices.tolist(),
            "values": result.values.tolist(),
        })

    return sparse_vectors


def embed_chunks(chunk_texts: list[str]) -> dict:
    """
    Convenience function: generates BOTH dense and sparse embeddings
    for a list of chunk texts in one call.

    Returns:
        {
            "dense": [[...768 numbers...], ...],
            "sparse": [{"indices": [...], "values": [...]}, ...]
        }
    """
    return {
        "dense": embed_dense(chunk_texts),
        "sparse": embed_sparse(chunk_texts),
    }