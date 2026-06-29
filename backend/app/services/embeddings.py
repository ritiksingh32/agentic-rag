"""
Embeddings service.

Generates both dense and sparse vectors for text chunks, enabling
hybrid search (Point F.3 in the RAG study guide).

ARCHITECTURE NOTE (deployment memory fix):
- Dense vector  -> uses huggingface_hub's official InferenceClient,
                   which calls Hugging Face's hosted inference (no
                   local PyTorch model loaded in this process). Using
                   the official client instead of raw requests to a
                   hardcoded URL, since HF's serverless API routing
                   has changed over time and the client handles this.
- Sparse vector -> computed locally with rank-bm25, a pure-Python
                   implementation of BM25 — a statistical calculation,
                   not a neural model, so no PyTorch/ML runtime needed.
"""

from huggingface_hub import InferenceClient
from rank_bm25 import BM25Okapi

from app.core.config import settings

DENSE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

hf_client = InferenceClient(
    provider="hf-inference",
    api_key=settings.HUGGINGFACE_API_KEY,
)


def embed_dense(texts: list[str]) -> list[list[float]]:
    """
    Generates dense (semantic) embeddings via Hugging Face's official
    InferenceClient. Returns a list of 384-number vectors, one per
    input text.
    """
    vectors = []
    for text in texts:
        result = hf_client.feature_extraction(text, model=DENSE_MODEL_NAME)
        # feature_extraction can return per-token vectors for some
        # models; for sentence-transformers models it returns a single
        # pooled vector, but we average just in case to always get one
        # fixed-size vector per input text.
        arr = result
        if hasattr(arr, "ndim") and arr.ndim > 1:
            arr = arr.mean(axis=0)
        vectors.append(arr.tolist() if hasattr(arr, "tolist") else list(arr))
    return vectors


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + lowercase tokenizer for BM25."""
    return text.lower().split()


def embed_sparse_for_corpus(all_chunk_texts: list[str]) -> list[dict]:
    """
    Computes BM25 sparse vectors for an ENTIRE corpus (all chunks of a
    document) at once. BM25 needs term-frequency statistics across the
    whole corpus to score correctly.

    Returns one sparse vector dict per input text, in the same order:
        {"indices": [12, 87, ...], "values": [0.81, 0.45, ...]}
    """
    tokenized_corpus = [_tokenize(t) for t in all_chunk_texts]
    bm25 = BM25Okapi(tokenized_corpus)
    vocabulary = {term: i for i, term in enumerate(bm25.idf.keys())}

    sparse_vectors = []
    for doc_tokens in tokenized_corpus:
        unique_terms = set(doc_tokens)
        indices, values = [], []
        for term in unique_terms:
            if term in vocabulary:
                score = bm25.idf.get(term, 0.0) * doc_tokens.count(term)
                if score > 0:
                    indices.append(vocabulary[term])
                    values.append(float(score))
        sparse_vectors.append({"indices": indices, "values": values})

    return sparse_vectors


def embed_chunks(chunk_texts: list[str]) -> dict:
    """
    Convenience function: generates BOTH dense and sparse embeddings
    for a list of chunk texts in one call. Used during ingestion.

    Returns:
        {
            "dense": [[...384 numbers...], ...],
            "sparse": [{"indices": [...], "values": [...]}, ...]
        }
    """
    return {
        "dense": embed_dense(chunk_texts),
        "sparse": embed_sparse_for_corpus(chunk_texts),
    }