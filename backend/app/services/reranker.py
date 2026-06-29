"""
Reranking service.

Re-scores a shortlist of retrieved chunks using a cross-encoder model
(Point F.2 in the study guide). Unlike the dense/sparse search in
retrieval.py — which score the query and each chunk INDEPENDENTLY,
then compare vectors — a cross-encoder reads the query and a chunk
TOGETHER as one input, producing a more accurate relevance score.

This is slower per-comparison, which is why it's only run on a small
shortlist (e.g. the ~20 candidates hybrid_search already narrowed down
to) rather than the whole collection.
"""

from sentence_transformers import CrossEncoder

# Loaded once at import time, reused for every call.
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    """
    Re-scores chunks against the query using a cross-encoder, and
    returns the top_n most relevant chunks, re-sorted by the new score.

    Args:
        query: the user's question
        chunks: the candidate list from hybrid_search() — each dict
                must have a "chunk_text" key
        top_n: how many chunks to keep after reranking (this becomes
               the final context size sent to the LLM)

    Returns the same chunk dicts, but re-sorted, trimmed to top_n, with
    an added "rerank_score" key.
    """
    if not chunks:
        return []

    # Cross-encoder expects a list of (query, chunk_text) pairs
    pairs = [(query, chunk["chunk_text"]) for chunk in chunks]
    scores = reranker.predict(pairs)

    for chunk, score in zip(chunks, scores):
        chunk["rerank_score"] = float(score)

    reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)

    return reranked[:top_n]