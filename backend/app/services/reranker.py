"""
Reranking service.

Re-scores a shortlist of retrieved chunks using a cross-encoder model
(Point F.2 in the study guide), via Hugging Face's hosted inference —
no local PyTorch model loaded in this process.

NOTE: cross-encoder/ms-marco-MiniLM-L-6-v2 is NOT deployed by any HF
Inference Provider (confirmed on its model page), so it cannot be
called via the hosted API at all, regardless of which client method is
used. We switched to BAAI/bge-reranker-v2-m3, a comparable cross-encoder
reranker that IS hosted, callable via the text_classification task.
"""

from huggingface_hub import InferenceClient

from app.core.config import settings

RERANK_MODEL_NAME = "BAAI/bge-reranker-v2-m3"

hf_client = InferenceClient(
    provider="hf-inference",
    api_key=settings.HUGGINGFACE_API_KEY,
)


def rerank(query: str, chunks: list[dict], top_n: int = 5) -> list[dict]:
    """
    Re-scores chunks against the query using a cross-encoder hosted on
    Hugging Face, and returns the top_n most relevant chunks.

    text_classification for a reranker model expects the query and
    chunk text combined as one input (commonly separated by [SEP] or
    similar) and returns a relevance score for that pair — so we call
    it once per chunk.
    """
    if not chunks:
        return []

    for chunk in chunks:
        try:
            result = hf_client.text_classification(
                f"{query}</s></s>{chunk['chunk_text']}",
                model=RERANK_MODEL_NAME,
            )
            # result is a list of {"label": ..., "score": ...}; take the
            # top score as this pair's relevance score.
            score = result[0].score if hasattr(result[0], "score") else result[0]["score"]
            chunk["rerank_score"] = float(score)
        except Exception as e:
            print(f"Rerank API call failed for one chunk: {e}")
            chunk["rerank_score"] = 0.0

    reranked = sorted(chunks, key=lambda c: c["rerank_score"], reverse=True)
    return reranked[:top_n]