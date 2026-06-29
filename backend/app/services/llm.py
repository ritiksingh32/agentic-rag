"""
LLM generation service.

Takes reranked chunks + the user's question, builds a citation-aware
prompt (Point D in the study guide — context-only instructions, an
explicit fallback phrase, and source labeling), and calls Groq's free
API to generate the final answer.
"""

from groq import Groq

from app.core.config import settings

# One shared client, created once at import time.
groq_client = Groq(api_key=settings.GROQ_API_KEY)

MODEL_NAME = "llama-3.1-8b-instant"


def build_prompt(query: str, chunks: list[dict]) -> str:
    """
    Builds the citation-aware RAG prompt.

    Each chunk is labeled [1], [2], [3]... so the LLM can reference
    exactly which source backs each claim in its answer — the upgrade
    over the original naive-RAG prompt, which had no source attribution.
    """
    if not chunks:
        sources_block = "(no relevant sources were found)"
    else:
        labeled_sources = []
        for i, chunk in enumerate(chunks, start=1):
            labeled_sources.append(
                f"[{i}] ({chunk['filename']}, page {chunk['page_number']}):\n{chunk['chunk_text']}"
            )
        sources_block = "\n\n".join(labeled_sources)

    prompt = f"""You are a research assistant. Answer the question using ONLY the numbered sources below.

Rules:
1. Answer ONLY using information from the sources provided.
2. Cite the source number for every claim you make, like [1] or [2].
3. If the answer is not present in the sources, say exactly:
   "I could not find this information in the documents."
4. Be concise and clear.

Sources:
{sources_block}

Question:
{query}
"""
    return prompt


def generate_answer(query: str, chunks: list[dict]) -> str:
    """
    Calls Groq's LLM with the citation-aware prompt and returns the
    generated answer text.
    """
    prompt = build_prompt(query, chunks)

    response = groq_client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,  # low temperature -> more focused, less "creative" answers
    )

    return response.choices[0].message.content