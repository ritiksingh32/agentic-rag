"""
Agent tools.

Defines every tool the agent can call, plus the JSON schemas that
describe those tools to the LLM (required for Groq's function-calling /
tool-use API). This is the concrete implementation of the tool list
from the project spec: retrieve_documents, web_search, compare_documents,
list_my_documents.

Each tool function returns a STRING (the tool's result, as text) —
this is what gets fed back into the agent's conversation after a tool
call, so the LLM can read it and decide what to do next.
"""

import json

from tavily import TavilyClient

from app.core.config import settings
from app.services.retrieval import hybrid_search
from app.services.reranker import rerank

tavily_client = TavilyClient(api_key=settings.TAVILY_API_KEY)


# ---------------------------------------------------------------------------
# Tool implementations — the actual Python functions that DO the work
# ---------------------------------------------------------------------------

def tool_retrieve_documents(user_id: str, query: str) -> str:
    """
    Searches the user's own uploaded documents (hybrid search + rerank)
    and returns the most relevant chunks, formatted as labeled sources
    the agent can cite from.
    """
    try:
        candidates = hybrid_search(query=query, user_id=user_id, limit=20)
        top_chunks = rerank(query=query, chunks=candidates, top_n=5)
    except Exception as e:
        return f"Tool failed with an error: {e}. Try a different approach or inform the user."

    if not top_chunks:
        return "No relevant chunks were found in the user's uploaded documents for this query."

    formatted = []
    for i, chunk in enumerate(top_chunks, start=1):
        formatted.append(
            f"[{i}] ({chunk['filename']}, page {chunk['page_number']}):\n{chunk['chunk_text']}"
        )
    return "\n\n".join(formatted)


def tool_web_search(query: str) -> str:
    """
    Searches the live web via Tavily — used when the user's documents
    don't contain the answer, or the question isn't about their documents.
    """
    try:
        results = tavily_client.search(query=query, max_results=5)
    except Exception as e:
        return f"Web search tool failed with an error: {e}. Try answering without it, or inform the user."

    if not results.get("results"):
        return "No web search results were found for this query."

    formatted = []
    for i, result in enumerate(results["results"], start=1):
        formatted.append(f"[{i}] ({result['url']}):\n{result['content']}")
    return "\n\n".join(formatted)


def tool_compare_documents(user_id: str, query: str, doc_id_a: str, doc_id_b: str) -> str:
    """
    Retrieves relevant chunks from TWO specific documents separately,
    so the agent can contrast them — something a single retrieval pass
    over mixed documents handles poorly (Point F.5/F.6 motivation).
    """
    try:
        chunks_a = hybrid_search(query=query, user_id=user_id, limit=10, doc_id=doc_id_a)
        chunks_b = hybrid_search(query=query, user_id=user_id, limit=10, doc_id=doc_id_b)
    except Exception as e:
        return f"Tool failed with an error: {e}."

    result = "DOCUMENT A:\n"
    result += "\n".join(c["chunk_text"] for c in chunks_a[:5]) or "(no relevant content found)"
    result += "\n\nDOCUMENT B:\n"
    result += "\n".join(c["chunk_text"] for c in chunks_b[:5]) or "(no relevant content found)"
    return result


def tool_list_my_documents(user_id: str, list_documents_fn) -> str:
    """
    Lists the user's uploaded documents WITHOUT searching their content.
    `list_documents_fn` is injected from the documents router/DB layer
    (built later) — kept as a parameter here so this module has no
    direct dependency on the database layer yet.
    """
    try:
        docs = list_documents_fn(user_id)
    except Exception as e:
        return f"Tool failed with an error: {e}."

    if not docs:
        return "This user has not uploaded any documents yet."

    return "\n".join(f"- {d['filename']} (doc_id: {d['doc_id']})" for d in docs)


# ---------------------------------------------------------------------------
# Tool schemas — describe each tool to the LLM for function calling
# ---------------------------------------------------------------------------

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "retrieve_documents",
            "description": (
                "Search the user's own uploaded documents for information "
                "relevant to a query. Use this when the question is likely "
                "answered by content the user has uploaded."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to run against the user's documents.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": (
                "Search the live web for current or general information NOT "
                "found in the user's uploaded documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to run on the web.",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_documents",
            "description": (
                "Compare content between TWO specific uploaded documents on "
                "a given topic. Use this when the user explicitly asks to "
                "compare, contrast, or find differences between two documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The topic or aspect to compare between the two documents.",
                    },
                    "doc_id_a": {
                        "type": "string",
                        "description": "The doc_id of the first document.",
                    },
                    "doc_id_b": {
                        "type": "string",
                        "description": "The doc_id of the second document.",
                    },
                },
                "required": ["query", "doc_id_a", "doc_id_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_my_documents",
            "description": (
                "List the filenames and IDs of documents the user has "
                "uploaded, without searching their content. Use this when "
                "the user asks what documents they have, or you need a "
                "doc_id to use with compare_documents."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]