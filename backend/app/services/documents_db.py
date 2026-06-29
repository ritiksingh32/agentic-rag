"""
Documents database service.

Wraps all Supabase Postgres operations for the 'documents' table, so
routers and the agent's list_my_documents tool don't talk to Supabase
directly — one place to change if the schema or DB ever changes.
"""

from app.core.supabase_client import supabase


def create_document_record(user_id: str, filename: str, page_count: int) -> dict:
    """
    Inserts a new row for an uploaded document. Called BEFORE ingestion
    starts, with status='processing', then updated to 'ready' once
    chunking/embedding/storage finishes successfully.
    """
    result = supabase.table("documents").insert({
        "user_id": user_id,
        "filename": filename,
        "page_count": page_count,
        "status": "processing",
    }).execute()

    return result.data[0]


def mark_document_ready(doc_id: str):
    """Updates a document's status to 'ready' once ingestion succeeds."""
    supabase.table("documents").update({"status": "ready"}).eq("id", doc_id).execute()


def mark_document_failed(doc_id: str):
    """Updates a document's status to 'failed' if ingestion errors out."""
    supabase.table("documents").update({"status": "failed"}).eq("id", doc_id).execute()


def list_documents(user_id: str) -> list[dict]:
    """
    Returns all documents belonging to a user. This is the function
    injected into the agent's list_my_documents tool.
    """
    result = (
        supabase.table("documents")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .execute()
    )

    # Reshape to match what the agent tool expects: {"filename", "doc_id"}
    return [
        {"filename": row["filename"], "doc_id": row["id"], "status": row["status"]}
        for row in result.data
    ]


def delete_document_record(user_id: str, doc_id: str):
    """
    Deletes a document's metadata row. The caller is responsible for
    also deleting its chunks from Qdrant (see vector_store.delete_document_chunks) —
    these are two separate stores, so both must be cleaned up (Point H).
    """
    supabase.table("documents").delete().eq("id", doc_id).eq("user_id", user_id).execute()