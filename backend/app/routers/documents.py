"""
Documents router.

Endpoints for uploading, listing, and deleting documents. Every route
requires authentication (get_current_user_id) — no endpoint trusts a
user_id supplied by the client.
"""

import os
import uuid
import shutil

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException

from app.core.auth import get_current_user_id
from app.services.document_loader import load_document
from app.services.chunker import chunk_pages
from app.services.vector_store import store_chunks, delete_document_chunks
from app.services import documents_db

router = APIRouter(prefix="/documents", tags=["documents"])

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    user_id: str = Depends(get_current_user_id),
):
    """
    Full ingestion pipeline: save file -> extract text -> chunk ->
    embed -> store in Qdrant, with a Postgres record tracking status.
    """
    if not file.filename.lower().endswith((".pdf", ".docx")):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported.")

    temp_path = os.path.join(UPLOAD_DIR, f"{uuid.uuid4()}_{file.filename}")
    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        pages = load_document(temp_path, file.filename)
        chunks = chunk_pages(pages)

        doc_record = documents_db.create_document_record(
            user_id=user_id,
            filename=file.filename,
            page_count=len(pages),
        )
        doc_id = doc_record["id"]

        stored_count = store_chunks(
            chunks=chunks,
            user_id=user_id,
            doc_id=doc_id,
            filename=file.filename,
        )

        documents_db.mark_document_ready(doc_id)

        return {
            "doc_id": doc_id,
            "filename": file.filename,
            "pages": len(pages),
            "chunks_stored": stored_count,
            "status": "ready",
        }

    except Exception as e:
        if "doc_id" in dir():
            documents_db.mark_document_failed(doc_id)
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


@router.get("")
def get_documents(user_id: str = Depends(get_current_user_id)):
    """Lists all documents belonging to the current user."""
    return documents_db.list_documents(user_id)


@router.delete("/{doc_id}")
def delete_document(doc_id: str, user_id: str = Depends(get_current_user_id)):
    """
    Deletes a document: removes its chunks from Qdrant AND its
    metadata row from Postgres (both stores must be cleaned up, Point H).
    """
    delete_document_chunks(user_id=user_id, doc_id=doc_id)
    documents_db.delete_document_record(user_id=user_id, doc_id=doc_id)
    return {"status": "deleted", "doc_id": doc_id}