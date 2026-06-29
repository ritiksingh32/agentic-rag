"""
FastAPI entrypoint for the Agentic RAG backend.

Run with:  uvicorn app.main:app --reload
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.core.config import settings
from app.core.qdrant_client import ensure_collection_exists
from app.routers import documents, chat

app = FastAPI(title="Agentic RAG API")

# Allow the Next.js frontend (running on localhost:3000 during development)
# to make requests to this backend. Without this, the browser blocks the
# requests due to CORS (Cross-Origin Resource Sharing) restrictions.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents.router)
app.include_router(chat.router)


@app.on_event("startup")
def on_startup():
    """
    Runs once when the server starts. Ensures the Qdrant collection
    exists before any requests come in.
    """
    ensure_collection_exists()


@app.get("/")
def health_check():
    """
    Simple liveness check — confirms the server is running and that
    settings loaded correctly from .env.
    """
    return {
        "status": "ok",
        "message": "Agentic RAG backend is running",
        "qdrant_configured": bool(settings.QDRANT_URL),
    }