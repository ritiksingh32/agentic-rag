"""
Chat router.

The endpoint the frontend calls to ask a question. Runs the agent loop,
saves the turn to chat_history, and returns the answer plus the
reasoning trace (which tools were called, and why) for the UI to display.
"""

from fastapi import APIRouter, Depends

from app.core.auth import get_current_user_id
from app.core.supabase_client import supabase
from app.models.chat import ChatRequest, ChatResponse
from app.services.agent import run_agent
from app.services.documents_db import list_documents

router = APIRouter(prefix="/chat", tags=["chat"])


@router.post("", response_model=ChatResponse)
def chat(request: ChatRequest, user_id: str = Depends(get_current_user_id)):
    history = [{"role": m.role, "content": m.content} for m in request.conversation_history]

    result = run_agent(
        user_id=user_id,
        question=request.question,
        conversation_history=history,
        list_documents_fn=list_documents,
    )

    # Save this turn to chat_history for future reference / memory across sessions
    supabase.table("chat_history").insert({
        "user_id": user_id,
        "question": request.question,
        "answer": result["answer"],
        "reasoning_trace": result["reasoning_trace"],
    }).execute()

    return result


@router.get("/history")
def get_chat_history(user_id: str = Depends(get_current_user_id)):
    """Returns this user's past conversation turns, most recent first."""
    result = (
        supabase.table("chat_history")
        .select("*")
        .eq("user_id", user_id)
        .order("created_at", desc=True)
        .limit(50)
        .execute()
    )
    return result.data