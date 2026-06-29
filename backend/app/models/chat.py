"""
Request/response models for the chat endpoint.
"""

from pydantic import BaseModel


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatRequest(BaseModel):
    question: str
    conversation_history: list[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str
    reasoning_trace: list[dict]
    hit_iteration_limit: bool