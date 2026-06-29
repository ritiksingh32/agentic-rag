"""
Auth dependency.

Verifies the Supabase JWT sent by the frontend (in the Authorization
header) and extracts the real user_id — this is what makes multi-tenancy
(Point H) real: every protected endpoint uses this to know WHO is asking,
instead of trusting a user_id the client could fake by just sending a
different value.
"""

from fastapi import Header, HTTPException

from app.core.supabase_client import supabase


def get_current_user_id(authorization: str = Header(...)) -> str:
    """
    FastAPI dependency — add `user_id: str = Depends(get_current_user_id)`
    to any route that needs to know the authenticated user.

    Expects the header:  Authorization: Bearer <supabase_jwt>
    """
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or malformed Authorization header.")

    token = authorization.removeprefix("Bearer ").strip()

    try:
        user_response = supabase.auth.get_user(token)
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    if not user_response or not user_response.user:
        raise HTTPException(status_code=401, detail="Invalid or expired token.")

    return user_response.user.id