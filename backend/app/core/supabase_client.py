"""
Supabase client setup.

One shared client, used for:
1. Verifying JWTs from the frontend (auth)
2. Reading/writing the 'documents' and 'chat_history' Postgres tables
"""

from supabase import create_client, Client

from app.core.config import settings

# Uses the SECRET key (not the publishable one) — this is the backend,
# which needs privileged access. Never expose this client/key to the frontend.
supabase: Client = create_client(
    settings.SUPABASE_URL,
    settings.SUPABASE_SECRET_KEY,
)