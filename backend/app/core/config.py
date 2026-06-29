"""
Centralized configuration for the Agentic RAG backend.

This module loads all secrets/settings from the .env file ONCE, validates
that required values are present, and exposes a single `settings` object
that the rest of the app imports from — instead of every file calling
os.getenv(...) independently and risking typos or missing-key bugs.
"""

import os
from dotenv import load_dotenv

# Load variables from the .env file into the process environment.
# This must happen before we try to read any of them below.
load_dotenv()


class Settings:
    """
    Holds every secret/config value the app needs, read once at startup.
    """

    def __init__(self):
        # --- Qdrant (vector database) ---
        self.QDRANT_URL = os.getenv("QDRANT_URL")
        self.QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")

        # --- Supabase (auth + Postgres) ---
        self.SUPABASE_URL = os.getenv("SUPABASE_URL")
        self.SUPABASE_PUBLISHABLE_KEY = os.getenv("SUPABASE_PUBLISHABLE_KEY")
        self.SUPABASE_SECRET_KEY = os.getenv("SUPABASE_SECRET_KEY")

        # --- Groq (LLM) ---
        self.GROQ_API_KEY = os.getenv("GROQ_API_KEY")

        # --- Tavily (web search tool) ---
        self.TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

        # Fail loudly at startup if anything required is missing,
        # rather than failing confusingly later mid-request.
        self._validate()

    def _validate(self):
        required = {
            "QDRANT_URL": self.QDRANT_URL,
            "QDRANT_API_KEY": self.QDRANT_API_KEY,
            "SUPABASE_URL": self.SUPABASE_URL,
            "SUPABASE_PUBLISHABLE_KEY": self.SUPABASE_PUBLISHABLE_KEY,
            "SUPABASE_SECRET_KEY": self.SUPABASE_SECRET_KEY,
            "GROQ_API_KEY": self.GROQ_API_KEY,
            "TAVILY_API_KEY": self.TAVILY_API_KEY,
        }

        missing = [key for key, value in required.items() if not value]

        if missing:
            raise ValueError(
                f"Missing required environment variables: {', '.join(missing)}. "
                f"Check your backend/.env file."
            )


# Create ONE shared instance, imported everywhere else in the app.
# e.g.  from app.core.config import settings
#       settings.QDRANT_URL
settings = Settings()