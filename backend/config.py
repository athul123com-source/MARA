"""Central MARA configuration.

Loads environment variables from .env (project root) and exposes them
as module-level constants so agents never read os.environ directly.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
QDRANT_URL = os.getenv("QDRANT_URL", "")          # e.g. http://localhost:6333 or Qdrant Cloud URL
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "")
# Embedded local Qdrant (no Docker needed): set QDRANT_PATH=./qdrant_storage
QDRANT_PATH = os.getenv("QDRANT_PATH", "")
QDRANT_COLLECTION = "mara_papers"

# Ingestion defaults
CHUNK_SIZE_WORDS = 200       # abstracts are short; 200-word chunks keep chunks topical
ARXIV_MAX_RESULTS = 25

# Models (per MARA guide: cheap model for dev, stronger for final demo)
PLANNER_MODEL = "claude-haiku-4-5-20251001"
CRITIC_MODEL = "claude-haiku-4-5-20251001"
WRITER_MODEL = "claude-sonnet-4-6"
EMBEDDING_MODEL = "text-embedding-3-small"


def get_qdrant_client():
    """Return a Qdrant client using whichever backend is configured.

    Priority: QDRANT_URL (server / cloud) > QDRANT_PATH (embedded on-disk,
    no Docker required) > in-memory (ephemeral, fine for tests).
    """
    from qdrant_client import QdrantClient

    if QDRANT_URL:
        kwargs = {"url": QDRANT_URL}
        if QDRANT_API_KEY:
            kwargs["api_key"] = QDRANT_API_KEY
        return QdrantClient(**kwargs)
    if QDRANT_PATH:
        return QdrantClient(path=QDRANT_PATH)
    return QdrantClient(":memory:")
