"""Placeholder for retrieval over local documents. NOT wired into the agent, not in tools.TOOLS,
and deliberately free of heavy imports so startup is unaffected while unused.

When it becomes real: a chunker, an embedding model (nomic-embed-text via Ollama), and a vector
store (sqlite-vec keeps it to one file), reached through a rule rather than a bound tool.
"""

from pathlib import Path

from ..config import ROOT

DOCS_DIR = ROOT / "docs"       # where your notes would live
STORE_PATH = ROOT / "rag.db"   # where the vectors would go
CHUNK_CHARS = 800


def available() -> bool:
    """Whether a built index exists. False keeps the agent from offering it."""
    return STORE_PATH.exists()


def index(paths: list[Path] | None = None) -> int:
    """Build the vector store. Returns the number of chunks indexed."""
    raise NotImplementedError("RAG is a placeholder - see rag.py")


def search(query: str, k: int = 4) -> str:
    """Return the k most relevant chunks, formatted for the model."""
    raise NotImplementedError("RAG is a placeholder - see rag.py")
