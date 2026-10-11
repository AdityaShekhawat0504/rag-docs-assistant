"""Central settings so every script uses the same paths and models."""
import os
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.getenv("RAG_DATA_DIR", "data")
DB_DIR = os.getenv("RAG_DB_DIR", "chroma_db")
COLLECTION = os.getenv("RAG_COLLECTION", "docs")

# Local, free embedding model (runs on CPU / Apple Silicon).
EMBED_MODEL = os.getenv("RAG_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "150"))
TOP_K = int(os.getenv("RAG_TOP_K", "5"))
