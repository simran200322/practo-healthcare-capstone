"""Configuration settings for Practo Healthcare Domain Support Agent."""
import os
from pathlib import Path
from dotenv import load_dotenv

# Base directory paths
BASE_DIR = Path(__file__).resolve().parent.parent
load_dotenv(BASE_DIR / ".env")

# Project directories
DATA_DIR = BASE_DIR / "data"
KB_DIR = BASE_DIR / "knowledge_base"
LOGS_DIR = BASE_DIR / "logs"
TRANSCRIPTS_DIR = BASE_DIR / "transcripts"
EVALUATION_DIR = BASE_DIR / "evaluation"

# Ensure runtime directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)
TRANSCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

# File paths
APPOINTMENTS_FILE = DATA_DIR / "appointments.json"
CHECKPOINTS_DB_FILE = DATA_DIR / "checkpoints.sqlite"
CHROMA_FIXED_DIR = DATA_DIR / "chroma_fixed"
CHROMA_SENTENCE_DIR = DATA_DIR / "chroma_sentence"
REQUEST_LOG_FILE = LOGS_DIR / "requests.jsonl"
MEMORY_FILE = DATA_DIR / "conversation_memory.json"

# Operational flags & Model parameters
# The PDF mandates deterministic MOCK_LLM mode with 0 API keys and 0 external network calls
MOCK_LLM = os.getenv("MOCK_LLM", "true").lower() in ("true", "1", "yes")

# Local SentenceTransformers model
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")

# Retrieval Collections
COLLECTION_NAME_FIXED = "practo_policies_fixed"
COLLECTION_NAME_SENTENCE = "practo_policies_sentence"

# RAG Cosine Similarity Threshold
# Empirically calibrated in Phase 5 based on in-scope vs out-of-scope query clusters
# Default initial value, updated dynamically or via env
RAG_SIMILARITY_THRESHOLD = float(os.getenv("RAG_SIMILARITY_THRESHOLD", "0.40"))

# Escalation parameters for appointment lookup (Task 6)
# Combines follow_up_required weight and normalized recency weight
FOLLOW_UP_WEIGHT = 0.50
RECENCY_WEIGHT = 0.50
ESCALATION_THRESHOLD = 0.55  # Justified based on dataset 80th percentile recency and follow-up flags

# MCP & API Server defaults
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
MCP_HOST = os.getenv("MCP_HOST", "127.0.0.1")
MCP_PORT = int(os.getenv("MCP_PORT", "8000"))
