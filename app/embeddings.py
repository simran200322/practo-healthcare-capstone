"""Local embedding pipeline using SentenceTransformers (all-MiniLM-L6-v2)."""
from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from app.config import EMBEDDING_MODEL_NAME

# Global cached embedding model instance
_EMBEDDING_MODEL = None


def get_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """Retrieve or initialize the cached SentenceTransformer model instance."""
    global _EMBEDDING_MODEL
    if _EMBEDDING_MODEL is None:
        # Downloads once locally to HuggingFace cache, runs completely offline thereafter
        _EMBEDDING_MODEL = SentenceTransformer(model_name)
    return _EMBEDDING_MODEL


def embed_texts(texts: List[str], model_name: str = EMBEDDING_MODEL_NAME) -> List[List[float]]:
    """Generate dense normalized vector embeddings for a list of text strings."""
    model = get_embedding_model(model_name)
    # normalize_embeddings=True makes inner product identical to cosine similarity
    embeddings = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    if isinstance(embeddings, np.ndarray):
        return embeddings.tolist()
    return [e.tolist() if hasattr(e, "tolist") else list(e) for e in embeddings]


def embed_query(query: str, model_name: str = EMBEDDING_MODEL_NAME) -> List[float]:
    """Generate normalized vector embedding for a single search query."""
    return embed_texts([query], model_name)[0]
