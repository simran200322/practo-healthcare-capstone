"""Vector indexing and retrieval engine using ChromaDB and SentenceTransformers."""
from pathlib import Path
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.config import Settings

from app.config import (
    CHROMA_FIXED_DIR,
    CHROMA_SENTENCE_DIR,
    COLLECTION_NAME_FIXED,
    COLLECTION_NAME_SENTENCE,
)
from app.embeddings import embed_texts, embed_query


def get_chroma_client(persist_dir: Path) -> chromadb.ClientAPI:
    """Instantiate a persistent ChromaDB client for the given directory."""
    persist_dir.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=str(persist_dir),
        settings=Settings(anonymized_telemetry=False)
    )


def index_chunks(
    chunks: List[Dict[str, Any]],
    collection_name: str,
    persist_dir: Path,
    recreate: bool = True
) -> chromadb.Collection:
    """Index chunks into a dedicated ChromaDB collection with cosine distance.
    
    Prevents duplicate chunks by clearing the collection when recreate=True.
    """
    client = get_chroma_client(persist_dir)
    
    if recreate:
        try:
            client.delete_collection(name=collection_name)
        except Exception:
            pass  # Collection didn't exist yet

    # Specify cosine distance metric
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"}
    )

    if not chunks:
        return collection

    ids = [c["chunk_id"] for c in chunks]
    texts = [c["text"] for c in chunks]
    metadatas = [
        {
            "parent_doc_id": c["parent_doc_id"],
            "title": c["title"],
            "topic": c["topic"],
            "strategy": c["strategy"]
        }
        for c in chunks
    ]

    embeddings = embed_texts(texts)

    # Upsert chunks into ChromaDB
    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas
    )

    return collection


def retrieve_top_k(
    query: str,
    collection_name: str = COLLECTION_NAME_SENTENCE,
    persist_dir: Optional[Path] = None,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """Retrieve top-k most similar chunks for a query from the specified collection.
    
    Returns chunks with cosine similarity in [0, 1].
    """
    if persist_dir is None:
        persist_dir = (
            CHROMA_SENTENCE_DIR
            if collection_name == COLLECTION_NAME_SENTENCE
            else CHROMA_FIXED_DIR
        )

    client = get_chroma_client(persist_dir)
    collection = client.get_collection(name=collection_name)

    query_embedding = embed_query(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"]
    )

    retrieved: List[Dict[str, Any]] = []
    if not results or not results["ids"] or not results["ids"][0]:
        return retrieved

    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    distances = results["distances"][0]

    for chunk_id, doc_text, meta, dist in zip(ids, docs, metas, distances):
        # In Chroma with cosine space: dist = 1 - cosine_similarity
        # So cosine_similarity = max(0.0, 1.0 - dist)
        similarity = max(0.0, round(1.0 - float(dist), 4))
        retrieved.append({
            "chunk_id": chunk_id,
            "text": doc_text,
            "parent_doc_id": meta["parent_doc_id"],
            "title": meta["title"],
            "topic": meta["topic"],
            "strategy": meta["strategy"],
            "distance": round(float(dist), 4),
            "similarity": similarity
        })

    return retrieved
