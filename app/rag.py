"""RAG core: Grounded generation with empirically calibrated similarity threshold and MOCK_LLM."""
from typing import Dict, Any, List, Optional
from app.config import (
    RAG_SIMILARITY_THRESHOLD,
    COLLECTION_NAME_SENTENCE,
    CHROMA_SENTENCE_DIR,
)
from app.retrieval import retrieve_top_k

FALLBACK_POLICY_RESPONSE = (
    "I do not have sufficient verified clinic policy information to answer your inquiry. "
    "Please contact clinic reception or Practo patient support directly."
)


def generate_grounded_response(
    query: str,
    retrieved_chunks: List[Dict[str, Any]],
    similarity_threshold: float = RAG_SIMILARITY_THRESHOLD
) -> Dict[str, Any]:
    """Generate a deterministic grounded answer using ONLY the retrieved context chunks.
    
    Under MOCK_LLM mode, if top-1 retrieval similarity is below the calibrated threshold,
    the system conservatively triggers the 'I don't know' fallback rather than hallucinating.
    """
    if not retrieved_chunks:
        return {
            "answer": FALLBACK_POLICY_RESPONSE,
            "grounded": False,
            "is_fallback": True,
            "top_similarity": 0.0,
            "sources": []
        }

    top_chunk = retrieved_chunks[0]
    top_similarity = top_chunk["similarity"]

    # Groundedness gating: refuse unsupported questions
    if top_similarity < similarity_threshold:
        return {
            "answer": FALLBACK_POLICY_RESPONSE,
            "grounded": False,
            "is_fallback": True,
            "top_similarity": top_similarity,
            "sources": []
        }

    # Format factual grounded response strictly from retrieved chunks
    unique_sources = []
    seen_docs = set()
    for c in retrieved_chunks:
        doc_id = c["parent_doc_id"]
        if doc_id not in seen_docs:
            seen_docs.add(doc_id)
            unique_sources.append({
                "doc_id": doc_id,
                "title": c["title"],
                "topic": c["topic"]
            })

    # Combine top retrieved text strictly preserving context facts
    primary_fact = top_chunk["text"]
    supporting_facts = [
        c["text"] for c in retrieved_chunks[1:]
        if c["parent_doc_id"] == top_chunk["parent_doc_id"]
    ]

    answer_body = primary_fact
    if supporting_facts:
        answer_body += " " + " ".join(supporting_facts[:1])

    formatted_answer = (
        f"Based on Practo clinic policy ({top_chunk['title']}): {answer_body}"
    )

    return {
        "answer": formatted_answer,
        "grounded": True,
        "is_fallback": False,
        "top_similarity": top_similarity,
        "sources": unique_sources
    }


def rag_query(
    query: str,
    collection_name: str = COLLECTION_NAME_SENTENCE,
    persist_dir: Optional[Any] = None,
    top_k: int = 3,
    similarity_threshold: float = RAG_SIMILARITY_THRESHOLD
) -> Dict[str, Any]:
    """End-to-end RAG pipeline: retrieval followed by grounded generation."""
    chunks = retrieve_top_k(
        query=query,
        collection_name=collection_name,
        persist_dir=persist_dir,
        top_k=top_k
    )
    result = generate_grounded_response(
        query=query,
        retrieved_chunks=chunks,
        similarity_threshold=similarity_threshold
    )
    result["query"] = query
    result["retrieved_chunks"] = chunks
    return result
