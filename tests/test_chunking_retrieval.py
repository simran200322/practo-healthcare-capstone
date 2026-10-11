"""Unit tests for chunking strategies, embeddings, and ChromaDB vector retrieval."""
import pytest
from app.knowledge_base import load_knowledge_base
from app.chunking import fixed_size_chunking, sentence_based_chunking, split_into_sentences
from app.retrieval import retrieve_top_k
from app.config import (
    COLLECTION_NAME_FIXED,
    COLLECTION_NAME_SENTENCE,
    CHROMA_FIXED_DIR,
    CHROMA_SENTENCE_DIR,
)

def test_split_into_sentences():
    text = "First sentence. Second sentence! Third sentence?"
    sentences = split_into_sentences(text)
    assert len(sentences) == 3
    assert sentences[0] == "First sentence."
    assert sentences[1] == "Second sentence!"
    assert sentences[2] == "Third sentence?"

def test_fixed_size_chunking():
    docs = load_knowledge_base()
    chunks = fixed_size_chunking(docs, chunk_size=180, chunk_overlap=40)
    assert len(chunks) > 0
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "parent_doc_id" in chunk
        assert "text" in chunk
        assert chunk["strategy"] == "fixed_size"
        assert len(chunk["text"]) <= 180

def test_sentence_based_chunking():
    docs = load_knowledge_base()
    chunks = sentence_based_chunking(docs)
    assert len(chunks) > 0
    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "parent_doc_id" in chunk
        assert "text" in chunk
        assert chunk["strategy"] == "sentence_based"
        assert len(chunk["text"]) > 10

def test_retrieval_on_sample_query():
    """Verify both collections produce sensible retrieval for an in-scope query."""
    query = "What is the fee for Cardiology consultation?"
    
    # Retrieve from sentence collection
    results_sentence = retrieve_top_k(query, COLLECTION_NAME_SENTENCE, CHROMA_SENTENCE_DIR, top_k=3)
    assert len(results_sentence) == 3
    assert any("Cardiology" in r["text"] or "consultation" in r["text"].lower() for r in results_sentence)
    assert results_sentence[0]["similarity"] > 0.40

    # Retrieve from fixed-size collection
    results_fixed = retrieve_top_k(query, COLLECTION_NAME_FIXED, CHROMA_FIXED_DIR, top_k=3)
    assert len(results_fixed) == 3
    assert any("Cardiology" in r["text"] or "consultation" in r["text"].lower() for r in results_fixed)
    assert results_fixed[0]["similarity"] > 0.40
