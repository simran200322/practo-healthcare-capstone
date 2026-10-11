"""Unit tests for knowledge base documents and coverage."""
import pytest
from app.knowledge_base import load_knowledge_base, validate_knowledge_base, REQUIRED_TOPICS

def test_knowledge_base_loading_and_counts():
    """Verify KB has at least 12 documents."""
    docs = load_knowledge_base()
    assert len(docs) >= 12

def test_all_required_topics_covered():
    """Verify all 12 required topics from the PDF are present."""
    docs = load_knowledge_base()
    topics = {d["topic"] for d in docs}
    for req in REQUIRED_TOPICS:
        assert req in topics, f"Required topic '{req}' must be covered"

def test_sentence_length_constraints():
    """Verify each document has between 2 and 5 sentences."""
    docs = load_knowledge_base()
    for doc in docs:
        assert 2 <= doc["sentence_count"] <= 5, (
            f"Doc '{doc['title']}' has {doc['sentence_count']} sentences (required 2-5)"
        )

def test_knowledge_base_validation_passes():
    """Verify validate_knowledge_base succeeds."""
    docs = load_knowledge_base()
    metrics = validate_knowledge_base(docs)
    assert metrics["covered_topics_count"] == 12
