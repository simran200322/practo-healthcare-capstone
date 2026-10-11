"""Unit tests for FastAPI endpoints: /health, /ask, and /add-document."""
import pytest
from fastapi.testclient import TestClient
from app.api import app

client = TestClient(app)

def test_health_check_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "collections" in data

def test_ask_endpoint_appointment_query():
    payload = {
        "query": "Please check status for APP-0001",
        "conversation_id": "test-api-session"
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "appointment_lookup"
    assert data["status"] == "success"
    assert "APP-0001" in data["data"]["response_text"]
    assert "trace_id" in data

def test_ask_endpoint_rag_policy_query():
    payload = {
        "query": "What is the cancellation and rescheduling policy?",
        "conversation_id": "test-api-session"
    }
    response = client.post("/ask", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["intent"] == "policy_rag"
    assert data["grounded"] is True
    assert "cancellation" in data["data"]["response_text"].lower()

def test_add_document_endpoint():
    payload = {
        "title": "Vaccination Walk-in Policy",
        "topic": "vaccination policy",
        "content": "Walk-in pediatric and adult vaccinations are available on weekdays between 9 AM and 1 PM. Patients must carry immunization booklets."
    }
    response = client.add_document = client.post("/add-document", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "doc_id" in data
    assert data["fixed_chunks_added"] > 0
    assert data["sentence_chunks_added"] > 0
