"""Unit tests for LangGraph agent orchestration, routing, memory, and schema conformance."""
import pytest
from app.graph import run_agent, build_agent_graph
from app.schemas import StructuredAgentResponse
from app.memory import get_conversation_history, clear_conversation

def test_graph_structure():
    """Verify LangGraph has >= 4 nodes."""
    graph = build_agent_graph()
    # Check nodes in graph
    node_names = set(graph.nodes.keys())
    assert len(node_names) >= 4
    expected_nodes = {
        "node_guardrail_input",
        "node_route_intent",
        "node_appointment_tool",
        "node_rag_tool",
        "node_output_formatting"
    }
    assert expected_nodes.issubset(node_names)

def test_conditional_routing_appointment_tool():
    """Verify query routes to appointment lookup tool."""
    res = run_agent("Please check the status of appointment APP-0003", conversation_id="test-route-app")
    assert res["intent"] == "appointment_lookup"
    assert "APP-0003" in res["data"]["response_text"]
    assert "status" in res["data"]["tool_output"]

def test_conditional_routing_rag_tool():
    """Verify query routes to RAG tool."""
    res = run_agent("What is the consultation fee structure by specialty?", conversation_id="test-route-rag")
    assert res["intent"] == "policy_rag"
    assert "Practo clinic policy" in res["data"]["response_text"]
    assert res["grounded"] is True

def test_structured_output_schema_conformance():
    """Verify every agent response validates against declared Pydantic schema."""
    res = run_agent("What is the telemedicine eligibility policy?", conversation_id="test-schema")
    validated = StructuredAgentResponse.model_validate(res)
    assert validated.response_id.startswith("resp-")
    assert validated.intent == "policy_rag"
    assert validated.status == "success"

def test_persisted_conversation_memory():
    """Verify conversation memory carries turns and isolates sessions."""
    convo_a = "test-memory-convo-a"
    convo_b = "test-memory-convo-b"
    clear_conversation(convo_a)
    clear_conversation(convo_b)

    run_agent("First message for session A", conversation_id=convo_a)
    run_agent("Second message for session A", conversation_id=convo_a)

    hist_a = get_conversation_history(convo_a)
    hist_b = get_conversation_history(convo_b)

    assert len(hist_a) == 4  # 2 user + 2 assistant messages
    assert len(hist_b) == 0  # isolated, nothing in session B
