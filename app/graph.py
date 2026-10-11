"""LangGraph agent workflow with conditional routing, tools, memory, and guardrails."""
import re
import uuid
from typing import Dict, Any, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

from app.config import RAG_SIMILARITY_THRESHOLD, COLLECTION_NAME_SENTENCE
from app.guardrails import (
    mask_contact_numbers,
    detect_prompt_injection,
    verify_output_groundedness,
)
from app.tools import check_appointment_status
from app.rag import rag_query
from app.memory import append_message, get_conversation_history
from app.schemas import StructuredAgentResponse


# LangGraph state representation
class GraphState(TypedDict):
    conversation_id: str
    trace_id: str
    raw_query: str
    masked_query: str
    pii_detected: bool
    injection_detected: bool
    intent: str
    tool_input: Dict[str, Any]
    tool_output: Dict[str, Any]
    response_text: str
    grounded: bool
    is_fallback: bool
    escalation_flag: bool
    error: str
    status: str
    structured_response: Dict[str, Any]


# Node 1: Input Guardrail
def node_guardrail_input(state: GraphState) -> Dict[str, Any]:
    raw_query = state.get("raw_query", "")
    masked_query, pii_found = mask_contact_numbers(raw_query)
    is_injected, injection_reason = detect_prompt_injection(masked_query)

    updates: Dict[str, Any] = {
        "masked_query": masked_query,
        "pii_detected": pii_found,
        "injection_detected": is_injected,
    }

    if is_injected:
        updates["status"] = "refusal"
        updates["intent"] = "refusal"
        updates["error"] = injection_reason
        updates["response_text"] = "Security Alert: Input contains prohibited prompt injection pattern. Request refused."
        updates["grounded"] = False

    return updates


# Node 2: Intent Router
def node_route_intent(state: GraphState) -> Dict[str, Any]:
    if state.get("intent") == "refusal":
        return {}

    query = state.get("masked_query", "")
    
    # Check for appointment ID pattern (e.g. APP-0001) or explicit appointment lookup phrases
    app_id_match = re.search(r'\b(APP-\d{4})\b', query, re.IGNORECASE)
    
    if app_id_match:
        record_id = app_id_match.group(1).upper()
        return {
            "intent": "appointment_lookup",
            "tool_input": {"record_id": record_id}
        }
    
    lookup_keywords = ["appointment", "booking status", "my appointment", "check status"]
    if any(k in query.lower() for k in lookup_keywords) and re.search(r'\b\d{1,4}\b', query):
        num_match = re.search(r'\b\d{1,4}\b', query)
        raw_num = int(num_match.group(0))
        record_id = f"APP-{raw_num:04d}"
        return {
            "intent": "appointment_lookup",
            "tool_input": {"record_id": record_id}
        }

    return {
        "intent": "policy_rag",
        "tool_input": {"query": query}
    }


# Conditional Edge Router
def route_intent_condition(state: GraphState) -> Literal["node_appointment_tool", "node_rag_tool", "node_output_formatting"]:
    intent = state.get("intent", "policy_rag")
    if intent == "refusal":
        return "node_output_formatting"
    elif intent == "appointment_lookup":
        return "node_appointment_tool"
    else:
        return "node_rag_tool"


# Node 3: Appointment Lookup Tool
def node_appointment_tool(state: GraphState) -> Dict[str, Any]:
    tool_in = state.get("tool_input", {})
    record_id = tool_in.get("record_id", "")
    result = check_appointment_status(record_id)

    return {
        "tool_output": result,
        "response_text": result["message"],
        "escalation_flag": result.get("escalation_recommended", False),
        "grounded": True,
        "is_fallback": not result.get("found", False),
        "status": "success" if result.get("found", False) else "refusal"
    }


# Node 4: RAG Tool
def node_rag_tool(state: GraphState) -> Dict[str, Any]:
    query = state.get("masked_query", "")
    rag_result = rag_query(
        query=query,
        collection_name=COLLECTION_NAME_SENTENCE,
        similarity_threshold=RAG_SIMILARITY_THRESHOLD
    )

    top_sim = rag_result.get("top_similarity", 0.0)
    is_grounded, ground_reason = verify_output_groundedness(
        answer=rag_result["answer"],
        top_similarity=top_sim,
        threshold=RAG_SIMILARITY_THRESHOLD
    )

    return {
        "tool_output": rag_result,
        "response_text": rag_result["answer"],
        "grounded": is_grounded,
        "is_fallback": rag_result.get("is_fallback", False),
        "escalation_flag": False,
        "status": "refusal" if rag_result.get("is_fallback") else "success"
    }


# Node 5: Output Formatting & Memory Persistence
def node_output_formatting(state: GraphState) -> Dict[str, Any]:
    resp_id = f"resp-{uuid.uuid4().hex[:8]}"
    trace_id = state.get("trace_id", f"trace-{uuid.uuid4().hex[:8]}")
    conv_id = state.get("conversation_id", "default-convo")
    masked_query = state.get("masked_query", "")
    intent = state.get("intent", "policy_rag")
    status = state.get("status", "success")
    if status not in ("success", "refusal", "error"):
        status = "success"

    # Construct and validate against declared Pydantic schema
    structured = StructuredAgentResponse(
        response_id=resp_id,
        trace_id=trace_id,
        conversation_id=conv_id,
        query_masked=masked_query,
        intent=intent,
        status=status,
        data={
            "response_text": state.get("response_text", ""),
            "tool_output": state.get("tool_output", {}),
            "pii_detected": state.get("pii_detected", False),
            "injection_detected": state.get("injection_detected", False)
        },
        escalation_flag=state.get("escalation_flag", False),
        grounded=state.get("grounded", True),
        error=state.get("error") if state.get("error") else None
    )

    # Persist turns into conversation history
    append_message(conv_id, "user", masked_query)
    append_message(conv_id, "assistant", state.get("response_text", ""), metadata={"trace_id": trace_id})

    return {
        "structured_response": structured.model_dump()
    }


def build_agent_graph() -> StateGraph:
    """Construct and compile the 5-node LangGraph workflow with conditional routing."""
    workflow = StateGraph(GraphState)

    # Add all 5 nodes
    workflow.add_node("node_guardrail_input", node_guardrail_input)
    workflow.add_node("node_route_intent", node_route_intent)
    workflow.add_node("node_appointment_tool", node_appointment_tool)
    workflow.add_node("node_rag_tool", node_rag_tool)
    workflow.add_node("node_output_formatting", node_output_formatting)

    # Define edges
    workflow.set_entry_point("node_guardrail_input")
    workflow.add_edge("node_guardrail_input", "node_route_intent")

    # Conditional edge routing based on query intent
    workflow.add_conditional_edges(
        "node_route_intent",
        route_intent_condition,
        {
            "node_appointment_tool": "node_appointment_tool",
            "node_rag_tool": "node_rag_tool",
            "node_output_formatting": "node_output_formatting"
        }
    )

    # Connect tool outputs to output formatter
    workflow.add_edge("node_appointment_tool", "node_output_formatting")
    workflow.add_edge("node_rag_tool", "node_output_formatting")
    workflow.add_edge("node_output_formatting", END)

    return workflow


# Compiled default agent runnable
_AGENT_APP = None


def get_agent_app():
    """Get or compile the LangGraph runnable instance."""
    global _AGENT_APP
    if _AGENT_APP is None:
        graph = build_agent_graph()
        _AGENT_APP = graph.compile()
    return _AGENT_APP


def run_agent(
    query: str,
    conversation_id: str = "default-conversation",
    trace_id: str = None
) -> Dict[str, Any]:
    """Execute query through the compiled LangGraph workflow."""
    app = get_agent_app()
    initial_state: GraphState = {
        "conversation_id": conversation_id,
        "trace_id": trace_id or f"trace-{uuid.uuid4().hex[:8]}",
        "raw_query": query,
        "masked_query": "",
        "pii_detected": False,
        "injection_detected": False,
        "intent": "policy_rag",
        "tool_input": {},
        "tool_output": {},
        "response_text": "",
        "grounded": True,
        "is_fallback": False,
        "escalation_flag": False,
        "error": "",
        "status": "success",
        "structured_response": {}
    }

    final_state = app.invoke(initial_state)
    return final_state["structured_response"]
