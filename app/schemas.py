"""Pydantic schemas and structured output models for Practo Support Agent."""
from typing import Optional, List, Dict, Any, Literal
from pydantic import BaseModel, Field


class AgentState(BaseModel):
    """LangGraph internal state."""
    conversation_id: str = "default-conversation"
    trace_id: str = "trace-default"
    raw_query: str = ""
    masked_query: str = ""
    pii_detected: bool = False
    injection_detected: bool = False
    intent: Literal["policy_rag", "appointment_lookup", "unsupported", "error"] = "policy_rag"
    tool_input: Dict[str, Any] = Field(default_factory=dict)
    tool_output: Dict[str, Any] = Field(default_factory=dict)
    response_text: str = ""
    grounded: bool = True
    is_fallback: bool = False
    escalation_flag: bool = False
    error: Optional[str] = None


class StructuredAgentResponse(BaseModel):
    """Declared JSON schema for every agent response as mandated by Capstone Part 2 Task 9."""
    response_id: str = Field(description="Unique identifier for the agent response")
    trace_id: str = Field(description="Correlation trace ID for request observability")
    conversation_id: str = Field(description="Conversation session ID")
    query_masked: str = Field(description="Query with PII masked before model/log exposure")
    intent: str = Field(description="Identified routing intent: policy_rag or appointment_lookup")
    status: Literal["success", "refusal", "error"] = Field(description="Execution status")
    data: Dict[str, Any] = Field(description="Structured payload containing answer or appointment info")
    escalation_flag: bool = Field(description="Whether clinical escalation is recommended")
    grounded: bool = Field(description="Whether the response is grounded in verified sources")
    error: Optional[str] = Field(default=None, description="Error message if applicable")


class AskRequest(BaseModel):
    """FastAPI request schema for POST /ask."""
    query: str = Field(..., min_length=2, max_length=1000, description="User question or appointment inquiry")
    conversation_id: Optional[str] = Field(default="default-conversation", description="Conversation thread ID")


class AddDocumentRequest(BaseModel):
    """FastAPI request schema for POST /add-document."""
    title: str = Field(..., min_length=3, description="Document title")
    topic: str = Field(..., min_length=3, description="Policy topic")
    content: str = Field(..., min_length=10, description="Substantive policy content")
