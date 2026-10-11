"""FastAPI application exposing REST endpoints for the Practo Domain Support Agent."""
import time
import uuid
from typing import Dict, Any
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware

from app.schemas import AskRequest, AddDocumentRequest, StructuredAgentResponse
from app.graph import run_agent
from app.chunking import fixed_size_chunking, sentence_based_chunking
from app.retrieval import index_chunks
from app.logging_config import log_request_event
from app.config import (
    COLLECTION_NAME_FIXED,
    COLLECTION_NAME_SENTENCE,
    CHROMA_FIXED_DIR,
    CHROMA_SENTENCE_DIR,
    MOCK_LLM,
    BASE_DIR
)

app = FastAPI(
    title="Practo Healthcare Domain Support Agent API",
    description="Production-grade domain support API answering clinic policy questions and checking appointments.",
    version="1.0.0"
)

# Enable CORS for frontend or local integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", tags=["UI"])
def serve_index_ui():
    """Serve the interactive web interface."""
    index_file = BASE_DIR / "app" / "static" / "index.html"
    if index_file.exists():
        return FileResponse(index_file)
    return {"message": "Practo Healthcare Agent API is running. Access /docs for Swagger UI."}


@app.get("/health", tags=["System"])
def health_check() -> Dict[str, Any]:
    """Health check endpoint reporting agent readiness and configuration mode."""
    return {
        "status": "healthy",
        "mock_llm_mode": MOCK_LLM,
        "collections": [COLLECTION_NAME_FIXED, COLLECTION_NAME_SENTENCE],
        "message": "Practo Support Agent is operational."
    }


@app.post("/ask", response_model=StructuredAgentResponse, tags=["Agent"])
def ask_agent(request: AskRequest) -> Dict[str, Any]:
    """Execute patient query or appointment lookup through the LangGraph workflow."""
    start_time = time.time()
    trace_id = f"trace-{uuid.uuid4().hex[:10]}"
    conv_id = request.conversation_id or "default-conversation"

    try:
        # Run through LangGraph workflow
        agent_response = run_agent(
            query=request.query,
            conversation_id=conv_id,
            trace_id=trace_id
        )
        duration_ms = (time.time() - start_time) * 1000.0

        # Secure structured JSON-Lines logging (PII masked automatically)
        log_request_event(
            trace_id=trace_id,
            endpoint="/ask",
            raw_query=request.query,
            status=agent_response.get("status", "success"),
            duration_ms=duration_ms,
            extra={"intent": agent_response.get("intent")}
        )

        return agent_response

    except Exception as exc:
        duration_ms = (time.time() - start_time) * 1000.0
        log_request_event(
            trace_id=trace_id,
            endpoint="/ask",
            raw_query=request.query,
            status="error",
            duration_ms=duration_ms,
            extra={"error": str(exc)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent workflow encountered an unexpected error: {str(exc)}"
        )


@app.post("/add-document", tags=["Knowledge Base"])
def add_document(request: AddDocumentRequest) -> Dict[str, Any]:
    """Ingest, chunk, embed, and index a new policy document into both vector collections."""
    start_time = time.time()
    trace_id = f"trace-{uuid.uuid4().hex[:10]}"

    if len(request.content.strip()) < 10:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Document content must contain at least 10 characters."
        )

    try:
        doc_id = f"KB-DOC-EXT-{uuid.uuid4().hex[:6]}"
        doc_item = [{
            "doc_id": doc_id,
            "title": request.title.strip(),
            "topic": request.topic.strip(),
            "content": request.content.strip()
        }]

        # 1. Chunk with fixed-size strategy and index
        fixed_chunks = fixed_size_chunking(doc_item, chunk_size=180, chunk_overlap=40)
        index_chunks(
            chunks=fixed_chunks,
            collection_name=COLLECTION_NAME_FIXED,
            persist_dir=CHROMA_FIXED_DIR,
            recreate=False  # Append/upsert into existing collection
        )

        # 2. Chunk with sentence-based strategy and index
        sentence_chunks = sentence_based_chunking(doc_item)
        index_chunks(
            chunks=sentence_chunks,
            collection_name=COLLECTION_NAME_SENTENCE,
            persist_dir=CHROMA_SENTENCE_DIR,
            recreate=False  # Append/upsert into existing collection
        )

        duration_ms = (time.time() - start_time) * 1000.0
        log_request_event(
            trace_id=trace_id,
            endpoint="/add-document",
            raw_query=f"Title: {request.title} | Topic: {request.topic}",
            status="success",
            duration_ms=duration_ms,
            extra={"doc_id": doc_id, "fixed_chunks": len(fixed_chunks), "sentence_chunks": len(sentence_chunks)}
        )

        return {
            "status": "success",
            "message": "Document indexed successfully into both vector collections.",
            "doc_id": doc_id,
            "title": request.title,
            "fixed_chunks_added": len(fixed_chunks),
            "sentence_chunks_added": len(sentence_chunks),
            "trace_id": trace_id
        }

    except Exception as exc:
        duration_ms = (time.time() - start_time) * 1000.0
        log_request_event(
            trace_id=trace_id,
            endpoint="/add-document",
            raw_query=request.title,
            status="error",
            duration_ms=duration_ms,
            extra={"error": str(exc)}
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to index document: {str(exc)}"
        )
