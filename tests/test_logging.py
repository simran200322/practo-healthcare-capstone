"""Unit tests verifying structured JSON-Lines logging and mandatory PII redaction on disk."""
import json
import pytest
from pathlib import Path
from app.logging_config import log_request_event

def test_structured_logging_and_pii_masking_on_disk(tmp_path):
    test_log_file = tmp_path / "test_requests.jsonl"
    
    # Query with raw contact number
    raw_query = "My phone number is 9876543210. How do I get insurance reimbursement?"
    trace_id = "trace-test-12345"

    entry = log_request_event(
        trace_id=trace_id,
        endpoint="/ask",
        raw_query=raw_query,
        status="success",
        duration_ms=42.5,
        log_file=test_log_file
    )

    # 1. Assert return entry is masked
    assert entry["pii_masked"] is True
    assert "9876543210" not in entry["query_masked"]
    assert "[REDACTED_PHONE]" in entry["query_masked"]
    assert entry["trace_id"] == trace_id

    # 2. Assert file on disk was created and read raw content
    assert test_log_file.exists()
    raw_file_content = test_log_file.read_text(encoding="utf-8")
    
    # 3. Mandatory guarantee: Unmasked phone number must NEVER exist in file content
    assert "9876543210" not in raw_file_content, "CRITICAL: Raw PII was leaked to log file on disk!"
    assert "[REDACTED_PHONE]" in raw_file_content

    # 4. Assert valid JSON-Lines entry
    parsed_line = json.loads(raw_file_content.strip())
    assert parsed_line["trace_id"] == trace_id
    assert parsed_line["duration_ms"] == 42.5
    assert parsed_line["status"] == "success"
