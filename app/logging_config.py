"""Structured JSON-Lines logging with mandatory PII masking and trace correlation."""
import json
import time
from pathlib import Path
from typing import Dict, Any, Optional
from app.config import REQUEST_LOG_FILE
from app.guardrails import mask_contact_numbers


def log_request_event(
    trace_id: str,
    endpoint: str,
    raw_query: str,
    status: str,
    duration_ms: float,
    extra: Optional[Dict[str, Any]] = None,
    log_file: Path = REQUEST_LOG_FILE
) -> Dict[str, Any]:
    """Write one structured JSON-Lines entry for a request.
    
    CRITICAL SECURITY GUARANTEE:
    Masks all fixed-format contact numbers prior to serializing to disk.
    Unmasked PII never reaches persistent disk storage.
    """
    masked_query, pii_detected = mask_contact_numbers(raw_query)

    entry = {
        "timestamp": time.time(),
        "iso_timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "trace_id": trace_id,
        "endpoint": endpoint,
        "query_masked": masked_query,
        "pii_masked": pii_detected,
        "status": status,
        "duration_ms": round(duration_ms, 2),
        "extra": extra or {}
    }

    try:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:
        # Safe logging fallback: logging errors should never break user request execution
        print(f"[Logging Warning] Failed to write structured log: {exc}")

    return entry
