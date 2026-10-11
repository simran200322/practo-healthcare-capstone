"""Unit tests for Part 4: FastMCP protocol, SQLite Checkpointing, Retries, and Timeouts."""
import pytest
import asyncio
import importlib.util
from pathlib import Path

_client_path = Path(__file__).resolve().parent.parent / "mcp" / "client.py"
_spec = importlib.util.spec_from_file_location("local_mcp_client", _client_path)
_mcp_client_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mcp_client_module)
run_mcp_client = _mcp_client_module.run_mcp_client
from resilience.checkpoint_demo import run_checkpoint_demonstration
from resilience.retry_demo import (
    execute_with_exponential_backoff,
    simulate_transient_node,
    _CALL_COUNTER
)
import resilience.retry_demo as retry_module
from resilience.timeout_demo import (
    execute_node_with_timeout,
    execute_graph_with_global_timeout
)


@pytest.mark.asyncio
async def test_mcp_client_tool_call_two_records():
    """Verify FastMCP client invokes check_appointment_status for >= 2 records."""
    results = await run_mcp_client(target_record_ids=["APP-0001", "APP-0002"])
    assert len(results) >= 2
    assert results[0]["record_id"] == "APP-0001"
    assert "Dermatology" in results[0]["response"]
    assert results[1]["record_id"] == "APP-0002"
    assert "Pediatrics" in results[1]["response"]


def test_sqlite_checkpointing_interruption_and_resumption():
    """Verify SQLite checkpointer pauses at node 2 and resumes without re-executing nodes 1 & 2."""
    final_state = run_checkpoint_demonstration()
    assert final_state is not None
    assert "APP-0001" in final_state["response_text"]


def test_exponential_backoff_retry_recovery():
    """Verify retry policy recovers transient failure within configured attempts."""
    retry_module._CALL_COUNTER = 0
    res = execute_with_exponential_backoff(simulate_transient_node, record_id="APP-0003")
    assert res["status"] == "success"
    assert res["resolved_attempt"] == 3


@pytest.mark.asyncio
async def test_per_node_timeout_clean_error():
    """Verify per-node timeout aborts with clean error without hanging."""
    res = await execute_node_with_timeout("APP-0001", node_timeout=0.10)
    assert res["status"] == "error"
    assert res["error_type"] == "NodeTimeoutError"
    assert res["is_hang"] is False


@pytest.mark.asyncio
async def test_global_graph_timeout_cancellation():
    """Verify global timeout cancels the run on overrun."""
    res = await execute_graph_with_global_timeout(global_budget=0.20)
    assert res["status"] == "cancelled"
    assert res["error_type"] == "GlobalGraphTimeoutError"
    assert res["clean_shutdown"] is True
