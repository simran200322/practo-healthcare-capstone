"""Demonstration of per-node timeout and global graph cancellation (Task 16)."""
import asyncio
import time
from typing import Dict, Any

PER_NODE_TIMEOUT_SECONDS = 0.40
GLOBAL_GRAPH_TIMEOUT_SECONDS = 0.80


async def sluggish_clinical_db_node(record_id: str, delay: float = 1.50) -> Dict[str, Any]:
    """Simulate a hanging or sluggish database call."""
    print(f"  [NODE STARTED] Requesting {record_id} (simulated delay: {delay:.2f}s)...")
    await asyncio.sleep(delay)
    return {"record_id": record_id, "data": "Appointment payload"}


async def execute_node_with_timeout(
    record_id: str,
    node_timeout: float = PER_NODE_TIMEOUT_SECONDS
) -> Dict[str, Any]:
    """Execute a single node wrapped in an explicit per-node timeout."""
    start = time.time()
    try:
        result = await asyncio.wait_for(
            sluggish_clinical_db_node(record_id, delay=1.50),
            timeout=node_timeout
        )
        return {"status": "success", "result": result}
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"  [PER-NODE TIMEOUT FIRED] Clean error caught after {elapsed:.2f}s (budget: {node_timeout:.2f}s)!")
        return {
            "status": "error",
            "error_type": "NodeTimeoutError",
            "node_name": "sluggish_clinical_db_node",
            "detail": f"Node exceeded execution budget of {node_timeout:.2f}s (aborted cleanly after {elapsed:.2f}s).",
            "is_hang": False
        }


async def multi_stage_workflow_step(step_name: str, duration: float) -> str:
    print(f"    --> Step '{step_name}' running for {duration:.2f}s...")
    await asyncio.sleep(duration)
    return f"Step {step_name} completed"


async def full_graph_workflow() -> Dict[str, Any]:
    """A multi-step graph workflow that overruns cumulative global time budget."""
    # Step 1 takes 0.5s
    res1 = await multi_stage_workflow_step("Guardrail Triage", 0.50)
    # Step 2 takes 0.5s -> Total 1.0s, which overruns the 0.80s global budget
    res2 = await multi_stage_workflow_step("Comprehensive RAG Lookup", 0.50)
    return {"step1": res1, "step2": res2}


async def execute_graph_with_global_timeout(
    global_budget: float = GLOBAL_GRAPH_TIMEOUT_SECONDS
) -> Dict[str, Any]:
    """Execute entire graph protected by a global cancellation timeout."""
    start = time.time()
    try:
        result = await asyncio.wait_for(
            full_graph_workflow(),
            timeout=global_budget
        )
        return {"status": "success", "result": result}
    except asyncio.TimeoutError:
        elapsed = time.time() - start
        print(f"  [GLOBAL TIMEOUT TRIGGERED] Graph cancelled after {elapsed:.2f}s exceeding budget of {global_budget:.2f}s!")
        return {
            "status": "cancelled",
            "error_type": "GlobalGraphTimeoutError",
            "detail": f"Graph run cancelled: total elapsed time {elapsed:.2f}s exceeded global limit of {global_budget:.2f}s.",
            "clean_shutdown": True
        }


async def run_timeout_demonstrations():
    print("=" * 75)
    print("PRACTO HEALTHCARE CAPSTONE - PER-NODE & GLOBAL TIMEOUT DEMO (Task 16)")
    print("=" * 75)

    print("\n[PART A: Per-Node Timeout Demonstration]")
    print(f"Configured Per-Node Timeout: {PER_NODE_TIMEOUT_SECONDS:.2f}s | Simulated Delay: 1.50s")
    print("-" * 75)
    node_res = await execute_node_with_timeout("APP-0001", node_timeout=PER_NODE_TIMEOUT_SECONDS)
    print(f"Result Status : {node_res['status']}")
    print(f"Error Type    : {node_res['error_type']}")
    print(f"Detail        : {node_res['detail']}")
    print(f"Clean (No Hang): {not node_res['is_hang']}")
    assert node_res["status"] == "error"
    assert node_res["error_type"] == "NodeTimeoutError"

    print("\n" + "-" * 75)
    print("[PART B: Global Graph Cancellation Timeout Demonstration]")
    print(f"Configured Global Timeout: {GLOBAL_GRAPH_TIMEOUT_SECONDS:.2f}s | Cumulative Steps: 1.00s")
    print("-" * 75)
    graph_res = await execute_graph_with_global_timeout(global_budget=GLOBAL_GRAPH_TIMEOUT_SECONDS)
    print(f"Result Status : {graph_res['status']}")
    print(f"Error Type    : {graph_res['error_type']}")
    print(f"Detail        : {graph_res['detail']}")
    print(f"Clean Shutdown: {graph_res['clean_shutdown']}")
    assert graph_res["status"] == "cancelled"
    assert graph_res["error_type"] == "GlobalGraphTimeoutError"

    print("\n" + "=" * 75)
    print("PER-NODE & GLOBAL TIMEOUTS VERIFIED WITH CLEAN CANCELLATION!")
    print("=" * 75)


def main():
    asyncio.run(run_timeout_demonstrations())


if __name__ == "__main__":
    main()
