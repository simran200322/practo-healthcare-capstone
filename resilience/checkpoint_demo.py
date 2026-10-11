"""Demonstration of SQLite-based LangGraph Checkpointing, Interruption, and Resumption (Task 15)."""
import sys
import sqlite3
import json
from pathlib import Path
from typing import Dict, Any, Literal
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import CHECKPOINTS_DB_FILE
from app.guardrails import mask_contact_numbers, detect_prompt_injection
from app.tools import check_appointment_status

# Global node execution counters to provide undeniable empirical proof of checkpoint recovery
NODE_EXECUTION_LOG = {
    "node_guardrail_input": 0,
    "node_route_intent": 0,
    "node_appointment_tool": 0,
    "node_output_formatting": 0
}


class ResilientState(TypedDict):
    thread_id: str
    query: str
    masked_query: str
    intent: str
    record_id: str
    tool_output: Dict[str, Any]
    response_text: str
    nodes_executed: list


def tracked_node_guardrail(state: ResilientState) -> Dict[str, Any]:
    NODE_EXECUTION_LOG["node_guardrail_input"] += 1
    print(f"  [NODE 1 EXECUTING] node_guardrail_input (Run #{NODE_EXECUTION_LOG['node_guardrail_input']})")
    masked, _ = mask_contact_numbers(state["query"])
    return {
        "masked_query": masked,
        "nodes_executed": state.get("nodes_executed", []) + ["node_guardrail_input"]
    }


def tracked_node_router(state: ResilientState) -> Dict[str, Any]:
    NODE_EXECUTION_LOG["node_route_intent"] += 1
    print(f"  [NODE 2 EXECUTING] node_route_intent (Run #{NODE_EXECUTION_LOG['node_route_intent']})")
    # Extract record ID
    import re
    match = re.search(r'APP-\d{4}', state["masked_query"])
    rec_id = match.group(0) if match else "APP-0001"
    return {
        "intent": "appointment_lookup",
        "record_id": rec_id,
        "nodes_executed": state.get("nodes_executed", []) + ["node_route_intent"]
    }


def tracked_node_appointment(state: ResilientState) -> Dict[str, Any]:
    NODE_EXECUTION_LOG["node_appointment_tool"] += 1
    print(f"  [NODE 3 EXECUTING] node_appointment_tool (Run #{NODE_EXECUTION_LOG['node_appointment_tool']})")
    res = check_appointment_status(state["record_id"])
    return {
        "tool_output": res,
        "response_text": res["message"],
        "nodes_executed": state.get("nodes_executed", []) + ["node_appointment_tool"]
    }


def tracked_node_output(state: ResilientState) -> Dict[str, Any]:
    NODE_EXECUTION_LOG["node_output_formatting"] += 1
    print(f"  [NODE 4 EXECUTING] node_output_formatting (Run #{NODE_EXECUTION_LOG['node_output_formatting']})")
    return {
        "nodes_executed": state.get("nodes_executed", []) + ["node_output_formatting"]
    }


def build_resilient_checkpoint_graph(checkpointer: SqliteSaver):
    builder = StateGraph(ResilientState)
    builder.add_node("node_guardrail_input", tracked_node_guardrail)
    builder.add_node("node_route_intent", tracked_node_router)
    builder.add_node("node_appointment_tool", tracked_node_appointment)
    builder.add_node("node_output_formatting", tracked_node_output)

    builder.set_entry_point("node_guardrail_input")
    builder.add_edge("node_guardrail_input", "node_route_intent")
    builder.add_edge("node_route_intent", "node_appointment_tool")
    builder.add_edge("node_appointment_tool", "node_output_formatting")
    builder.add_edge("node_output_formatting", END)

    # Deliberately interrupt before node_appointment_tool so exactly 2 nodes execute in phase 1
    return builder.compile(
        checkpointer=checkpointer,
        interrupt_before=["node_appointment_tool"]
    )


def run_checkpoint_demonstration():
    print("=" * 80)
    print("PRACTO HEALTHCARE CAPSTONE - SQLITE CHECKPOINTING & RESUMPTION (Task 15)")
    print("=" * 80)
    
    # Ensure fresh test database file
    db_path = CHECKPOINTS_DB_FILE
    if db_path.exists():
        try:
            db_path.unlink()
        except Exception:
            pass

    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    checkpointer = SqliteSaver(conn)

    app = build_resilient_checkpoint_graph(checkpointer)
    thread_id = "practo-clinical-thread-99"
    config = {"configurable": {"thread_id": thread_id}}

    print(f"\n[PHASE A: Launching Initial Workflow Run with Thread ID: '{thread_id}']")
    print("Configured to deliberately INTERRUPT after Node 2 before Node 3 executes.")
    print("-" * 80)

    initial_input: ResilientState = {
        "thread_id": thread_id,
        "query": "Check appointment status for APP-0001 please",
        "masked_query": "",
        "intent": "",
        "record_id": "",
        "tool_output": {},
        "response_text": "",
        "nodes_executed": []
    }

    # Execute Part 1 until the interrupt point
    state_at_interrupt = app.invoke(initial_input, config=config)

    print("\n>> DELIBERATE INTERRUPTION TRIGGERED!")
    print(f">> State successfully captured and serialized into '{db_path.name}'.")
    print(f">> Nodes executed so far: {state_at_interrupt.get('nodes_executed')}")
    print(f">> Current Node Run Counts: {NODE_EXECUTION_LOG}")
    assert NODE_EXECUTION_LOG["node_guardrail_input"] == 1
    assert NODE_EXECUTION_LOG["node_route_intent"] == 1
    assert NODE_EXECUTION_LOG["node_appointment_tool"] == 0
    assert NODE_EXECUTION_LOG["node_output_formatting"] == 0

    print("\n" + "-" * 80)
    print(f"[PHASE B: Resuming Same Thread ID '{thread_id}' from SQLite Checkpoint]")
    print("Graph will resume without re-executing completed Nodes 1 & 2.")
    print("-" * 80)

    # Resuming execution passing None as input to load stored checkpoint
    final_resumed_state = app.invoke(None, config=config)

    print("\n>> WORKFLOW COMPLETED SUCCESSFULLY UPON RESUMPTION!")
    print(f">> Total nodes recorded in final state: {final_resumed_state.get('nodes_executed')}")
    print(f">> Final Node Execution Counts across entire lifecycle:")
    for node, count in NODE_EXECUTION_LOG.items():
        print(f"     - {node:25s}: Executed {count} time(s)")

    print("\n>> CHECKPOINT VERIFICATION PROOF:")
    print(f"     Node 1 executed count = {NODE_EXECUTION_LOG['node_guardrail_input']} (Loaded from SQLite, NOT re-executed)")
    print(f"     Node 2 executed count = {NODE_EXECUTION_LOG['node_route_intent']} (Loaded from SQLite, NOT re-executed)")
    print(f"     Node 3 executed count = {NODE_EXECUTION_LOG['node_appointment_tool']} (Executed upon resumption)")
    print(f"     Node 4 executed count = {NODE_EXECUTION_LOG['node_output_formatting']} (Executed upon resumption)")
    
    assert NODE_EXECUTION_LOG["node_guardrail_input"] == 1, "Node 1 should NOT have re-executed"
    assert NODE_EXECUTION_LOG["node_route_intent"] == 1, "Node 2 should NOT have re-executed"
    assert NODE_EXECUTION_LOG["node_appointment_tool"] == 1, "Node 3 should execute once upon resume"
    assert NODE_EXECUTION_LOG["node_output_formatting"] == 1, "Node 4 should execute once upon resume"

    print(f"\nFinal Appointment Response: {final_resumed_state['response_text']}")
    print("=" * 80)
    print("SQLITE CHECKPOINTING & RESUMPTION VERIFIED WITH 100% MATHEMATICAL CERTAINTY!")
    print("=" * 80)

    conn.close()
    return final_resumed_state


if __name__ == "__main__":
    run_checkpoint_demonstration()
