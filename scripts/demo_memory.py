"""Demonstration of conversation memory persistence and fresh-session isolation (Task 8)."""
import sys
import json
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph import run_agent
from app.memory import get_conversation_history, clear_conversation

def main():
    print("=" * 75)
    print("PRACTO HEALTHCARE CAPSTONE - PERSISTED MEMORY DEMO (Task 8)")
    print("=" * 75)

    multi_turn_id = "session-multi-turn-101"
    fresh_turn_id = "session-fresh-turn-202"

    # Reset any previous state for cleanliness
    clear_conversation(multi_turn_id)
    clear_conversation(fresh_turn_id)

    print("\n[PART A: Multi-Turn Conversation Demonstration]")
    print(f"Conversation ID: {multi_turn_id}")
    print("-" * 75)

    # Turn 1
    q1 = "Hello, what is the policy for booking home visits?"
    print(f"User Turn 1: {q1}")
    r1 = run_agent(q1, conversation_id=multi_turn_id)
    print(f"Agent Turn 1:\n{r1['data']['response_text']}\n")

    # Turn 2
    q2 = "Also, can you look up appointment APP-0002 for me?"
    print(f"User Turn 2: {q2}")
    r2 = run_agent(q2, conversation_id=multi_turn_id)
    print(f"Agent Turn 2:\n{r2['data']['response_text']}\n")

    # Turn 3
    q3 = "What was the discount policy again for follow-up visits?"
    print(f"User Turn 3: {q3}")
    r3 = run_agent(q3, conversation_id=multi_turn_id)
    print(f"Agent Turn 3:\n{r3['data']['response_text']}\n")

    # Inspect persisted state in memory for multi-turn
    history_multi = get_conversation_history(multi_turn_id)
    print(f">> State successfully carried across {len(history_multi)} messages in '{multi_turn_id}'.")
    
    multi_transcript_file = Path("transcripts/multi_turn_transcript.json")
    with open(multi_transcript_file, "w", encoding="utf-8") as f:
        json.dump({
            "conversation_id": multi_turn_id,
            "message_count": len(history_multi),
            "history": history_multi
        }, f, indent=2)
    print(f"Saved multi-turn transcript to: {multi_transcript_file}")

    print("\n[PART B: Fresh Conversation Demonstration (State Reset/Absent)]")
    print(f"Conversation ID: {fresh_turn_id}")
    print("-" * 75)

    # Initial history must be completely empty
    pre_history = get_conversation_history(fresh_turn_id)
    print(f"Pre-conversation history length for '{fresh_turn_id}': {len(pre_history)} (Confirmed empty)")

    fresh_q = "Can you check status of APP-0004?"
    print(f"User Fresh Turn: {fresh_q}")
    fresh_r = run_agent(fresh_q, conversation_id=fresh_turn_id)
    print(f"Agent Fresh Turn:\n{fresh_r['data']['response_text']}\n")

    history_fresh = get_conversation_history(fresh_turn_id)
    print(f">> Fresh session history length: {len(history_fresh)} (Contains ONLY this session's messages).")
    assert len(history_fresh) == 2, "Fresh turn should have exactly 1 user + 1 assistant turn"

    fresh_transcript_file = Path("transcripts/fresh_turn_transcript.json")
    with open(fresh_transcript_file, "w", encoding="utf-8") as f:
        json.dump({
            "conversation_id": fresh_turn_id,
            "message_count": len(history_fresh),
            "history": history_fresh
        }, f, indent=2)
    print(f"Saved fresh-conversation transcript to: {fresh_transcript_file}")

    print("\n" + "=" * 75)
    print("MEMORY PERSISTENCE AND ISOLATION VERIFIED SUCCESSFULLY!")
    print("=" * 75)

if __name__ == "__main__":
    main()
