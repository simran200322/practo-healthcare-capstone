"""Conversation memory manager persisting multi-turn dialogue state to a JSON file."""
import json
import time
from typing import List, Dict, Any, Optional
from pathlib import Path
from app.config import MEMORY_FILE


def _load_all_conversations(filepath: Path = MEMORY_FILE) -> Dict[str, List[Dict[str, Any]]]:
    """Load the full conversation map from the JSON storage file."""
    if not filepath.exists():
        return {}
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def _save_all_conversations(data: Dict[str, List[Dict[str, Any]]], filepath: Path = MEMORY_FILE) -> None:
    """Save the conversation map to the JSON storage file."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def append_message(
    conversation_id: str,
    role: str,
    content: str,
    metadata: Optional[Dict[str, Any]] = None,
    filepath: Path = MEMORY_FILE
) -> None:
    """Append a user or assistant message to the persistent conversation history."""
    all_convos = _load_all_conversations(filepath)
    if conversation_id not in all_convos:
        all_convos[conversation_id] = []

    entry = {
        "timestamp": time.time(),
        "role": role,
        "content": content,
        "metadata": metadata or {}
    }
    all_convos[conversation_id].append(entry)
    _save_all_conversations(all_convos, filepath)


def get_conversation_history(
    conversation_id: str,
    filepath: Path = MEMORY_FILE
) -> List[Dict[str, Any]]:
    """Retrieve all prior messages for a given conversation ID."""
    all_convos = _load_all_conversations(filepath)
    return all_convos.get(conversation_id, [])


def clear_conversation(
    conversation_id: str,
    filepath: Path = MEMORY_FILE
) -> None:
    """Clear memory for a specific conversation session."""
    all_convos = _load_all_conversations(filepath)
    if conversation_id in all_convos:
        del all_convos[conversation_id]
        _save_all_conversations(all_convos, filepath)


def reset_all_memory(filepath: Path = MEMORY_FILE) -> None:
    """Reset the entire conversation memory storage."""
    if filepath.exists():
        filepath.unlink()
