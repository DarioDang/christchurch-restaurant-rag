# core/history.py

"""
Conversation history management for the stateless chat API.
Two responsibilities: (1) prevent client-supplied history from overriding
the agent's instructions, (2) truncate long histories without breaking
function_call / function_call_output pairing.
"""

from typing import List, Dict, Any
from prompt import DEVELOPER_PROMPT

def sanitize_client_history(client_messages: List[Dict[str,Any]]) -> List[Dict[str, Any]]:
    """
    Build the authoritative history for an LLM turn.

    The developer prompt is always injected server - side. Any developer / system - role entries sent by
    the client are dropped first, so a tampered client payload cant override agent instructions by smuggling 
    its own developer message.
    """

    sanitized = [m for m in client_messages if m.get("role") not in ("developer", "system")]
    return [{"role": "developer", "content": DEVELOPER_PROMPT}] + sanitized

def cleanup_chat_history(messages: List[Dict[str, Any]], max_messages: int = 25) -> List[Dict[str, Any]]:
    """
    Truncate conversation history to `max_messages`, preserving function_call /
    function_call_output pairing.

    Naively slicing the last N messages risks starting the kept window in the
    middle of a tool-call pair — e.g. keeping a function_call_output but cutting
    off the function_call that preceded it. The OpenAI Responses API rejects a
    history shaped like that. This walks backward from the truncation point to
    make sure pairs stay whole, and drops a trailing function_call that has no
    output rather than send an incomplete one.
    """
    if len(messages) <= max_messages:
        return messages

    # developer/system prompt(s) always survive truncation
    head = [m for m in messages if m.get("role") in ("developer", "system")]
    body = [m for m in messages if m.get("role") not in ("developer", "system")]

    keep_count = max_messages - len(head)
    if keep_count <= 0:
        return head

    cut_index = max(len(body) - keep_count, 0)
    tail = body[cut_index:]

    # If we cut right before a function_call_output, its matching function_call
    # got left behind in the dropped portion — pull it back in.
    if tail and tail[0].get("type") == "function_call_output":
        target_call_id = tail[0].get("call_id")
        for i in range(cut_index - 1, -1, -1):
            candidate = body[i]
            if candidate.get("type") == "function_call" and candidate.get("call_id") == target_call_id:
                tail = body[i:]
                break

    # If the last kept message is a function_call with no following output
    # (because the output got trimmed off the far end — shouldn't normally
    # happen since we trim from the front, but cheap to guard), drop it.
    if tail and tail[-1].get("type") == "function_call":
        tail = tail[:-1]

    return head + tail 