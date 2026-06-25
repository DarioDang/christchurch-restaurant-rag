"""
Drives a real multi-turn conversation against the running FastAPI server
to confirm cleanup_chat_history holds up under real (not synthetic) data,
and that the server doesn't error out as history grows past max_messages.

Run with: python test_multiturn.py
(requires `uvicorn api.main:app --reload` already running in another terminal)
"""

import requests

BASE_URL = "http://localhost:8000"

QUERIES = [
    "best sushi in Christchurch", "Italian restaurants", "is Hello Vietnam open now?",
    "cheap restaurants with delivery", "Korean food", "best pizza", "Thai restaurants",
    "Vietnamese food with takeout", "vegan options", "best coffee shops",
    "Mexican restaurants", "seafood near the city center", "best ramen",
    "Chinese restaurants open late", "budget-friendly dinner spots",
]


def validate_pairing(messages):
    violations = []
    for i, msg in enumerate(messages):
        if msg.get("type") == "function_call_output":
            if i == 0 or messages[i - 1].get("type") != "function_call" or \
               messages[i - 1].get("call_id") != msg.get("call_id"):
                violations.append(f"Orphaned output at index {i}")
        if msg.get("type") == "function_call":
            if i + 1 >= len(messages) or messages[i + 1].get("type") != "function_call_output":
                violations.append(f"Unpaired function_call at index {i}")
    return violations


def main():
    messages = []
    for turn, query in enumerate(QUERIES, 1):
        messages.append({"role": "user", "content": query})
        print(f"\n--- Turn {turn}: \"{query}\" ---")

        resp = requests.post(f"{BASE_URL}/api/chat", json={"messages": messages}, timeout=180)

        if resp.status_code != 200:
            print(f"❌ FAILED at turn {turn}: HTTP {resp.status_code}")
            print(resp.text[:1000])
            return

        data = resp.json()
        messages = data["messages"]

        violations = validate_pairing(messages)
        dev_count = sum(1 for m in messages if m.get("role") == "developer")

        print(f"  History length: {len(messages)} | Developer messages: {dev_count} | "
              f"Pairing violations: {len(violations)}")
        if violations:
            print(f"  ❌ {violations}")
            return
        if dev_count != 1:
            print(f"  ❌ Expected exactly 1 developer message, found {dev_count}")
            return
        if not data.get("reply"):
            print(f"  ⚠️  No reply text returned")

    print(f"\n✅ All {len(QUERIES)} turns completed successfully. Final history length: {len(messages)}")


if __name__ == "__main__":
    main()