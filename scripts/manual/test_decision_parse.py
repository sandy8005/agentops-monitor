# test_decision_parse.py
from agent import parse_decision

# Valid structured responses — the new format
print(parse_decision('{"decision": "Apply", "reason": "strong match"}'))   # Apply
print(parse_decision('{"decision": "Skip", "reason": "do not apply here"}')) # Skip (word "apply" in reason no longer confuses it)
print(parse_decision('```json\n{"decision": "Maybe", "reason": "partial"}\n```'))  # Maybe (handles fences)

# Invalid decision value — Pydantic rejects it
try:
    parse_decision('{"decision": "Probably", "reason": "x"}')
    print("FAIL: should have rejected")
except Exception:
    print("OK: rejected invalid decision value")