# test_normalize.py
from agent import normalize_decision
cases = [
    "Apply", "Apply.", "Maybe - good fit", "I'd say Skip", "SKIP",
    "Definitely apply!", "unclear",
    "Do not apply", "don't apply to this one",
    "Maybe, but do not apply yet", "avoid applying here",
]
for c in cases:
    print(f"{c!r:32} -> {normalize_decision(c)}")