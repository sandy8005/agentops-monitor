from agent import normalize_decision

cases = ["Apply", "Apply.", "  apply  ", "Maybe - good fit",
         "I'd say Skip", "SKIP", "Definitely apply!", "unclear"]
for c in cases:
    print(f"{c!r:25} -> {normalize_decision(c)}")