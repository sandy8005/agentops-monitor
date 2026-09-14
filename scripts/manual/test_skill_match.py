# test_skill_match.py
from scorer import _skill_present
import re

def check(skill, resume):
    low = resume.lower()
    tokens = set(re.findall(r"[a-z0-9\+\#\.]+", low))
    return _skill_present(skill, low, tokens)

cases = [
    ("Go", "Experienced Django developer", False),   # the bug you found
    ("Go", "I write Go and Rust", True),             # real Go match
    ("R", "React and Redux expert", False),          # R inside React
    ("Java", "JavaScript and TypeScript", False),    # Java inside JavaScript
    ("Python", "Senior Python engineer", True),
    ("machine learning", "did machine learning research", True),
    ("c++", "strong c++ background", True),
]
for skill, resume, expected in cases:
    got = check(skill, resume)
    mark = "OK" if got == expected else "FAIL"
    print(f"{mark}  {skill!r:18} in {resume!r:35} -> {got} (expected {expected})")