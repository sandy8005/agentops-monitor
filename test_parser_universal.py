# test_parser_universal.py
from parser import _normalize_experience, _normalize_education, _to_float

# experience with the "wrong" keys (role/months) — should normalize, not crash
exp = _normalize_experience([
    {"role": "Software Engineer", "company": "X", "months": 24},   # role+months
    {"title": "Dev", "company": "Y", "years": 3},                  # already correct
    {"position": "Intern", "employer": "Z"},                       # position+employer, no duration
])
print("experience:", exp)
# expect: title/company/years for all three; months 24 -> 2.0 years

edu = _normalize_education([
    {"degree": "MS", "university": "LTU", "year": 2025},           # int year, 'university' key
])
print("education:", edu)
# expect: year "2025" as string, institution "LTU"

print("_to_float tests:", _to_float(2025), _to_float("24 months"), _to_float(None))