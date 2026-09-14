# test_months.py
from parser import _normalize_experience
r = _normalize_experience([{"title": "Dev", "company": "X", "months": "24 months"}])
print("24 months ->", r[0]["years"], "years")   # should be 2.0, not 0.17
r2 = _normalize_experience([{"title": "Dev", "company": "Y", "months": 6}])
print("6 months ->", r2[0]["years"], "years")     # should be 0.5