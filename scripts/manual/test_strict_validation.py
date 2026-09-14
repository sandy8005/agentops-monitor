# test_strict_validation.py
from schemas import _strict_float, Evaluation, ParsedResume
import pydantic

print(_strict_float(2), _strict_float("2 years"), _strict_float("24 months"))  # 2.0 2.0 2.0

for bad in ["about two", "several", "N/A"]:
    try:
        _strict_float(bad, "years"); print(f"FAIL: '{bad}' should raise")
    except ValueError:
        print(f"OK: '{bad}' rejected")

try:
    Evaluation(relevance_score=99, faithfulness_score=-2, completeness_score=5,
               hallucination_detected=False, hallucinated_claims=[], notes="x")
    print("FAIL: out-of-range should raise")
except pydantic.ValidationError:
    print("OK: out-of-range evaluation scores rejected")