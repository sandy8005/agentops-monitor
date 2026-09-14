# test_experience_reconcile.py
from parser import _reconcile_experience

# LLM claims 4 years; itemized roles sum to 1.5 -> should use grounded 1.5 and flag
p = {"years_experience": 4.0,
     "experience": [{"title": "A", "company": "X", "years": 1.0},
                    {"title": "B", "company": "Y", "years": 0.5}]}
out = _reconcile_experience(p)
print("used:", out["years_experience"], "| stated:", out["years_experience_stated"],
      "| summed:", out["years_experience_summed"])
print("discrepancy:", out["experience_discrepancy"] is not None)
# expect: used 1.5, stated 4.0, summed 1.5, discrepancy True

# Agreement within tolerance -> keep stated, no flag
p2 = {"years_experience": 3.0,
      "experience": [{"title": "A", "company": "X", "years": 2.0},
                     {"title": "B", "company": "Y", "years": 1.0}]}
out2 = _reconcile_experience(p2)
print("used:", out2["years_experience"], "| discrepancy:", out2["experience_discrepancy"] is not None)
# expect: used 3.0, discrepancy False