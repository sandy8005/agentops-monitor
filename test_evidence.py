# test_evidence.py
from scorer import calculate_match_score
parsed = {"skills":["python","flask"],"years_experience":3,"projects":[{"name":"a","tech":["python"]}],"education":[]}
reqs = {"required_skills":["python","flask"],"required_any_of":[],"preferred_skills":["docker"],"min_years_experience":2,"responsibilities":[]}
r = calculate_match_score(parsed, reqs, "python flask dev", None, None)
print("missing (required):", r["missing_skills"])       # [] — docker NOT here
print("missing preferred:", r["missing_preferred"])     # ['docker'] — informational