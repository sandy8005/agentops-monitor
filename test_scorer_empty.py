from scorer import calculate_match_score

parsed = {"skills": ["python"], "years_experience": 3,
          "projects": [{"name": "x", "tech": ["python"]}], "education": []}
empty_reqs = {"required_skills": [], "preferred_skills": [],
              "min_years_experience": 0, "responsibilities": []}

result = calculate_match_score(parsed, empty_reqs, "python developer resume")
print("Score:", result["score"])
print("Decision:", result["decision"])
print("Insufficient:", result["insufficient_requirements"])
print("Breakdown:", result["breakdown"])