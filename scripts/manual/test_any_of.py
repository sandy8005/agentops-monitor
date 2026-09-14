from scorer import calculate_match_score

parsed = {"skills": ["django"], "years_experience": 4,
          "projects": [{"name": "web", "tech": ["django"]}], "education": []}

# Job needs Python AND (Flask OR Django). Candidate has Django only.
reqs = {
    "required_skills": ["python"],
    "required_any_of": [["flask", "django"]],
    "preferred_skills": [],
    "min_years_experience": 2,
    "responsibilities": []
}
resume = "Senior Django developer with 4 years building web apps in Python and Django."

result = calculate_match_score(parsed, reqs, resume)
print("Score:", result["score"])
print("Breakdown:", result["breakdown"])
# required = 2 items (python, flask-or-django group). Both satisfied
# (python present, django satisfies the group) => required 50/50.