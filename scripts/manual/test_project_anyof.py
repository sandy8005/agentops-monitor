# test_project_anyof.py
from scorer import calculate_match_score

# Candidate has Django (not Flask); requirement is "Flask OR Django"
parsed = {
    "skills": ["python", "django"],
    "years_experience": 3,
    "projects": [{"name": "web app", "tech": ["python", "django"]}],
    "education": []
}
reqs = {
    "required_skills": ["python"],
    "required_any_of": [["flask", "django"]],
    "preferred_skills": [],
    "min_years_experience": 2,
    "responsibilities": []
}
result = calculate_match_score(parsed, reqs, "python django developer", None, None)
print("breakdown:", result["breakdown"])
# required: python(1) + flask-or-django group satisfied by django(1) = 2/2 -> 50.0
# projects: python in projects(1) + flask-or-django group satisfied by django in projects(1) = 2/2 -> 15.0
# So projects should be FULL 15.0, not half — proving the group semantics match.
print("projects should be 15.0:", result["breakdown"]["projects"])