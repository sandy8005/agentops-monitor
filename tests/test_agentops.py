def test_required_any_of_satisfied_by_one_member():
    # "flask OR django" satisfied by django alone -> FULL required credit.
    # With no preferred skills in these reqs, the preferred bucket is absent and its
    # weight is renormalized across required/projects/experience, so a fully-satisfied
    # required section reads 62.5 (50 of the 85 applicable points, scaled to 100),
    # not the raw 50. Full credit == the required bucket's full normalized weight.
    r = calculate_match_score(
        _resume(["python", "django"]),
        _reqs(required=["python"], any_of=[["flask", "django"]]),
        "python django", None, None
    )
    assert r["breakdown"]["required"] == 62.5

def test_project_score_uses_same_or_semantics():
    # django-only candidate should get FULL project credit for a flask-OR-django group
    r = calculate_match_score(
        _resume(["python", "django"],
                projects=[{"name": "web app", "tech": ["python", "django"]}]),
        _reqs(required=["python"], any_of=[["flask", "django"]]),
        "python django", None, None
    )
    assert r["breakdown"]["projects"] == 18.8   # 15 renormalized (preferred absent)