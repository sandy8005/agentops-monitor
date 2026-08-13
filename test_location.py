from scorer import _location_score

cases = [
    # job_loc, job_mode, user_loc, user_mode, expected-ish
    ("Remote", "", "Michigan", "hybrid"),
    ("Detroit", "", "Michigan", "hybrid"),
    ("", "", "Michigan", "hybrid"),          # no data → 3.0
    ("Ann Arbor", "", "Michigan", "hybrid"),
    ("London, UK", "onsite", "Michigan", "hybrid"),
    ("Anywhere", "", "Michigan", "hybrid"),
]
for jl, jm, ul, um in cases:
    print(f"job={jl!r:15} mode={jm!r:8} -> {_location_score(jl, jm, ul, um)}")