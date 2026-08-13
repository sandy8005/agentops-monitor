# test_search_filter.py
from job_source import search_jobs

print("--- role only ---")
jobs = search_jobs("AI/ML Engineer", "Michigan")
print(f"  {len(jobs)} jobs")

print("--- role + work_mode=remote ---")
jobs = search_jobs("AI/ML Engineer", "Michigan", "remote")
print(f"  {len(jobs)} jobs")
for j in jobs[:5]:
    print(f"    {j['title']}  [{j.get('work_mode') or j.get('location') or 'no-mode'}]")