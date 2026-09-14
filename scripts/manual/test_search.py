# test_search.py
from job_source import search_jobs

print("--- target: 'AI/ML Engineer' ---")
jobs = search_jobs("AI/ML Engineer")
for j in jobs:
    print(f"  {j['title']}")

print("\n--- target: 'Backend' ---")
jobs = search_jobs("Backend")
for j in jobs:
    print(f"  {j['title']}")

print("\n--- no target (all) ---")
jobs = search_jobs()
print(f"  {len(jobs)} jobs")