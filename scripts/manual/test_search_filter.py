# test_search_filter.py
from job_source import search_jobs

# A role that should match few or none — verify we DON'T get the whole DB
jobs = search_jobs(target_role="quantum blockchain astrologer")
print(f"Returned {len(jobs)} jobs for a nonsense role")
# Should be a small sample (<= 3), NOT the full pool

# A real role — verify we get the genuine matches
jobs2 = search_jobs(target_role="engineer")
print(f"Returned {len(jobs2)} jobs for 'engineer'")