# test_zero_match.py
from job_source import search_jobs
print("nonsense:", search_jobs(target_role="quantum blockchain astrologer"))
print("engineer count:", len(search_jobs(target_role="engineer")))