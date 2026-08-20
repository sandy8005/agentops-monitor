# test_employment_filter.py
from job_source import search_jobs
allj = search_jobs(target_role="engineer")
ft = search_jobs(target_role="engineer", employment_type="full-time")
intern = search_jobs(target_role="engineer", employment_type="internship")
print(f"engineer, any type: {len(allj)}")
print(f"engineer, full-time: {len(ft)}")
print(f"engineer, internship: {len(intern)}")
# full-time should be <= any; internship should show only internship-typed (or a sample if none)