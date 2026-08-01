import json, psycopg2, os
from dotenv import load_dotenv
from schemas import ParsedResume, JobRequirements
from pydantic import ValidationError
load_dotenv()

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()

# validate a stored parsed resume
cur.execute("""
    SELECT l.response FROM llm_calls l JOIN steps s ON l.step_id = s.id
    WHERE s.run_id = 23 AND s.step_name = 'parse_resume'
""")
raw = cur.fetchone()[0]
try:
    validated = ParsedResume(**json.loads(raw))
    print(f"✓ Resume valid: {len(validated.skills)} skills, {validated.years_experience} years")
except ValidationError as e:
    print(f"✗ Resume validation failed:\n{e}")

conn.close()

# now test that BAD data is correctly REJECTED
print("\nTesting rejection of bad data:")
bad = {"skills": ["Python"], "years_experience": "about two",
       "education": [], "projects": [], "experience": []}
try:
    ParsedResume(**bad)
    print("✗ Bad data was accepted — validation NOT working")
except ValidationError:
    print("✓ Bad data correctly rejected (years_experience must be a number)")