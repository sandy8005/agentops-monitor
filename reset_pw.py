import sys, os, psycopg2
from dotenv import load_dotenv
from auth import hash_password

load_dotenv()
username = sys.argv[1] if len(sys.argv) > 1 else "admin"
new_pw   = sys.argv[2] if len(sys.argv) > 2 else "changeme12"

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
    password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
)
cur = conn.cursor()
cur.execute("UPDATE users SET password_hash = %s WHERE username = %s",
            (hash_password(new_pw), username))
if cur.rowcount == 0:
    print(f"No user '{username}' found.")
else:
    conn.commit()
    print(f"Password for '{username}' reset to: {new_pw}")
conn.close()