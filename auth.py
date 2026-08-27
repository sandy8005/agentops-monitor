import psycopg2, os
from datetime import datetime
from dotenv import load_dotenv
import bcrypt

load_dotenv()


def get_connection():
    return psycopg2.connect(
        dbname=os.getenv("DB_NAME"), user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"), host=os.getenv("DB_HOST"), port=os.getenv("DB_PORT")
    )


def hash_password(plain):
    """
    One-way bcrypt hash. We store this, never the plaintext.
    bcrypt only uses the first 72 BYTES of a password (a property of the
    algorithm), so we encode to bytes and the library handles salting.
    """
    pw_bytes = plain.encode("utf-8")
    # bcrypt.gensalt() produces a random salt each time, so identical passwords
    # get different hashes — this defeats rainbow-table attacks.
    hashed = bcrypt.hashpw(pw_bytes, bcrypt.gensalt())
    return hashed.decode("utf-8")   # store as text


def verify_password(plain, hashed):
    """Check a login attempt against the stored hash. Returns True/False."""
    try:
        return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


def create_user(username, password, role="user"):
    """Create a user with a hashed password. Raises if username already exists."""
    if not username or not password:
        raise ValueError("username and password are required")
    if len(password) < 8:
        raise ValueError("password must be at least 8 characters")
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT 1 FROM users WHERE username = %s", (username,))
    if cur.fetchone():
        conn.close()
        raise ValueError(f"username '{username}' is already taken")
    cur.execute(
        "INSERT INTO users (username, password_hash, role, created_at) VALUES (%s, %s, %s, %s) RETURNING id",
        (username, hash_password(password), role, datetime.now())
    )
    user_id = cur.fetchone()[0]
    conn.commit()
    conn.close()
    return user_id


def authenticate(username, password):
    """Verify credentials. Returns the user dict on success, None on failure."""
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT id, username, password_hash, role FROM users WHERE username = %s", (username,))
    row = cur.fetchone()
    conn.close()
    if not row:
        return None
    if not verify_password(password, row[2]):
        return None
    return {"id": row[0], "username": row[1], "role": row[3]}