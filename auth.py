from timeutil import utcnow
from database import get_connection
import bcrypt

# bcrypt only ever looks at the first 72 BYTES of a password. bcrypt >= 5.0 no longer
# silently truncates longer input — hashpw() raises ValueError — so the limit is
# validated explicitly here (on the UTF-8 encoded length, since a 30-character
# password of multi-byte characters can already exceed 72 bytes).
BCRYPT_MAX_BYTES = 72
MIN_PASSWORD_CHARS = 12


def validate_password(plain):
    """Raise ValueError with a user-facing message if the password is unacceptable."""
    if not plain:
        raise ValueError("password is required")
    if len(plain) < MIN_PASSWORD_CHARS:
        raise ValueError(f"password must be at least {MIN_PASSWORD_CHARS} characters")
    if len(plain.encode("utf-8")) > BCRYPT_MAX_BYTES:
        raise ValueError(f"password is too long (max {BCRYPT_MAX_BYTES} bytes when UTF-8 encoded)")


def hash_password(plain):
    """One-way bcrypt hash (random per-hash salt). Validates length first, so a
    >72-byte password produces a clear ValueError instead of bcrypt's internal one."""
    validate_password(plain)
    hashed = bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt())
    return hashed.decode("utf-8")   # store as text


def verify_password(plain, hashed):
    """Check a login attempt against the stored hash. Returns True/False. A password
    that can't be a valid bcrypt input (e.g. >72 bytes) simply fails to verify."""
    try:
        pw = (plain or "").encode("utf-8")
        if len(pw) > BCRYPT_MAX_BYTES:
            return False
        return bcrypt.checkpw(pw, hashed.encode("utf-8"))
    except (ValueError, TypeError):
        return False


# A real bcrypt hash (same cost factor as stored hashes) of a random value that is
# never a password. authenticate() checks against it when the username does not
# exist, so an unknown user costs the same bcrypt work as a wrong password and the
# response time does not reveal which usernames exist. Built lazily (one hash).
_DUMMY_HASH = None


def _dummy_hash():
    global _DUMMY_HASH
    if _DUMMY_HASH is None:
        import secrets
        _DUMMY_HASH = bcrypt.hashpw(secrets.token_hex(16).encode("ascii"),
                                    bcrypt.gensalt()).decode("utf-8")
    return _DUMMY_HASH


class UsernameTaken(ValueError):
    """The username is already registered."""


def create_user(username, password, role="user"):
    """Create a user with a hashed password. Raises ValueError on invalid input and
    UsernameTaken (a ValueError) on a taken username. The connection is always
    returned to the pool.

    INSERT first and let the UNIQUE constraint decide: a "SELECT then INSERT" has
    a race in which two concurrent sign-ups both pass the SELECT and the loser gets
    a raw database error instead of this application error."""
    if not username:
        raise ValueError("username is required")
    password_hash = hash_password(password)   # validates length/bytes
    from psycopg2 import errors as pg_errors
    with get_connection() as conn:
        cur = conn.cursor()
        try:
            cur.execute(
                "INSERT INTO users (username, password_hash, role, created_at) "
                "VALUES (%s, %s, %s, %s) RETURNING id",
                (username, password_hash, role, utcnow()),
            )
        except pg_errors.UniqueViolation:
            raise UsernameTaken(f"username '{username}' is already taken") from None
        return cur.fetchone()[0]


def authenticate(username, password):
    """Verify credentials. Returns the user dict on success, None on failure.
    Does the same bcrypt work whether or not the username exists."""
    with get_connection() as conn:
        cur = conn.cursor()
        cur.execute("SELECT id, username, password_hash, role FROM users WHERE username = %s",
                    (username,))
        row = cur.fetchone()
    if not row:
        verify_password(password, _dummy_hash())     # equalize timing; result ignored
        return None
    if not verify_password(password, row[2]):
        return None
    return {"id": row[0], "username": row[1], "role": row[3]}