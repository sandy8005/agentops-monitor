# test_auth.py
from auth import hash_password, verify_password

def test_hash_is_not_plaintext():
    h = hash_password("mysecret-1234")
    assert h != "mysecret-1234"        # never stored in the clear
    assert verify_password("mysecret-1234", h) is True
    assert verify_password("wrongpass", h) is False

def test_password_over_72_bytes_is_rejected_cleanly():
    # bcrypt 5.x raises on >72-byte input instead of truncating; we validate first and
    # give a clear error — measured in BYTES, so multi-byte characters count properly.
    import pytest
    with pytest.raises(ValueError, match="too long"):
        hash_password("a" * 73)
    with pytest.raises(ValueError, match="too long"):
        hash_password("é" * 37)          # 37 chars but 74 bytes in UTF-8
    assert verify_password("a" * 73, hash_password("a" * 72)) is False


def test_short_passwords_are_rejected():
    import pytest
    with pytest.raises(ValueError, match="at least 12"):
        hash_password("short-pass1")     # 11 characters