# test_auth.py
from auth import hash_password, verify_password

def test_hash_is_not_plaintext():
    h = hash_password("mysecret123")
    assert h != "mysecret123"        # never stored in the clear
    assert verify_password("mysecret123", h) is True
    assert verify_password("wrongpass", h) is False