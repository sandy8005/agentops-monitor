"""
Free-text input length limits — guard memory, LLM cost, latency, and DB size.
Oversized username/target_role/location/comment are rejected with 400. Needs Postgres.
"""
import uuid

import pytest

pytestmark = [pytest.mark.db]

from fastapi.testclient import TestClient
from auth import create_user
import api


def _client():
    c = TestClient(api.app)
    tok = c.get("/csrf").json()["csrf_token"]
    c.headers.update({"X-CSRF-Token": tok})
    return c


def test_login_rejects_oversized_username():
    c = _client()
    r = c.post("/login", data={"username": "u" * 300, "password": "x"})
    assert r.status_code == 400 and "too long" in r.json()["detail"]


def test_start_run_rejects_oversized_target_role_and_location():
    c = _client()
    u = "lim_" + uuid.uuid4().hex[:8]
    create_user(u, "password123")
    assert c.post("/login", data={"username": u, "password": "password123"}).status_code == 200

    r = c.post("/runs?resume_id=1&target_role=" + "z" * 300)
    assert r.status_code == 400 and "target_role is too long" in r.json()["detail"]

    r = c.post("/runs?resume_id=1&target_role=engineer&location=" + "L" * 300)
    assert r.status_code == 400 and "location is too long" in r.json()["detail"]


def test_resume_rejects_oversized_comment():
    c = _client()
    u = "lim2_" + uuid.uuid4().hex[:8]
    create_user(u, "password123")
    assert c.post("/login", data={"username": u, "password": "password123"}).status_code == 200
    r = c.post("/runs/1/resume?decision=Skip&comment=" + "c" * 3000)
    assert r.status_code == 400 and "comment is too long" in r.json()["detail"]