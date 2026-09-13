"""
Unit tests for the Adzuna fetch fixes — pure logic, no real network, no DB.

Covers:
  1. Unknown employment type  -> "" (not a guessed "full-time")
  2. Error classification      -> missing_keys / auth_error / rate_limited /
                                  http_error / network_error / empty / success

The upsert COALESCE-refresh semantics are SQL and require a live Postgres, so
they're exercised by integration runs, not here.

Run:  pytest test_adzuna.py -v
"""
import types
import pytest
import requests

import adzuna_jobs


# ---------- 1. Employment type: unknown when Adzuna gives no signal ----------

def test_employment_type_unknown_when_no_signal():
    assert adzuna_jobs._infer_employment_type(None, None) == ""
    assert adzuna_jobs._infer_employment_type("", "") == ""

def test_employment_type_from_contract_time():
    assert adzuna_jobs._infer_employment_type("full_time", None) == "full-time"
    assert adzuna_jobs._infer_employment_type("part_time", None) == "part-time"

def test_employment_type_from_contract_type():
    assert adzuna_jobs._infer_employment_type(None, "contract") == "contract"
    assert adzuna_jobs._infer_employment_type(None, "permanent") == "full-time"


# ---------- helpers to fake requests.get without touching the network ----------

class _FakeResp:
    def __init__(self, status_code=200, json_data=None, raise_json=False):
        self.status_code = status_code
        self._json = json_data if json_data is not None else {}
        self._raise_json = raise_json

    @property
    def ok(self):
        return 200 <= self.status_code < 300

    def json(self):
        if self._raise_json:
            raise ValueError("no json")
        return self._json


@pytest.fixture(autouse=True)
def _set_keys(monkeypatch):
    # Ensure keys look present for all tests except the missing-keys one.
    monkeypatch.setattr(adzuna_jobs, "ADZUNA_APP_ID", "id")
    monkeypatch.setattr(adzuna_jobs, "ADZUNA_APP_KEY", "key")


def _patch_get(monkeypatch, resp=None, exc=None):
    def fake_get(*a, **k):
        if exc is not None:
            raise exc
        return resp
    monkeypatch.setattr(adzuna_jobs.requests, "get", fake_get)


# ---------- 2. Error classification ----------

def test_missing_keys(monkeypatch):
    monkeypatch.setattr(adzuna_jobs, "ADZUNA_APP_ID", None)
    monkeypatch.setattr(adzuna_jobs, "ADZUNA_APP_KEY", None)
    jobs, status, err = adzuna_jobs.fetch_adzuna_jobs("ai", "texas")
    assert jobs == [] and status == "missing_keys"

def test_auth_error_401(monkeypatch):
    _patch_get(monkeypatch, _FakeResp(status_code=401))
    jobs, status, err = adzuna_jobs.fetch_adzuna_jobs("ai")
    assert jobs == [] and status == "auth_error"

def test_auth_error_403(monkeypatch):
    _patch_get(monkeypatch, _FakeResp(status_code=403))
    jobs, status, err = adzuna_jobs.fetch_adzuna_jobs("ai")
    assert status == "auth_error"

def test_rate_limited_429(monkeypatch):
    _patch_get(monkeypatch, _FakeResp(status_code=429))
    jobs, status, err = adzuna_jobs.fetch_adzuna_jobs("ai")
    assert jobs == [] and status == "rate_limited"

def test_http_error_500(monkeypatch):
    _patch_get(monkeypatch, _FakeResp(status_code=500))
    jobs, status, err = adzuna_jobs.fetch_adzuna_jobs("ai")
    assert jobs == [] and status == "http_error"

def test_network_error(monkeypatch):
    _patch_get(monkeypatch, exc=requests.exceptions.ConnectionError("boom"))
    jobs, status, err = adzuna_jobs.fetch_adzuna_jobs("ai")
    assert jobs == [] and status == "network_error"

def test_bad_json_is_http_error(monkeypatch):
    _patch_get(monkeypatch, _FakeResp(status_code=200, raise_json=True))
    jobs, status, err = adzuna_jobs.fetch_adzuna_jobs("ai")
    assert jobs == [] and status == "http_error"

def test_empty_when_ok_but_no_results(monkeypatch):
    _patch_get(monkeypatch, _FakeResp(status_code=200, json_data={"results": []}))
    jobs, status, err = adzuna_jobs.fetch_adzuna_jobs("ai")
    assert jobs == [] and status == "empty" and err is None

def test_success_returns_jobs(monkeypatch):
    payload = {"results": [{
        "id": "123", "title": "AI Engineer",
        "description": "build ai systems and models",
        "company": {"display_name": "Acme"},
        "location": {"display_name": "Austin, TX"},
        "contract_time": "full_time",
        "created": "2026-01-01T00:00:00Z",
        "redirect_url": "https://adzuna.example/123",
    }]}
    _patch_get(monkeypatch, _FakeResp(status_code=200, json_data=payload))
    jobs, status, err = adzuna_jobs.fetch_adzuna_jobs("ai")
    assert status == "success" and len(jobs) == 1
    j = jobs[0]
    assert j["external_id"] == "adzuna:123"
    assert j["employment_type"] == "full-time"
    assert j["apply_url"] == "https://adzuna.example/123"

def test_success_but_unknown_employment_type(monkeypatch):
    # A job with NO contract signal keeps employment_type "" (unknown), not full-time.
    payload = {"results": [{
        "id": "9", "title": "ML Engineer",
        "description": "machine learning pipelines and models",
        "company": {"display_name": "Beta"},
        "location": {"display_name": "Remote"},
    }]}
    _patch_get(monkeypatch, _FakeResp(status_code=200, json_data=payload))
    jobs, status, err = adzuna_jobs.fetch_adzuna_jobs("ml")
    assert status == "success"
    assert jobs[0]["employment_type"] == ""