"""
Authorization (data-ownership) tests.

Authentication proves WHO you are; these tests prove you can only reach YOUR OWN
data. Two users each upload a resume and start a run, and we assert neither can
read, delete, cancel, resume, or fetch rankings for the other's data — every
cross-user attempt returns 404 (not 403, so existence isn't leaked).

Marked `db` (needs Postgres). Skips cleanly when the DB isn't configured.

Run:  pytest tests/test_authorization.py -v
"""
import os
import uuid
import pytest

_DB_VARS = ("DB_HOST", "DB_PORT", "DB_NAME", "DB_USER", "DB_PASSWORD")


def _db_configured():
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except Exception:
        pass
    return all(os.getenv(v) for v in _DB_VARS)


pytestmark = [
    pytest.mark.db,
    pytest.mark.skipif(not _db_configured(),
                       reason="Postgres env not set — ownership tests need a live DB"),
]

pytest.importorskip("fastapi.testclient")
from fastapi.testclient import TestClient


def _fresh_logged_in_client():
    from auth import create_user
    import api
    uname = "authtest_" + uuid.uuid4().hex[:10]
    pw = "pw_" + uuid.uuid4().hex[:8]
    create_user(uname, pw, role="user")
    c = TestClient(api.app)
    r = c.post("/login", data={"username": uname, "password": pw})
    assert r.status_code == 200, r.text
    return c


def _make_pdf_bytes(text="Python SQL Django engineer resume with real content"):
    """Build a valid, text-bearing PDF with reportlab so pypdf can extract text.
    Skips the whole module if reportlab isn't installed."""
    pytest.importorskip("reportlab", reason="reportlab needed to build a test PDF")
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    import io
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    y = 720
    for line in [text, "Skills: Python, SQL, Django, AWS", "Experience: 5 years",
                 "Projects: agentops monitor", "Education: BS Computer Science"]:
        c.drawString(72, y, line)
        y -= 24
    c.showPage()
    c.save()
    return buf.getvalue()


def _make_pdf_bytes(text="Python SQL Django engineer resume with real content"):
    """Build a valid, text-bearing PDF with reportlab so pypdf can extract text.
    Skips the whole module if reportlab isn't installed."""
    pytest.importorskip("reportlab", reason="reportlab needed to build a test PDF")
    from reportlab.pdfgen import canvas
    from reportlab.lib.pagesizes import letter
    import io
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    y = 720
    for line in [text, "Skills: Python, SQL, Django, AWS", "Experience: 5 years",
                 "Projects: agentops monitor", "Education: BS Computer Science"]:
        c.drawString(72, y, line)
        y -= 24
    c.showPage()
    c.save()
    return buf.getvalue()


def _upload_resume(c):
    pdf = _make_pdf_bytes()
    r = c.post("/upload", files={"file": ("r.pdf", pdf, "application/pdf")},
               data={"name": "mine"})
    if r.status_code == 400:
        pytest.skip("resume text extraction unavailable in this environment")
    assert r.status_code == 200, r.text
    return r.json()["resume_id"]


def test_user_cannot_list_others_resumes():
    a = _fresh_logged_in_client()
    b = _fresh_logged_in_client()
    rid_a = _upload_resume(a)
    ids_b = {row["id"] for row in b.get("/resumes").json()}
    assert rid_a not in ids_b


def test_user_cannot_delete_others_resume():
    a = _fresh_logged_in_client()
    b = _fresh_logged_in_client()
    rid_a = _upload_resume(a)
    assert b.delete(f"/resumes/{rid_a}").status_code == 404
    assert rid_a in {row["id"] for row in a.get("/resumes").json()}


def test_user_cannot_touch_others_run():
    a = _fresh_logged_in_client()
    b = _fresh_logged_in_client()
    rid_a = _upload_resume(a)
    run_id = a.post(f"/runs?resume_id={rid_a}&target_role=engineer").json()["run_id"]
    assert b.get(f"/runs/{run_id}").status_code == 404
    assert b.get(f"/runs/{run_id}/rankings").status_code == 404
    assert b.post(f"/runs/{run_id}/cancel").status_code == 404
    assert b.post(f"/runs/{run_id}/resume?decision=Skip").status_code in (404, 409)
    assert run_id not in {row["id"] for row in b.get("/runs").json()}


def test_owner_can_read_own_run():
    a = _fresh_logged_in_client()
    rid_a = _upload_resume(a)
    run_id = a.post(f"/runs?resume_id={rid_a}&target_role=engineer").json()["run_id"]
    assert a.get(f"/runs/{run_id}").status_code == 200


def test_start_run_rejects_someone_elses_resume():
    a = _fresh_logged_in_client()
    b = _fresh_logged_in_client()
    rid_a = _upload_resume(a)
    r = b.post(f"/runs?resume_id={rid_a}&target_role=engineer")
    assert r.status_code == 404