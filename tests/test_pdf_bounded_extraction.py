"""
PDF extraction output is bounded AT THE SOURCE: the child stops extracting past
max_chars, reports truncation explicitly, and the parent never buffers more than
the byte bound max_chars implies.
"""
import io
import json
import os

import pytest

import pdf_reader

reportlab = pytest.importorskip("reportlab")


def _pdf(tmp_path, pages, line="Python engineer building data pipelines. " * 2,
         lines_per_page=40):
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=letter)
    for _ in range(pages):
        y = 750
        for _ in range(lines_per_page):
            c.drawString(20, y, line)
            y -= 18
        c.showPage()
    c.save()
    path = tmp_path / "resume.pdf"
    path.write_bytes(buf.getvalue())
    return str(path)


def test_small_pdf_is_not_truncated(tmp_path):
    out = pdf_reader.read_resume_file_isolated(_pdf(tmp_path, 1), max_chars=60_000)
    assert out["truncated"] is False
    assert "Python engineer" in out["text"]
    assert out["chars_seen"] >= len(out["text"])


def test_child_stops_at_max_chars_and_reports_truncation(tmp_path):
    path = _pdf(tmp_path, 10)
    full = pdf_reader.extract_text_bounded(path, max_chars=10**9)
    assert len(full["text"]) > 5_000                      # the document really is big

    out = pdf_reader.read_resume_file_isolated(path, max_chars=1_000)
    assert out["truncated"] is True
    assert len(out["text"]) == 1_000
    assert out["text"] == full["text"][:1_000]
    assert 1_000 < out["chars_seen"] < len(full["text"])  # stopped early, not read to end


def test_in_process_extraction_stops_after_the_page_that_crosses_the_bound(tmp_path,
                                                                           monkeypatch):
    """The extractor must not keep pulling pages once the bound is crossed."""
    path = _pdf(tmp_path, 10)
    calls = []
    import pypdf
    real = pypdf.PageObject.extract_text

    def counting(self, *a, **k):
        calls.append(1)
        return real(self, *a, **k)
    monkeypatch.setattr(pypdf.PageObject, "extract_text", counting)
    out = pdf_reader.extract_text_bounded(path, max_chars=500)
    assert out["truncated"] is True and len(calls) == 1


def test_parent_rejects_a_result_larger_than_the_byte_bound(tmp_path, monkeypatch):
    """Even a misbehaving child cannot make the parent read an unbounded result."""
    path = _pdf(tmp_path, 1)
    real_run = pdf_reader.subprocess.run

    def child_writes_too_much(cmd, **kw):
        out_path = cmd[cmd.index("--out") + 1]
        with open(out_path, "w") as f:
            f.write(json.dumps({"text": "x" * 50_000, "truncated": False,
                                "chars_seen": 50_000}))
        return real_run(["true"] if os.name == "posix" else ["cmd", "/c", "exit 0"])
    monkeypatch.setattr(pdf_reader.subprocess, "run", child_writes_too_much)
    with pytest.raises(pdf_reader.PdfExtractionError):
        pdf_reader.read_resume_file_isolated(path, max_chars=100)


def test_parent_does_not_capture_stdout(tmp_path, monkeypatch):
    seen = {}
    real_run = pdf_reader.subprocess.run

    def spy(cmd, **kw):
        seen.update(kw)
        return real_run(cmd, **kw)
    monkeypatch.setattr(pdf_reader.subprocess, "run", spy)
    pdf_reader.read_resume_file_isolated(_pdf(tmp_path, 1), max_chars=60_000)
    assert "capture_output" not in seen
    assert seen["stdout"] is pdf_reader.subprocess.DEVNULL


def test_temp_result_file_is_removed(tmp_path, monkeypatch):
    made = []
    real = pdf_reader.tempfile.mkstemp

    def track(*a, **k):
        fd, p = real(*a, **k)
        made.append(p)
        return fd, p
    monkeypatch.setattr(pdf_reader.tempfile, "mkstemp", track)
    pdf_reader.read_resume_file_isolated(_pdf(tmp_path, 1))
    assert made and not os.path.exists(made[0])

# --- Backlog P2: page-limit truncation is reported, not silent ---------------

def test_page_limit_truncation_is_reported(tmp_path):
    path = _pdf(tmp_path, 4, lines_per_page=2)
    out = pdf_reader.extract_text_bounded(path, max_chars=10**6, max_pages=2)
    assert out["truncated"] is True
    assert out["truncated_reason"] == "page_limit"
    assert (out["pages_read"], out["pages_total"]) == (2, 4)


def test_exactly_max_pages_is_not_truncated(tmp_path):
    out = pdf_reader.extract_text_bounded(_pdf(tmp_path, 2, lines_per_page=2),
                                          max_chars=10**6, max_pages=2)
    assert out["truncated"] is False and out["truncated_reason"] is None
    assert out["pages_read"] == out["pages_total"] == 2


def test_char_limit_reason(tmp_path):
    out = pdf_reader.extract_text_bounded(_pdf(tmp_path, 3), max_chars=500)
    assert out["truncated"] is True and out["truncated_reason"] == "char_limit"


def test_child_reports_page_limit_through_the_process_boundary(tmp_path):
    path = _pdf(tmp_path, pdf_reader.PDF_MAX_PAGES + 1, lines_per_page=1)
    out = pdf_reader.read_resume_file_isolated(path, max_chars=60_000)
    assert out["truncated"] is True and out["truncated_reason"] == "page_limit"
    assert out["pages_read"] == pdf_reader.PDF_MAX_PAGES
    assert out["pages_total"] == pdf_reader.PDF_MAX_PAGES + 1


def test_parent_rejects_inconsistent_truncation_reason(tmp_path, monkeypatch):
    path = _pdf(tmp_path, 1)
    real_run = pdf_reader.subprocess.run

    def child_lies(cmd, **kw):
        out_path = cmd[cmd.index("--out") + 1]
        with open(out_path, "w") as f:
            f.write(json.dumps({"text": "x", "truncated": False,
                                "truncated_reason": "page_limit", "chars_seen": 1,
                                "pages_total": 9, "pages_read": 1}))
        return real_run(["true"] if os.name == "posix" else ["cmd", "/c", "exit 0"])
    monkeypatch.setattr(pdf_reader.subprocess, "run", child_lies)
    with pytest.raises(pdf_reader.PdfExtractionError):
        pdf_reader.read_resume_file_isolated(path, max_chars=100)


@pytest.mark.db
def test_upload_api_surfaces_page_limit_truncation(tmp_path, monkeypatch):
    import uuid
    from fastapi.testclient import TestClient
    import api
    from auth import create_user
    monkeypatch.setattr(api, "read_resume_file_isolated", lambda *a, **k: {
        "text": "Synthetic Python engineer resume text.", "truncated": True,
        "truncated_reason": "page_limit", "chars_seen": 38,
        "pages_total": 80, "pages_read": 50})
    uname, pw = "pdftest_" + uuid.uuid4().hex[:10], "pw_" + uuid.uuid4().hex[:12]
    create_user(uname, pw, role="user")
    c = TestClient(api.app)
    c.headers.update({"X-CSRF-Token": c.get("/csrf").json()["csrf_token"]})
    assert c.post("/login", data={"username": uname, "password": pw}).status_code == 200
    pdf = open(_pdf(tmp_path, 1), "rb").read()
    r = c.post("/upload", files={"file": ("r.pdf", pdf, "application/pdf")},
               data={"name": "pages"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["truncated"] is True and body["truncated_reason"] == "page_limit"
    assert (body["pages_read"], body["pages_total"]) == (50, 80)
    assert "80 pages" in body["message"] and "first 50" in body["message"]
