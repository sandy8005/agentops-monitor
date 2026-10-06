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