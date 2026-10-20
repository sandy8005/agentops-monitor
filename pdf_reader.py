"""
Resume PDF text extraction.

Parsing an arbitrary uploaded PDF is attacker-reachable work (decompression bombs,
pathological object graphs, parser bugs). It therefore does NOT run inside the API
process: read_resume_file_isolated() runs pypdf in a SHORT-LIVED CHILD PROCESS
with a wall-clock timeout and, on POSIX, CPU-time, address-space and file-size
limits. A PDF that hangs, explodes in memory or crashes the parser kills only that
child; the API just gets a PdfExtractionError.

OUTPUT IS BOUNDED AT THE SOURCE. A small, highly compressible PDF can expand into
a huge amount of text. The child is told the maximum number of characters the API
will keep (max_chars) and stops accumulating once it has more than that, so it
never emits an unbounded payload. The parent does not buffer the child's stdout in
memory at all: the child writes to a temporary FILE (capped on POSIX with
RLIMIT_FSIZE), and the parent refuses to read a result larger than the byte bound
that max_chars implies. Truncation is reported explicitly ("truncated": true).

Platform note: the CPU / memory / file-size limits are POSIX-only. On Windows the
wall-clock timeout and the child's own max_chars bound are the controls.

This is a containment step, not a sandbox: for a public service, also run the API
with least privilege (or move extraction to a dedicated worker/container), and keep
pypdf patched.
"""
import json
import os
import subprocess
import sys
import tempfile

PDF_TIMEOUT_SECONDS = 20
PDF_CPU_SECONDS = 15
PDF_MAX_MEMORY_BYTES = 512 * 1024 * 1024
PDF_MAX_PAGES = 50
# Default bound when a caller does not pass one (the API passes MAX_RESUME_CHARS).
PDF_MAX_TEXT_CHARS = 60_000
# json.dumps escapes non-ASCII as \uXXXX: at most 12 bytes per character (a
# surrogate pair), plus the envelope. The parent never reads more than this.
_BYTES_PER_CHAR_BOUND = 12
_ENVELOPE_BYTES = 1024


_TRUNCATION_REASONS = (None, "char_limit", "page_limit")


class PdfExtractionError(RuntimeError):
    """The PDF could not be parsed within the resource limits."""


def _result_byte_bound(max_chars):
    return int(max_chars) * _BYTES_PER_CHAR_BOUND + _ENVELOPE_BYTES


def extract_text_bounded(file_path, max_chars, max_pages=PDF_MAX_PAGES):
    """In-process extraction that STOPS once more than `max_chars` characters were
    extracted or `max_pages` pages were read. Returns {"text": <at most max_chars
    chars>, "truncated": bool, "truncated_reason": None | "char_limit" |
    "page_limit", "chars_seen": int, "pages_total": int, "pages_read": int}.
    chars_seen is exact when not truncated and a lower bound when truncated — the
    rest of the document is never extracted. Truncation is never silent: a PDF
    with pages beyond max_pages reports truncated=True, reason "page_limit"."""
    from pypdf import PdfReader
    max_chars = int(max_chars)
    reader = PdfReader(file_path)
    pages_total = len(reader.pages)
    parts, total, reason = [], 0, None
    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            reason = "page_limit"                     # pages remain unread
            break
        piece = page.extract_text() or ""
        sep = 1 if parts else 0                       # the "\n" join separator
        parts.append(piece)
        total += sep + len(piece)
        if total > max_chars:
            reason = "char_limit"
            break
    text = "\n".join(parts).strip()
    if len(text) > max_chars:
        text = text[:max_chars]
        reason = "char_limit"
    return {"text": text, "truncated": reason is not None, "truncated_reason": reason,
            "chars_seen": total, "pages_total": pages_total, "pages_read": len(parts)}


def _limit_resources(max_output_bytes):   # runs in the child, before exec (POSIX only)
    def apply():
        import resource
        resource.setrlimit(resource.RLIMIT_CPU, (PDF_CPU_SECONDS, PDF_CPU_SECONDS))
        resource.setrlimit(resource.RLIMIT_AS, (PDF_MAX_MEMORY_BYTES, PDF_MAX_MEMORY_BYTES))
        # The child's only file write is its result: it can never exceed the bound.
        resource.setrlimit(resource.RLIMIT_FSIZE, (max_output_bytes, max_output_bytes))
    return apply


def read_resume_file_isolated(file_path, max_chars=PDF_MAX_TEXT_CHARS,
                              timeout=PDF_TIMEOUT_SECONDS):
    """Extract text in a resource-limited child process.

    Returns {"text", "truncated", "truncated_reason", "chars_seen", "pages_total",
    "pages_read"} (see extract_text_bounded). Raises
    PdfExtractionError on timeout, crash, an oversized or malformed result."""
    max_chars = int(max_chars)
    bound = _result_byte_bound(max_chars)
    here = os.path.dirname(os.path.abspath(__file__))
    out_fd, out_path = tempfile.mkstemp(prefix="pdf_text_", suffix=".json")
    try:
        cmd = [sys.executable, "-I", os.path.join(here, "pdf_reader.py"),
               "--extract", file_path, "--max-chars", str(max_chars), "--out", out_path]
        kwargs = {"timeout": timeout, "cwd": here,
                  "stdin": subprocess.DEVNULL, "stdout": subprocess.DEVNULL,
                  "stderr": subprocess.DEVNULL,
                  "env": {"PATH": os.environ.get("PATH", ""),
                          "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}}
        if os.name == "posix":
            kwargs["preexec_fn"] = _limit_resources(bound)
        try:
            proc = subprocess.run(cmd, **kwargs)
        except subprocess.TimeoutExpired:
            raise PdfExtractionError("PDF parsing timed out")
        if proc.returncode != 0:
            raise PdfExtractionError("PDF could not be parsed")
        size = os.fstat(out_fd).st_size
        if size > bound:
            raise PdfExtractionError("PDF parser returned oversized output")
        with os.fdopen(out_fd, "rb") as f:
            out_fd = None
            raw = f.read(bound + 1)
        try:
            data = json.loads(raw.decode("utf-8"))
            text = data["text"]
            truncated = bool(data["truncated"])
            reason = data["truncated_reason"]
            chars_seen = int(data["chars_seen"])
            pages_total = int(data["pages_total"])
            pages_read = int(data["pages_read"])
        except (ValueError, KeyError, TypeError, UnicodeDecodeError):
            raise PdfExtractionError("PDF parser returned malformed output")
        if (not isinstance(text, str) or len(text) > max_chars
                or reason not in _TRUNCATION_REASONS or truncated != (reason is not None)):
            raise PdfExtractionError("PDF parser returned malformed output")
        return {"text": text, "truncated": truncated, "truncated_reason": reason,
                "chars_seen": chars_seen, "pages_total": pages_total,
                "pages_read": pages_read}
    finally:
        if out_fd is not None:
            os.close(out_fd)
        try:
            os.remove(out_path)
        except OSError:
            pass


def _main(argv):
    args = dict(zip(argv[1::2], argv[2::2]))
    if len(argv) != 7 or set(args) != {"--extract", "--max-chars", "--out"}:
        print("usage: pdf_reader.py --extract <file.pdf> --max-chars N --out <result.json>",
              file=sys.stderr)
        return 2
    result = extract_text_bounded(args["--extract"], int(args["--max-chars"]))
    with open(args["--out"], "w", encoding="utf-8") as f:
        f.write(json.dumps(result))
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv))