"""
Resume PDF text extraction.

Parsing an arbitrary uploaded PDF is attacker-reachable work (decompression bombs,
pathological object graphs, parser bugs). It therefore does NOT run inside the API
process any more: read_resume_file_isolated() runs pypdf in a SHORT-LIVED CHILD
PROCESS with a wall-clock timeout and, on POSIX, CPU-time and address-space limits.
A PDF that hangs, explodes in memory or crashes the parser kills only that child;
the API just gets a PdfExtractionError.

This is a containment step, not a sandbox: for a public service, also run the API
with least privilege (or move extraction to a dedicated worker/container), and keep
pypdf patched.
"""
import json
import os
import subprocess
import sys

PDF_TIMEOUT_SECONDS = 20
PDF_CPU_SECONDS = 15
PDF_MAX_MEMORY_BYTES = 512 * 1024 * 1024
PDF_MAX_PAGES = 50


class PdfExtractionError(RuntimeError):
    """The PDF could not be parsed within the resource limits."""


def read_resume_file(file_path, max_pages=PDF_MAX_PAGES):
    """In-process extraction (used by the isolated child and by trusted scripts)."""
    from pypdf import PdfReader
    reader = PdfReader(file_path)
    parts = []
    for i, page in enumerate(reader.pages):
        if i >= max_pages:
            break
        parts.append(page.extract_text() or "")
    return "\n".join(parts).strip()


def _limit_resources():   # runs in the child, before exec (POSIX only)
    import resource
    resource.setrlimit(resource.RLIMIT_CPU, (PDF_CPU_SECONDS, PDF_CPU_SECONDS))
    resource.setrlimit(resource.RLIMIT_AS, (PDF_MAX_MEMORY_BYTES, PDF_MAX_MEMORY_BYTES))


def read_resume_file_isolated(file_path, timeout=PDF_TIMEOUT_SECONDS):
    """Extract text in a resource-limited child process. Raises PdfExtractionError."""
    here = os.path.dirname(os.path.abspath(__file__))
    cmd = [sys.executable, "-I", os.path.join(here, "pdf_reader.py"), "--extract", file_path]
    kwargs = {"capture_output": True, "timeout": timeout, "cwd": here,
              "env": {"PATH": os.environ.get("PATH", ""),
                      "PYTHONPATH": os.pathsep.join(p for p in sys.path if p)}}
    if os.name == "posix":
        kwargs["preexec_fn"] = _limit_resources
    try:
        proc = subprocess.run(cmd, **kwargs)
    except subprocess.TimeoutExpired:
        raise PdfExtractionError("PDF parsing timed out")
    if proc.returncode != 0:
        raise PdfExtractionError("PDF could not be parsed")
    try:
        return json.loads(proc.stdout.decode("utf-8"))["text"]
    except (ValueError, KeyError, UnicodeDecodeError):
        raise PdfExtractionError("PDF parser returned malformed output")


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "--extract":
        text = read_resume_file(sys.argv[2])
        sys.stdout.write(json.dumps({"text": text}))
        sys.exit(0)
    print("usage: pdf_reader.py --extract <file.pdf>", file=sys.stderr)
    sys.exit(2)