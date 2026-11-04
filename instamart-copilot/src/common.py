"""Shared helpers: repository paths, import path for mock_llm.py, and a report tee."""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

RAW = os.path.join(ROOT, "data", "raw")
POLICIES = os.path.join(ROOT, "data", "policies")
CLEAN = os.path.join(ROOT, "data", "clean")
DB_PATH = os.path.join(ROOT, "data", "instamart.db")
OUT = os.path.join(ROOT, "outputs")
REPORTS = os.path.join(OUT, "reports")
STATE = os.path.join(ROOT, "state")
SQL_FILE = os.path.join(ROOT, "sql", "queries.sql")

for _d in (CLEAN, OUT, REPORTS, STATE, os.path.dirname(SQL_FILE)):
    os.makedirs(_d, exist_ok=True)


class _Tee:
    def __init__(self, stream, fh):
        self.stream, self.fh = stream, fh

    def write(self, s):
        self.stream.write(s)
        self.fh.write(s)

    def flush(self):
        self.stream.flush()
        self.fh.flush()


def start_report(name):
    """Mirror everything printed from now on into outputs/reports/<name>.txt."""
    fh = open(os.path.join(REPORTS, name + ".txt"), "w", encoding="utf-8")
    sys.stdout = _Tee(sys.__stdout__, fh)
    return fh


def check(label, got, expected, tol=None):
    """Print a reference-checkpoint comparison and return True/False."""
    if tol is None:
        ok = got == expected
    else:
        ok = abs(got - expected) <= tol
    print(f"  [{'OK ' if ok else 'DIFF'}] {label}: got {got!r}  reference {expected!r}")
    return ok
