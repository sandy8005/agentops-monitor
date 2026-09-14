"""
Pytest configuration root.

Tests live in tests/ but import application modules by bare name
(`from scorer import ...`, `import autonomous_graph`). Since the app isn't an
installed package, we prepend the repo root to sys.path here so those imports
resolve no matter what directory pytest is invoked from. conftest.py at the repo
root is auto-loaded by pytest before test collection, which is exactly the hook
we need for this.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)