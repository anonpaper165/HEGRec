"""Repository paths. Override with HEGREC_HOME (repo root), HEGREC_DATA (datasets) and HEGREC_RESULTS (output)."""
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent


def _root(env: str, default: Path) -> str:
    return str(Path(os.environ.get(env, default)).resolve())


REPO = _root("HEGREC_HOME", _HERE.parent)
DATA = _root("HEGREC_DATA", Path(REPO) / "data")
RESULTS = _root("HEGREC_RESULTS", Path(REPO) / "results")

__all__ = ["REPO", "DATA", "RESULTS"]
