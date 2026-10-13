"""Repository-relative path resolution for the HEGRec experiment scripts.

Every script imports its roots from here instead of hard-coding an absolute
path, so the tree can be cloned anywhere. Each root can be overridden with an
environment variable:

    HEGREC_HOME     repository root holding src/ and models/   (default: ..)
    HEGREC_DATA     dataset directory                           (default: $HEGREC_HOME/data)
    HEGREC_RESULTS  result JSON directory                       (default: $HEGREC_HOME/results)
"""
import os
from pathlib import Path

_HERE = Path(__file__).resolve().parent            # <repo>/experiments


def _root(env: str, default: Path) -> str:
    return str(Path(os.environ.get(env, default)).resolve())


HOMO_GCN = _root("HEGREC_HOME", _HERE.parent)
REPO = HOMO_GCN
DATA = _root("HEGREC_DATA", Path(HOMO_GCN) / "data")
RESULTS = _root("HEGREC_RESULTS", Path(HOMO_GCN) / "results")

__all__ = ["REPO", "HOMO_GCN", "DATA", "RESULTS"]
