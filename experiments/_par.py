"""Parallel map over sampled queries (fork). HEGREC_NPROC sets the number of workers (default 1)."""
import os
import multiprocessing as mp
import numpy as np

_F = None


def _run(chunk):
    return [_F(x) for x in chunk]


def pmap(f, items, nproc=None):
    global _F
    items = list(items)
    nproc = int(nproc or os.environ.get("HEGREC_NPROC", 1))
    if nproc <= 1 or len(items) < 2:
        return [f(x) for x in items]
    _F = f
    chunks = [c for c in np.array_split(np.arange(len(items)), nproc * 4) if len(c)]
    with mp.get_context("fork").Pool(nproc) as pool:
        parts = pool.map(_run, [[items[i] for i in c] for c in chunks])
    return [y for part in parts for y in part]


def pmean(f, items, nproc=None):
    return float(np.mean(pmap(f, items, nproc)))
