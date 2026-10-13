"""
Parallel evaluation over sampled queries (fork, so workers share the parent's read-only arrays).

HEGREC_NPROC sets the number of worker processes (default 1: serial, identical to a plain loop).
The function may be a closure: it is published as a module global before the pool forks, so it is
never pickled. Each worker gets its own copy of any scratch buffer the closure mutates.
"""
import os
import multiprocessing as mp
import numpy as np

_F = None


def _run(chunk):
    return [_F(x) for x in chunk]


def pmap(f, items, nproc=None):
    """[f(x) for x in items], in order, over nproc forked workers."""
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
