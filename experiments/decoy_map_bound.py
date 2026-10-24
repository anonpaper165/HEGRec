"""Exact best-guess linkage under redrawn decoys (Table 3, 5 redrawn decoys on Amazon-Book).
Output: results/decoy_map_bound.json
"""
from _paths import DATA, RESULTS
import json
import numpy as np
from math import comb

SLOTS = 4096
KS = [0, 1, 2, 3, 5, 8]


def load(ds):
    d = {}
    for line in open(f"{DATA}/{ds}/train.txt"):
        p = line.split()
        if len(p) > 1:
            d[int(p[0])] = set(map(int, p[1:]))
    return {u: v for u, v in d.items() if v}


def subset_max(best, C):
    for b in range(C):
        step = 1 << b
        v = best.reshape(-1, 2 * step)
        np.maximum(v[:, step:], v[:, :step], out=v[:, step:])
    return best


def run(ds):
    tr = load(ds)
    users = sorted(tr)
    N = len(users)
    C = max(i for v in tr.values() for i in v) // SLOTS + 1
    masks = np.fromiter((sum(1 << c for c in {i // SLOTS for i in tr[u]}) for u in users),
                        dtype=np.int64, count=N)
    uniq = np.unique(masks)
    m_over_N = len(uniq) / N

    best = np.full(1 << C, -1, dtype=np.int8)
    pc = np.array([bin(int(x)).count("1") for x in uniq], dtype=np.int8)
    np.maximum.at(best, uniq, pc)
    subset_max(best, C)

    ysz = np.zeros(1 << C, dtype=np.int8)
    for b in range(C):
        ysz[1 << b:] = ysz[1 << b:]
    ysz = np.unpackbits(np.arange(1 << C, dtype=">u4").view(np.uint8)
                        .reshape(-1, 4), axis=1).sum(1).astype(np.int8)

    out = {"users": N, "C": C, "m_over_N": m_over_N, "V": {}}
    valid = best >= 0
    bsel = best[valid].astype(np.int64)
    asel = ysz[valid].astype(np.int64) - bsel
    for k in KS:
        tbl = np.zeros((C + 1, k + 1), dtype=np.float64)
        for bp in range(C + 1):
            for a in range(k + 1):
                r = k - a
                tbl[bp, a] = comb(bp, r) if 0 <= r <= bp else 0.0
        ok = (asel >= 0) & (asel <= k)
        tot = tbl[bsel[ok], asel[ok]].sum() / comb(C, k)
        V = tot / N
        out["V"][str(k)] = V
        print(f"   k={k:<2} V={V:.6f}  baseline 대비 {V/m_over_N:.3f}x", flush=True)
    return out


if __name__ == "__main__":
    res = {"slots": SLOTS, "ks": KS, "datasets": {}}
    for ds in ["gowalla", "amazon-book"]:
        print(f"== {ds}", flush=True)
        res["datasets"][ds] = run(ds)
    with open(f"{RESULTS}/decoy_map_bound.json", "w") as f:
        json.dump(res, f, indent=1)
    print("wrote results/decoy_map_bound.json")
