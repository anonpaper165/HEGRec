"""Unequal request rates and rate-aware grouping in the initial codebook (§5.5: 0.206, 0.098, 1.21x).
Output: results/dormant_prior_robust.json
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
from _paths import RESULTS
import sys, json, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_amazon2023_hk as H
from location_padding2 import to_bits, Codebook
SLOTS = 4096
KEYS_MIB, SCORES_MIB, CHUNK_KIB = 8.0, 23.4, 158.7


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


OUT = os.path.join(RESULTS, "dormant_prior_robust.json")
DAY = 86400 * 1000


class RateCodebook(Codebook):
    def __init__(self, B, k, pi):
        self.C = B.shape[1]
        self.order = np.argsort(B.sum(0), kind="stable")
        keys = self.keys(B)
        idx = np.argsort(keys, kind="stable"); sk = keys[idx]; p = pi[idx]
        n = len(idx); starts = [0]; i = 0
        while i < n:
            j = i; s = 0.0; m = 0.0
            while j < n and (j - i < k or s < k * m or (j < n and j > i and sk[j] == sk[j - 1])):
                s += p[j]; m = max(m, p[j]); j += 1
            if j < n: starts.append(j)
            i = j
        if len(starts) > 1:
            last = starts[-1]; tail = p[last:]
            if len(tail) < k or tail.sum() < k * tail.max(): starts.pop()
        self.bounds = np.array(starts + [n])
        self.first_key = keys[idx[self.bounds[:-1]]]
        self.covers = np.zeros((len(starts), self.C), dtype=bool)
        self.group_of = np.empty(n, dtype=np.int64)
        for g in range(len(starts)):
            mem = idx[self.bounds[g]:self.bounds[g + 1]]
            self.covers[g] = B[mem].any(0); self.group_of[mem] = g


def classes(cb, B):
    sent = cb.send(B)
    return np.where(sent >= 0, sent, len(cb.covers)), sent


def traffic(cb, sent, B, base_chunks):
    ch = np.where(sent >= 0, cb.covers[np.maximum(sent, 0)].sum(1), B.shape[1])
    mib = lambda c: KEYS_MIB + SCORES_MIB + c * CHUNK_KIB / 1024
    return round(float(mib(ch.mean()) / mib(base_chunks)), 3)


def prior_success(cls, pi):
    n = int(cls.max()) + 1
    mx = np.zeros(n); np.maximum.at(mx, cls, pi)
    return round(float(mx.sum()), 4)


def main():
    U, I, T, nU, nI = H.load_hk()
    o = np.lexsort((I, U)); U, I, T = U[o], I[o], T[o]
    C = math.ceil(nI / SLOTS)
    off = np.searchsorted(U, np.arange(nU + 1))
    B = to_bits([np.unique(I[off[u]:off[u + 1]] // SLOTS) for u in range(nU)], C)
    deg = np.bincount(U, minlength=nU).astype(float)
    last = np.full(nU, -1, dtype=np.int64); np.maximum.at(last, U, T)
    tmax = int(T.max())
    base_chunks = B.sum(1).mean()
    rng = np.random.default_rng(0)
    out = {"note": __doc__.strip().split("\n\n")[0], "users": int(nU), "C": C, "dormant": {}, "prior_robust": {}}

    for k in (5, 10, 20, 50):
        cb = Codebook(B, k); cls, sent = classes(cb, B)
        row = {}
        for days in (30, 120, 365):
            act = last > tmax - days * DAY
            for name, a in (("time", act), ("random", rng.random(nU) < act.mean())):
                na = np.bincount(cls, weights=a.astype(float), minlength=int(cls.max()) + 1)[cls]
                au = a & (na > 0)
                row[f"{days}d_{name}"] = {
                    "active_share": round(float(a.mean()), 4),
                    "linkage_active": round(float(np.mean(1.0 / na[au])), 4),
                    "share_active_lt_half_k": round(float(np.mean(na[au] < k / 2)), 4),
                    "share_active_alone": round(float(np.mean(na[au] == 1)), 4)}
        row["all_members"] = round(float(np.mean(1.0 / np.bincount(cls)[cls])), 4)
        out["dormant"][str(k)] = row
        log("dormant k", k, row)
        json.dump(out, open(OUT, "w"), indent=2)

    pi = deg / deg.sum()
    for k in (5, 10, 20):
        t0 = time.time()
        base = Codebook(B, k); bc, bs = classes(base, B)
        rob = RateCodebook(B, k, pi); rc, rs = classes(rob, B)
        size = np.bincount(rob.group_of)
        out["prior_robust"][str(k)] = {
            "base": {"rate_success": prior_success(bc, pi), "equal_linkage": round(float(np.mean(1.0 / np.bincount(bc)[bc])), 4),
                     "groups": int(len(base.covers)), "traffic_vs_sparse": traffic(base, bs, B, base_chunks)},
            "robust": {"rate_success": prior_success(rc, pi), "equal_linkage": round(float(np.mean(1.0 / np.bincount(rc)[rc])), 4),
                       "groups": int(len(rob.covers)), "size_median": int(np.median(size)), "size_p99": int(np.quantile(size, 0.99)),
                       "size_max": int(size.max()), "traffic_vs_sparse": traffic(rob, rs, B, base_chunks),
                       "sends_all_pct": round(100 * float(np.mean(rs < 0)), 3)},
            "secs": round(time.time() - t0, 1)}
        log("prior k", k, out["prior_robust"][str(k)])
        json.dump(out, open(OUT, "w"), indent=2)
    log("SAVED", OUT)


if __name__ == "__main__":
    main()
