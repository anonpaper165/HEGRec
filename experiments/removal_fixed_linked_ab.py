"""Partial histories, one and two linked requests (Table 9 linkage rows; §5.3-5.4; Fig. 3b).
REMOVAL_PS=1.0 REMOVAL_NSEED=1 REMOVAL_OUT=removal_fixed_linked_sparse_ab.json gives the no-removal value 0.765.
Output: results/removal_fixed_linked_ab.json
"""
from _paths import DATA, REPO, RESULTS
import os, sys, json, math, time
import numpy as np
sys.path.insert(0, f"{REPO}")
from src.dataset import RecDataset

SLOTS, SPLIT = 4096, 0.8
PS = tuple(float(x) for x in os.environ["REMOVAL_PS"].split(",")) if os.environ.get("REMOVAL_PS") else (0.75, 0.5, 0.25)
NSEED = int(os.environ.get("REMOVAL_NSEED", 5))
OUT = os.environ.get("REMOVAL_OUT", "removal_fixed_linked_ab.json")


def mask(items):
    m = 0
    for c in set((np.asarray(items) // SLOTS).tolist()): m |= 1 << int(c)
    return m


def bits(M, C):
    M = np.array(M, dtype=np.uint64)
    return ((M[:, None] >> np.arange(C, dtype=np.uint64)) & np.uint64(1)).astype(np.float32)


def scores(Qb, R, b0, b1):
    I = Qb[b0:b1] @ R.T; nq = Qb[b0:b1].sum(1, keepdims=True); rn = R.sum(1)[None, :]
    return I / np.maximum(rn + nq - I, 1), I / np.maximum(nq, 1)


def hit(S, own, rn_tie=None):
    rows = np.arange(S.shape[0]); top = S >= S.max(1, keepdims=True) - 1e-6
    if rn_tie is not None:
        m = np.where(top, rn_tie[None, :], np.inf).min(1, keepdims=True); top &= rn_tie[None, :] == m
    return np.where(top[rows, own], 1.0 / top.sum(1), 0.0)


def attack(Q1, Q2, R1, R2, C):
    A1, A2, R1b, R2b = bits(Q1, C), bits(Q2, C), bits(R1, C), bits(R2, C)
    U = bits([a | b for a, b in zip(Q1, Q2)], C); rn2 = R2b.sum(1)
    out = {k: [] for k in ("end_jaccard", "end_share", "sum_jaccard", "sum_share", "union_jaccard", "union_share")}
    for b0 in range(0, len(Q1), 1024):
        b1 = min(b0 + 1024, len(Q1)); own = np.arange(b0, b1)
        j1, s1 = scores(A1, R1b, b0, b1); j2, s2 = scores(A2, R2b, b0, b1); ju, su = scores(U, R2b, b0, b1)
        out["end_jaccard"].append(hit(j2, own)); out["end_share"].append(hit(s2, own, rn2))
        out["sum_jaccard"].append(hit(j1 + j2, own)); out["sum_share"].append(hit(s1 + s2, own, rn2))
        out["union_jaccard"].append(hit(ju, own)); out["union_share"].append(hit(su, own, rn2))
    return {k: float(np.concatenate(v).mean()) for k, v in out.items()}


def run(sub):
    ds = RecDataset(f"{DATA}/{sub}", sub, rating_threshold=0); C = int(math.ceil(ds.n_items / SLOTS))
    users = [u for u in sorted(ds.train_dict) if ds.train_dict[u]]
    rng = np.random.default_rng(0); full, start = [], []
    for u in users:
        s = np.array(sorted(ds.train_dict[u])); keep = rng.random(len(s)) < SPLIT
        if not keep.any(): keep[rng.integers(len(s))] = True
        full.append(s); start.append(keep)
    tu = [j for j, u in enumerate(users) if len(ds.test_dict.get(u, []))]
    R1 = [mask(full[j][start[j]]) for j in tu]; R2 = [mask(full[j]) for j in tu]
    res = {"users_scored": len(tu), "C": C, "p": {}}
    for p in PS:
        t0 = time.time(); acc = {"fixed": [], "redrawn": []}
        for seed in range(NSEED):
            r2 = np.random.default_rng(100 + seed)
            F1, F2, D1, D2 = [], [], [], []
            for j in tu:
                s, st = full[j], start[j]
                c = r2.random(len(s)) < p
                if not c.any(): c[r2.integers(len(s))] = True
                F2.append(mask(s[c])); F1.append(mask(s[c & st]))
                d1 = r2.random(len(s)) < p
                D1.append(mask(s[d1 & st])); D2.append(mask(s[c]))
            acc["fixed"].append(attack(F1, F2, R1, R2, C)); acc["redrawn"].append(attack(D1, D2, R1, R2, C))
        rec = {}
        for arm, rows in acc.items():
            mean = {k: round(float(np.mean([r[k] for r in rows])), 5) for k in rows[0]}
            rec[arm] = {"rules": mean, "R1_end": max(mean["end_jaccard"], mean["end_share"]),
                        "R2_linked": max(mean.values())}
        res["p"][str(p)] = rec; print(sub, p, json.dumps(rec), f"[{time.time()-t0:.0f}s]", flush=True)
    return res


if __name__ == "__main__":
    out = {"note": __doc__.strip().split("\n\n")[0], "datasets": {}}
    for name, sub in (("Amazon-Book", "amazon-book"), ("Gowalla", "gowalla")):
        out["datasets"][name] = run(sub)
        json.dump(out, open(os.path.join(RESULTS, OUT), "w"), indent=2)
    print("SAVED")
