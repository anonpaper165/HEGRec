"""
AR2023-HK: loading, chunk sets and linkage at scale (Amazon Reviews 2023, Home_and_Kitchen, 5-core).

About 2.9M users, 764K items (C = 187 chunks of 4,096 items) and density 0.0013%. Interactions are
timestamped, so a stored profile can be a genuine snapshot: profile = everything the user did up to a
date T, request = the full history.

Matching rules: `exact` (identical chunk set, all users) and `nn_jaccard` (keep the profiles whose chunk
set lies inside the request's, rank them by Jaccard; Sec. 2.3). An inverted chunk -> users index scores
only users sharing a chunk with the request; the true profile is always among them.

Check: the same code first reproduces Amazon-Book linkage (0.5878 chunk set, 0.8962 per-item trace).

Output: results/dataset_amazon2023_hk.json
"""
from _paths import DATA, HOMO_GCN, REPO, RESULTS  # repo-relative roots
import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_v] = "1"

import csv, json, math, time, datetime as dt
import numpy as np
from collections import defaultdict

# Raw Amazon Reviews 2023 Home_and_Kitchen export (user, item, rating, timestamp).
# Not redistributed with this tree; point HEGREC_HK_CSV at the downloaded file.
HK = os.environ.get("HEGREC_HK_CSV", os.path.join(DATA, "ar2023_hk", "hk.csv"))
HOMO = f"{HOMO_GCN}"
ROOT = f"{REPO}"
SLOTS = 4096
SEED = 42
# Query sample for the Jaccard nearest-neighbour rule. Exact matching is over the full
# population; the NN rule cannot be, since each query scans every reference sharing a chunk.
# Env-overridable so a larger sample can be run without changing the default that produced
# the committed JSON. hk_random_removal_control.py imports this value, so both stay matched.
NN_QUERIES = int(os.environ.get("NN_QUERIES", 1000))
UNSEEN_TARGETS = [0.05, 0.10, 0.25, 0.50]


def reid(keys):
    """Linkage under exact matching: mean_u 1/|class(u)| (Observation 1)."""
    g = defaultdict(int)
    for k in keys:
        g[k] += 1
    return float(np.mean([1.0 / g[k] for k in keys]))


def reid_paired(ref_fp, qry_fp):
    """Exact-match Re-ID with a stale reference: the true user scores only if its own (stale)
    fingerprint still equals its query fingerprint. Reference pool is the full population."""
    g = defaultdict(int)
    for f in ref_fp:
        g[f] += 1
    tot = 0.0
    for k in range(len(qry_fp)):
        c = g.get(qry_fp[k], 0)
        if c and ref_fp[k] == qry_fp[k]:
            tot += 1.0 / c
    return tot / len(qry_fp)


# ------------------------------------------------------------------ regression guard

def guard_amazon_book():
    tr = {}
    for line in open(f"{HOMO}/data/amazon-book/train.txt"):
        p = line.split()
        if len(p) > 1:
            tr[int(p[0])] = sorted(map(int, p[1:]))
    us = [u for u in sorted(tr) if tr[u]]
    deg = [len(tr[u]) for u in us]
    cs = [tuple(sorted({i // SLOTS for i in tr[u]})) for u in us]
    got = {"channel2_chunkindex_reid": round(reid(cs), 4),
           "full_reid": round(reid(list(zip(deg, cs))), 4)}
    want = {"channel2_chunkindex_reid": 0.5878, "full_reid": 0.8962}
    ok = all(abs(got[k] - want[k]) < 0.0015 for k in want)
    print(f"  [{'ok' if ok else 'FAIL'}] Amazon-Book guard: {got} vs published {want}", flush=True)
    return {"recomputed": got, "published": want, "ok": bool(ok)}


# ------------------------------------------------------------------ load

def load_hk():
    t0 = time.time()
    umap, imap = {}, {}
    ur, ir, tr_ = [], [], []
    with open(HK, newline="") as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            u = umap.get(r[0])
            if u is None:
                u = umap[r[0]] = len(umap)
            i = imap.get(r[1])
            if i is None:
                i = imap[r[1]] = len(imap)      # first-appearance order = "original" layout
            ur.append(u); ir.append(i); tr_.append(int(r[3]))
    U = np.asarray(ur, dtype=np.int64); del ur
    I = np.asarray(ir, dtype=np.int64); del ir
    T = np.asarray(tr_, dtype=np.int64); del tr_
    print(f"  loaded {len(U):,} rows / {len(umap):,} users / {len(imap):,} items "
          f"in {time.time()-t0:.0f}s", flush=True)
    return U, I, T, len(umap), len(imap)


def user_chunksets(U, ch, nU, C):
    """Vectorised: unique (user, chunk) pairs -> per-user sorted chunk tuple + CSR arrays."""
    key = np.unique(U * C + ch)
    uu = (key // C).astype(np.int64)
    uc = (key % C).astype(np.int32)
    cnt = np.bincount(uu, minlength=nU)
    off = np.concatenate([[0], np.cumsum(cnt)]).astype(np.int64)
    tup = [tuple(uc[off[u]:off[u + 1]].tolist()) for u in range(nU)]
    return tup, uc, off, cnt.astype(np.int32)


def inverted_index(uc, off, nU, C):
    """chunk -> array of users holding it (CSR over chunks)."""
    users = np.repeat(np.arange(nU, dtype=np.int64), np.diff(off))
    order = np.argsort(uc, kind="stable")
    cs = uc[order]
    us = users[order]
    starts = np.searchsorted(cs, np.arange(C), side="left")
    ends = np.searchsorted(cs, np.arange(C), side="right")
    return us, starts, ends


def reid_nn(ref_us, ref_starts, ref_ends, ref_nact, ref_ord,
            qry_tup, qry_nact, qry_ord, q_idx, nU):
    """
    Jaccard nearest-neighbour over the chunk set, using the inverted index.
    |A n Q| is accumulated only over users sharing a chunk with Q; |A u Q| = |A| + |Q| - |A n Q|.
    Feasibility filter ref_ord <= qry_ord (losing interactions cannot raise degree).
    """
    acc = np.zeros(nU, dtype=np.int16)
    tot = 0.0
    for qi in q_idx:
        Q = qry_tup[qi]
        touched = []
        for c in Q:
            sl = ref_us[ref_starts[c]:ref_ends[c]]
            acc[sl] += 1
            touched.append(sl)
        cand = np.unique(np.concatenate(touched)) if touched else np.empty(0, dtype=np.int64)
        inter = acc[cand].astype(np.float64)
        union = ref_nact[cand] + len(Q) - inter
        j = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        j[ref_ord[cand] > qry_ord[qi]] = -1.0
        best = j.max() if len(j) else -1.0
        pos = np.flatnonzero(cand == qi)
        if len(pos) and j[pos[0]] >= best:
            tot += 1.0 / int(np.count_nonzero(j == best))
        for sl in touched:
            acc[sl] = 0
    return tot / len(q_idx)


# ------------------------------------------------------------------ main

def main():
    res = {"dataset": "Amazon Reviews 2023 - Home_and_Kitchen (5-core, rating_only)",
           "source": "McAuley-Lab/Amazon-Reviews-2023, benchmark/5core/rating_only",
           "slots": SLOTS, "seed": SEED,
           "convention": "per-item trace = (exact degree, chunk set); linkage = mean_u 1/|class|"}

    print("=== regression guard ===", flush=True)
    res["validation_amazon_book"] = guard_amazon_book()

    print("=== load ===", flush=True)
    U, I, T, nU, nI = load_hk()
    C = math.ceil(nI / SLOTS)
    ch = I // SLOTS
    deg = np.bincount(U, minlength=nU).astype(np.int64)

    t0 = time.time()
    tup, uc, off, nact = user_chunksets(U, ch, nU, C)
    print(f"  chunk sets built in {time.time()-t0:.0f}s", flush=True)

    res["scale"] = {
        "users": int(nU), "items": int(nI), "interactions": int(len(U)), "C_chunks": int(C),
        "mean_degree": round(float(deg.mean()), 2), "median_degree": int(np.median(deg)),
        "max_degree": int(deg.max()),
        "density_pct": round(float(len(U) / (nU * nI) * 100), 6),
        "mean_active_chunks": round(float(nact.mean()), 3),
        "dense_over_sparse_upload": round(C / float(nact.mean()), 2),
        "time_range": [str(dt.datetime.utcfromtimestamp(int(T.min()) / 1000).date()),
                       str(dt.datetime.utcfromtimestamp(int(T.max()) / 1000).date())]}
    print(f"  {res['scale']}", flush=True)

    print("=== (1) audit metrics at scale ===", flush=True)
    dl = deg.tolist()
    res["audit_at_scale"] = {
        "channel1_degree_reid": round(reid(dl), 6),
        "channel2_chunkindex_reid": round(reid(tup), 6),
        "full_reid": round(reid(list(zip(dl, tup))), 6),
        "random_guess": float(1.0 / nU)}
    for k, v in res["audit_at_scale"].items():
        print(f"    {k:28} {v:.6g}", flush=True)

    print("=== (2) real temporal staleness ===", flush=True)
    rng = np.random.default_rng(SEED)
    arms = {}
    for frac in UNSEEN_TARGETS:
        t0 = time.time()
        Tcut = int(np.quantile(T, 1.0 - frac))
        keep = T <= Tcut
        rU, rch = U[keep], ch[keep]
        rdeg = np.bincount(rU, minlength=nU).astype(np.int64)
        rtup, ruc, roff, rnact = user_chunksets(rU, rch, nU, C)
        have = np.flatnonzero(rdeg > 0)
        q_idx = np.sort(rng.choice(have, min(NN_QUERIES, len(have)), replace=False))
        ru, rs, re_ = inverted_index(ruc, roff, nU, C)

        a = {"snapshot_date": str(dt.datetime.utcfromtimestamp(Tcut / 1000).date()),
             "unseen_interaction_fraction": round(float((~keep).mean()), 4),
             "users_with_nonempty_reference": int(len(have)),
             "nn_query_sample": int(len(q_idx)),
             "ch2_reid_exact": round(reid_paired(rtup, tup), 6),
             "full_reid_exact": round(reid_paired(list(zip(rdeg.tolist(), rtup)),
                                                  list(zip(dl, tup))), 6),
             "full_reid_nn_jaccard": round(
                 reid_nn(ru, rs, re_, rnact.astype(np.int64), rdeg,
                         tup, nact, deg, q_idx, nU), 6)}
        # CAREFUL: the two rules do not share a denominator.
        #   full_reid_exact      -- averaged over ALL nU users (reid_paired)
        #   full_reid_nn_jaccard -- averaged over a sample of `have` (rdeg > 0), i.e. only the
        #                           users the operator actually holds a stale profile for
        # A user with an empty stale reference cannot be matched by either rule, so the two are
        # comparable only after rescaling the NN number by |have|/nU. We record both so nothing
        # downstream has to guess which population a figure refers to, and we keep best_of_rules
        # only as a within-population upper bound -- it is NOT a max over commensurable values.
        a["attackable_fraction"] = round(len(have) / nU, 6)
        a["full_reid_nn_jaccard_population"] = round(
            a["full_reid_nn_jaccard"] * len(have) / nU, 6)
        a["full_reid_best_of_rules"] = max(a["full_reid_exact"], a["full_reid_nn_jaccard"])
        a["full_reid_best_of_rules_population"] = max(
            a["full_reid_exact"], a["full_reid_nn_jaccard_population"])
        arms[f"unseen_{int(frac*100)}pct"] = a
        print(f"    unseen {frac:.0%} (snapshot {a['snapshot_date']}): exact={a['full_reid_exact']:.4f} "
              f"nn={a['full_reid_nn_jaccard']:.4f} best={a['full_reid_best_of_rules']:.4f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
    res["temporal_staleness"] = arms

    os.makedirs(f"{ROOT}/results", exist_ok=True)
    json.dump(res, open(f"{ROOT}/results/dataset_amazon2023_hk.json", "w"), indent=2)
    print(f"\nSAVED {ROOT}/results/dataset_amazon2023_hk.json")


if __name__ == "__main__":
    main()
