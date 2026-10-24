"""Linkage against out-of-date profiles on Amazon-Book and Gowalla (App. D.1 exact matching), and shared data loaders.
Output: results/stale_profile.json
"""
from _paths import DATA, REPO, RESULTS
import os, sys, math, json, time
import numpy as np
from collections import defaultdict

REPO_DIR = f"{REPO}"
ROOT = f"{REPO}"
SLOTS = 4096
SEED = 42
RATES = [1.0, 0.9, 0.75, 0.5, 0.25]
MAX_QUERIES = int(os.environ.get("HEGREC_AB_MAXQ", 12000))


def load(ds):
    def rd(fn):
        d = {}
        for line in open(f"{DATA}/{ds}/{fn}"):
            p = line.split()
            if len(p) > 1:
                d[int(p[0])] = set(map(int, p[1:]))
        return d
    tr, te = rd("train.txt"), rd("test.txt")
    n_items = max(max(v) for v in tr.values() if v) + 1
    for v in te.values():
        if v:
            n_items = max(n_items, max(v) + 1)
    return tr, te, n_items


def make_bucket_fn(ds):
    cwd = os.getcwd()
    try:
        sys.path.insert(0, REPO_DIR)
        os.chdir(REPO_DIR)
        from src.dataset import RecDataset
        from src.bucketing import DegreeBucketing
        rds = RecDataset(f"data/{ds}", ds, rating_threshold=0)
        op = DegreeBucketing(rds, bucketing="adaptive")
        n_b = len(op._adaptive_boundaries) if op._adaptive_boundaries else len(op.user_buckets)
        return (lambda d: op.adaptive_bucket_id(d)), f"degree bucketing adaptive (auto, {n_b} buckets)"
    except Exception as e:
        print(f"  [warn] adaptive bucketing unavailable ({e}); using log2 fallback", flush=True)
        return (lambda d: int(math.ceil(math.log2(max(d, 2))))), "log2 fallback"
    finally:
        os.chdir(cwd)


def reid_exact(ref_fp, qry_fp, ref_users, qry_users):
    groups = defaultdict(int)
    for u in ref_users:
        groups[ref_fp[u]] += 1
    tot = 0.0
    for u in qry_users:
        k = groups.get(qry_fp[u], 0)
        if k and ref_fp[u] == qry_fp[u]:
            tot += 1.0 / k
    return tot / len(qry_users)


_POP = np.zeros(1 << 16, dtype=np.int32)
for _i in range(1, 1 << 16):
    _POP[_i] = _POP[_i >> 1] + (_i & 1)


def popcount32(x):
    return _POP[x & 0xFFFF] + _POP[(x >> 16) & 0xFFFF]


def reid_nn_jaccard(ref_mask, ref_ord, qry_mask, qry_ord, q_idx):
    tot = 0.0
    for qi in q_idx:
        mq, oq = qry_mask[qi], qry_ord[qi]
        inter = popcount32(ref_mask & mq).astype(np.float64)
        union = popcount32(ref_mask | mq).astype(np.float64)
        jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        jac[ref_ord > oq] = -1.0
        best = jac.max()
        if jac[qi] < best:
            continue
        tot += 1.0 / int(np.count_nonzero(jac == best))
    return tot / len(q_idx)


def reid_subset(ref_mask, ref_ord, qry_mask, qry_ord, q_idx, rng):
    inv = np.bitwise_not(qry_mask[q_idx])
    tot = 0.0
    order = np.argsort(ref_ord, kind="stable")
    rm_sorted = ref_mask[order]
    ro_sorted = ref_ord[order]
    for j, qi in enumerate(q_idx):
        hi = np.searchsorted(ro_sorted, qry_ord[qi], side="right")
        if hi == 0:
            tot += 1.0
            continue
        cand = np.count_nonzero((rm_sorted[:hi] & inv[j]) == 0)
        tot += 1.0 / max(cand, 1)
    return tot / len(q_idx)


def masks_of(support, users):
    m = np.zeros(len(users), dtype=np.uint32)
    for k, u in enumerate(users):
        acc = 0
        for i in support[u]:
            acc |= (1 << (i // SLOTS))
        m[k] = acc
    return m


def run_dataset(ds, label):
    t0 = time.time()
    tr, te, M = load(ds)
    users = sorted([u for u in tr if tr[u]])
    C = math.ceil(M / SLOTS)
    assert C <= 32, f"C={C} exceeds uint32 mask width"
    N = len(users)
    rng = np.random.default_rng(SEED)

    bucket_fn, bk_note = make_bucket_fn(ds)
    full_support = {u: tr[u] | te.get(u, set()) for u in users}
    growth = np.array([len(full_support[u]) / len(tr[u]) - 1.0 for u in users])

    q_idx = np.arange(N)
    sampled = False
    if N > MAX_QUERIES:
        q_idx = np.sort(rng.choice(N, MAX_QUERIES, replace=False))
        sampled = True

    out = {
        "dataset": label, "users": N, "items": M, "C_chunks": C,
        "mean_train_degree": round(float(np.mean([len(tr[u]) for u in users])), 2),
        "query_users_evaluated": int(len(q_idx)),
        "query_subsampled": sampled,
        "arms": {},
    }

    def evaluate(tag, ref_support, qry_support, note):
        ref_deg = {u: len(ref_support[u]) for u in users}
        qry_deg = {u: len(qry_support[u]) for u in users}
        ref_bk = {u: bucket_fn(d) for u, d in ref_deg.items()}
        qry_bk = {u: bucket_fn(d) for u, d in qry_deg.items()}

        rmask = masks_of(ref_support, users)
        qmask = masks_of(qry_support, users)
        rdeg = np.array([ref_deg[u] for u in users], dtype=np.int64)
        qdeg = np.array([qry_deg[u] for u in users], dtype=np.int64)
        rbk = np.array([ref_bk[u] for u in users], dtype=np.int64)
        qbk = np.array([qry_bk[u] for u in users], dtype=np.int64)
        zero = np.zeros(N, dtype=np.int64)

        rfp = {"ch2": {u: rmask[k] for k, u in enumerate(users)},
               "sparse": {u: (rdeg[k], rmask[k]) for k, u in enumerate(users)},
               "bucketed": {u: (rbk[k], rmask[k]) for k, u in enumerate(users)}}
        qfp = {"ch2": {u: qmask[k] for k, u in enumerate(users)},
               "sparse": {u: (qdeg[k], qmask[k]) for k, u in enumerate(users)},
               "bucketed": {u: (qbk[k], qmask[k]) for k, u in enumerate(users)}}

        res = {"note": note, "bucketing": bk_note}
        for regime, ro, qo in [("ch2", zero, zero), ("sparse", rdeg, qdeg), ("bucketed", rbk, qbk)]:
            e = round(reid_exact(rfp[regime], qfp[regime], users, users), 5)
            e_sub = round(reid_exact(rfp[regime], qfp[regime], users,
                                     [users[i] for i in q_idx]), 5)
            s = round(reid_subset(rmask, ro, qmask, qo, q_idx, rng), 5)
            j = round(reid_nn_jaccard(rmask, ro, qmask, qo, q_idx), 5)
            res[regime] = {
                "reid_exact_match": e, "reid_exact_match_on_query_sample": e_sub,
                "reid_subset_match": s, "reid_nn_jaccard": j,
                "reid_best_of_rules": max(e_sub, s, j),
            }
        return res

    out["arms"]["A0_synchronized_control"] = evaluate(
        "A0", full_support, full_support,
        "reference = query = train+test support (no staleness; isolates the staleness effect)")

    out["arms"]["A_real_heldout_growth"] = evaluate(
        "A", tr, full_support,
        f"reference = train support; query = train+test support (real held-out interactions). "
        f"mean support growth = {np.mean(growth) * 100:.1f}% (median {np.median(growth) * 100:.1f}%)")
    out["mean_support_growth_pct"] = round(float(np.mean(growth) * 100), 2)
    out["median_support_growth_pct"] = round(float(np.median(growth) * 100), 2)

    sweep = {}
    for p in RATES:
        if p >= 1.0:
            ref = tr
        else:
            ref = {}
            for u in users:
                items = sorted(tr[u])
                k = max(1, int(round(p * len(items))))
                ref[u] = set(rng.choice(items, size=k, replace=False).tolist())
        sweep[f"p={p}"] = evaluate(f"B_p{p}", ref, tr,
                                   f"reference = uniform random {int(p * 100)}% of train support; "
                                   f"query = full train support (synthetic; no timestamps available)")
    out["arms"]["B_observation_rate_sweep"] = sweep

    published = {"Gowalla": {"ch2": 0.0345, "sparse": 0.3483, "bucketed": 0.1081},
                 "Amazon-Book": {"ch2": 0.5878, "sparse": 0.8962, "bucketed": 0.7265}}[label]
    p1 = sweep["p=1.0"]
    check = {}
    for k, want in published.items():
        got = p1[k]["reid_exact_match"]
        check[k] = {"published": want, "recomputed": got,
                    "ok": bool(abs(got - want) < 0.0015)}
    out["validation_p1_matches_published"] = check
    if not all(v["ok"] for v in check.values()):
        print(f"  [FAIL] {label} p=1.0 does not reproduce published Re-ID: {check}", flush=True)
    else:
        print(f"  [ok] {label} p=1.0 reproduces published Re-ID for all three regimes", flush=True)

    out["runtime_sec"] = round(time.time() - t0, 1)
    return out


def main():
    results = {
        "experiment": "stale / partial reference-profile robustness of the Re-ID threat model",
        "seed": SEED,
        "slots": SLOTS,
        "convention": "Re-ID = mean_u 1/|candidates(u)| if u in candidates else 0 "
                      "(identical to the paper's 1/k rule; p=1.0 reproduces published values)",
        "datasets": {},
    }
    for ds, label in [("gowalla", "Gowalla"), ("amazon-book", "Amazon-Book")]:
        print(f"\n=== {label} ===", flush=True)
        r = run_dataset(ds, label)
        results["datasets"][label] = r
        print(json.dumps(r["arms"]["A_real_heldout_growth"], indent=2), flush=True)
        print(f"[{label}] done in {r['runtime_sec']}s", flush=True)

    os.makedirs(RESULTS, exist_ok=True)
    path = f"{RESULTS}/stale_profile.json"
    json.dump(results, open(path, "w"), indent=2)
    print(f"\nSAVED {path}")


if __name__ == "__main__":
    main()
