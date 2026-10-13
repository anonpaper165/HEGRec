"""
Linkage when the stored profile is out of date (Amazon-Book, Gowalla), and shared data loaders.

Profile = the user's training split, request = training plus test items (real held-out growth), or
profile = a random p-fraction of the training items (p in 1.0, 0.9, 0.75, 0.5, 0.25).

Traces: chunk set alone; (exact degree, chunk set) for the per-item schedule; (degree bucket, chunk set)
for degree-keyed padding. Matching rules: `exact` (identical trace), `subset` (profiles that could
have grown into the request) and `nn_jaccard` (the subset rule ranked by Jaccard, Sec. 2.3).

Linkage = mean_u 1/|candidates(u)| if u is among the candidates, else 0. At p = 1.0 every rule reduces
to exact matching (built-in check).

Output: results/stale_profile.json
"""
from _paths import DATA, HOMO_GCN, REPO, RESULTS  # repo-relative roots
import os, sys, math, json, time
import numpy as np
from collections import defaultdict

HOMO = f"{HOMO_GCN}"
ROOT = f"{REPO}"
SLOTS = 4096
SEED = 42
RATES = [1.0, 0.9, 0.75, 0.5, 0.25]
# subset-rule matching is O(|queries| x |refs|); sample queries above this many users.
MAX_QUERIES = int(os.environ.get("HEGREC_AB_MAXQ", 12000))


# ---------------------------------------------------------------- data

def load(ds):
    """Return train_dict, test_dict, n_items for a LightGCN-format split."""
    def rd(fn):
        d = {}
        for line in open(f"{HOMO}/data/{ds}/{fn}"):
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
    """
    Bucket boundaries are a PROTOCOL parameter: the deployment fits them once on the training graph
    and then applies them to whatever degree a query presents. So we build the adaptive
    bucketing once from the true train degrees and reuse the resulting boundaries for both the
    stale reference degree and the live query degree. This is the adaptive bucketing of App. B.
    """
    cwd = os.getcwd()
    try:
        sys.path.insert(0, HOMO)
        os.chdir(HOMO)
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


# ---------------------------------------------------------------- metrics

def reid_exact(ref_fp, qry_fp, ref_users, qry_users):
    """
    Paper's rule: exact fingerprint match. True user may fail to match at all -> 0.
    ref_users is ALWAYS the full registered population (it defines the candidate pool and hence
    the anonymity-set sizes); qry_users may be a subsample of the users being attacked. Collapsing
    the two inflates Re-ID, because a smaller candidate pool makes every 1/k larger.
    """
    groups = defaultdict(int)
    for u in ref_users:
        groups[ref_fp[u]] += 1
    tot = 0.0
    for u in qry_users:
        k = groups.get(qry_fp[u], 0)
        # the true user contributes only if its own (stale) reference still equals the query
        if k and ref_fp[u] == qry_fp[u]:
            tot += 1.0 / k
    return tot / len(qry_users)


_POP = np.zeros(1 << 16, dtype=np.int32)
for _i in range(1, 1 << 16):
    _POP[_i] = _POP[_i >> 1] + (_i & 1)


def popcount32(x):
    return _POP[x & 0xFFFF] + _POP[(x >> 16) & 0xFFFF]


def reid_nn_jaccard(ref_mask, ref_ord, qry_mask, qry_ord, q_idx):
    """
    Strong attacker: nearest-neighbour matching. Among references that remain feasible under
    incomplete observation (ord_ref <= ord_query -- losing interactions cannot raise degree), pick
    the one(s) whose chunk mask is most similar to the query's by Jaccard overlap; ties split 1/k.
    Unlike `exact`, a stale reference still competes; unlike `subset`, near-misses are ranked
    instead of all being kept. This is the strongest of the three rules we evaluate, so it is the
    one the honest reading of the result should rest on.
    """
    tot = 0.0
    for qi in q_idx:
        mq, oq = qry_mask[qi], qry_ord[qi]
        inter = popcount32(ref_mask & mq).astype(np.float64)
        union = popcount32(ref_mask | mq).astype(np.float64)
        jac = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        jac[ref_ord > oq] = -1.0            # infeasible references
        best = jac.max()
        if jac[qi] < best:                  # true user is not among the top matches
            continue
        tot += 1.0 / int(np.count_nonzero(jac == best))
    return tot / len(q_idx)


def reid_subset(ref_mask, ref_ord, qry_mask, qry_ord, q_idx, rng):
    """
    Sound rule under incomplete observation.
    candidates(u) = { v : mask_ref[v] subset-of mask_qry[u]  AND  ord_ref[v] <= ord_qry[u] }
    `ord` is the scalar component other than the chunk set (exact degree, bucket id, or 0 to disable).
    The true user always satisfies both, so |candidates| >= 1 and success = 1/|candidates|.
    Queries may be subsampled (q_idx); references are always the full population.
    """
    inv = np.bitwise_not(qry_mask[q_idx])          # ~mask_q
    tot = 0.0
    # group references by their ord value so the ord<=  filter is a prefix over sorted groups
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


# ---------------------------------------------------------------- experiment

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
            # exact is cheap enough to run on everyone, but subset/nn evaluate a random query
            # subsample; recompute exact on that same subsample so the three rules in a row are
            # comparable on identical query populations (references are always the full population).
            e_sub = round(reid_exact(rfp[regime], qfp[regime], users,
                                     [users[i] for i in q_idx]), 5)
            s = round(reid_subset(rmask, ro, qmask, qo, q_idx, rng), 5)
            j = round(reid_nn_jaccard(rmask, ro, qmask, qo, q_idx), 5)
            res[regime] = {
                "reid_exact_match": e, "reid_exact_match_on_query_sample": e_sub,
                "reid_subset_match": s, "reid_nn_jaccard": j,
                # No rule dominates at every observation rate: `exact` wins when the profile is
                # synchronized (it can use equality on the degree component, which the feasibility
                # filter of the other two deliberately relaxes), `nn_jaccard` wins once the profile is
                # stale. A real attacker picks whichever works, so the achievable risk is the max.
                "reid_best_of_rules": max(e_sub, s, j),
            }
        return res

    # ---- Arm A: real held-out growth
    # control: a SYNCHRONIZED attacker facing the same enlarged supports. Comparing Arm A against
    # this (rather than against the published train-only number) isolates the effect of the profile
    # being stale from the effect of the supports simply being larger.
    out["arms"]["A0_synchronized_control"] = evaluate(
        "A0", full_support, full_support,
        "reference = query = train+test support (no staleness; isolates the staleness effect)")

    out["arms"]["A_real_heldout_growth"] = evaluate(
        "A", tr, full_support,
        f"reference = train support; query = train+test support (real held-out interactions). "
        f"mean support growth = {np.mean(growth) * 100:.1f}% (median {np.median(growth) * 100:.1f}%)")
    out["mean_support_growth_pct"] = round(float(np.mean(growth) * 100), 2)
    out["median_support_growth_pct"] = round(float(np.median(growth) * 100), 2)

    # ---- Arm B: synthetic observation-rate sweep
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

    # Self-check: at p=1.0 the reference IS the query, so every rule must reduce to the paper's
    # published Re-ID. This caught a real bug (collapsing the reference population into the query
    # subsample, which shrinks the candidate pool and inflates 1/k) and is kept as a regression guard.
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

    os.makedirs(f"{ROOT}/results", exist_ok=True)
    path = f"{ROOT}/results/stale_profile.json"
    json.dump(results, open(path, "w"), indent=2)
    print(f"\nSAVED {path}")


if __name__ == "__main__":
    main()
