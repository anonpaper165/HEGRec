"""How fast do redrawn decoys wear off when the operator sees several requests?

decoy_frontier_hk.py prices redrawn decoys against ONE request. The limitation it
leaves open: an operator that can tell R requests come from the same user (one
session, one device token) sees R traces y_r = B_u union D_r with independent D_r,
and can intersect them. This measures the Bayes-optimal attacker on that tuple.

Channel. D_r is a uniform k-subset of [C], independent across r and of the user,
and the user's active chunks B_u are the same in every request (worst case: the
history does not change between requests). For a registered mask m,

    P(y_1..y_R | m) = prod_r C(|m|, k - a_r) / C(C, k),   a_r = |y_r| - |m|,

zero unless m is contained in every y_r, i.e. in their intersection.

Estimator. Same as decoy_frontier_hk.py, extended to tuples: success is
V_R = (1/N) sum_{tuples} max_u P(tuple | u); sampling a user uniformly and
redrawing its R decoy sets samples P(tuple), so averaging
max_m L(m) / sum_m n_m L(m) estimates V_R. Products are taken in log space
(C(187,32)^R overflows float64 for R >= 9).

Guard: R=1 must reproduce decoy_frontier_hk.json inside sampling error.

Output: results/decoy_multirequest.json
"""
from _paths import DATA, RESULTS
import json, math, os, time
import numpy as np
from decoy_frontier_hk import pack, load_amazon_book, load_hk

SEED = 42
RS = [1, 2, 3, 5, 10, 20]
PLAN = {"Amazon-Book": ([3, 5, 8, 16], 20000),
        "AR2023-HK": ([8, 16, 32], 2000)}


def estimate(sup_list, arr, pc, nm, W, C, k, R, n_q, rng):
    N = len(sup_list)
    lcmb = np.full((C + 1, k + 1), -np.inf)
    for p in range(C + 1):
        for j in range(min(p, k) + 1):
            lcmb[p, j] = math.lgamma(p + 1) - math.lgamma(j + 1) - math.lgamma(p - j + 1)
    who = rng.randint(0, N, size=n_q)
    vals = np.empty(n_q)
    for t in range(n_q):
        base = set(sup_list[who[t]])
        ys = [base | {int(b) for b in rng.choice(C, size=k, replace=False)}
              for _ in range(R)]
        inter = set.intersection(*ys)
        y = np.zeros(W, dtype=np.uint64)
        for c in inter:
            y[c >> 6] |= np.uint64(1) << np.uint64(c & 63)
        ny = ~y
        ok = (arr[:, 0] & ny[0]) == 0
        for w in range(1, W):
            ok &= (arr[:, w] & ny[w]) == 0
        p = pc[ok]
        ll = np.zeros(p.shape[0])
        for yr in ys:
            a = len(yr) - p
            j = k - a
            good = (j >= 0) & (j <= k)
            term = np.full(p.shape[0], -np.inf)
            term[good] = lcmb[p[good], j[good]]
            ll += term
        top = ll.max()
        if not np.isfinite(top):
            vals[t] = 0.0
            continue
        wt = np.exp(ll - top)
        vals[t] = 1.0 / float((nm[ok] * wt).sum())
    return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(n_q))


def run(name, sup):
    ks, n_q = PLAN[name]
    users = sorted(sup)
    sup_list = [sorted(sup[u]) for u in users]
    C = max(max(v) for v in sup_list) + 1
    arr, pc, nm, W = pack([set(s) for s in sup_list], C)
    rng = np.random.RandomState(SEED)
    out = {"users": len(users), "C": C, "n_queries": n_q, "k": {}}
    for k in ks:
        out["k"][str(k)] = {}
        for R in RS:
            t0 = time.time()
            V, se = estimate(sup_list, arr, pc, nm, W, C, k, R, n_q, rng)
            out["k"][str(k)][str(R)] = {"V": round(V, 6), "stderr": round(se, 6)}
            print("  %s k=%-3d R=%-3d V=%.4f +-%.4f (%.0fs)" % (
                name, k, R, V, se, time.time() - t0), flush=True)
    return out


if __name__ == "__main__":
    res = {"seed": SEED, "Rs": RS,
           "channel": "R requests, y_r = B_u union D_r, D_r iid uniform k-subsets; "
                      "attacker knows the R requests share a user",
           "datasets": {}}
    ref = json.load(open(os.path.join(RESULTS, "decoy_frontier_hk.json")))["datasets"]
    for name, loader in [("Amazon-Book", load_amazon_book), ("AR2023-HK", load_hk)]:
        res["datasets"][name] = run(name, loader())
        one = {r["k"]: r for r in ref[name]["rows"]}
        for k, byR in res["datasets"][name]["k"].items():
            e = one.get(int(k))
            if e:
                d = abs(byR["1"]["V"] - e["V"])
                tol = max(4 * math.hypot(byR["1"]["stderr"], e["stderr"]), 0.005)
                print("  guard %s k=%s R=1 %.4f vs %.4f %s" % (
                    name, k, byR["1"]["V"], e["V"], "ok" if d <= tol else "MISMATCH"))
        with open(os.path.join(RESULTS, "decoy_multirequest.json"), "w") as f:
            json.dump(res, f, indent=1)
    print("wrote results/decoy_multirequest.json")
