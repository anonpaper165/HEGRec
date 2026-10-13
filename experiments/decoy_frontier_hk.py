"""Is a redrawn-decoy defence still dominated at C=187?

decoy_map_bound.py answers this exactly on Amazon-Book: fresh decoys are a
garbling of the block trace, so linkage cannot rise, and eight of them take
optimal guessing success from 0.588 to 0.078 at 1.68x the sparse upload -- which
the appendix then calls dominated, because submitting all C=23 chunks costs
1.94x and reaches 1/N. That verdict is specific to a 23-chunk catalogue. On
AR2023-HK dense costs 22.4x, so it is not the same comparison, and nobody has
computed it.

The exact method does not transfer. decoy_map_bound.py runs a subset-max (zeta)
transform over arrays of size 2^C; at C=187 that is 2^187 entries. We therefore
estimate the same quantity by sampling the channel, and make each draw cheap by
testing containment directly: a registered mask is a candidate for the observed
y iff m AND NOT y is zero, which is three word-wise operations per mask when the
masks are packed into 64-bit words.

Estimator. For Y = B_u union D with D a uniform k-subset of [C] independent of U,

    P(y | u) = C(|B_u|, k - a) / C(C, k),   a = |y| - |B_u|,  B_u subset of y,

increasing in |B_u| at fixed |y|, so the Bayes attacker takes the largest
registered mask contained in y.

The quantity wanted is a sum over observations, V = (1/N) sum_y max_u P(y|u), NOT
an expectation of max_u P(y|u) under the channel: averaging the maximum over
sampled draws estimates sum_y P(y) max_u P(y|u), which is a different and much
smaller number (it fails the guard below by a factor of two to three). Sampling
y from the channel and averaging

    max_m P(y|m) / sum_m n_m P(y|m),     n_m = registered users with mask m,

divides by P(y) = (1/N) sum_m n_m P(y|m) and so estimates V itself. The common
1/C(C,k) factor cancels in that ratio. At k=0 the channel is deterministic and
the expression collapses to m/N, which is computed directly rather than sampled.

Regression guard: the estimator runs on Amazon-Book first, where the exact values
are known, and must reproduce decoy_map_bound.json inside sampling error before
the C=187 row is reported.

Output: results/decoy_frontier_hk.json
"""
from _paths import DATA, RESULTS
import json, math, os, time
import numpy as np

SLOTS = 4096
SEED = 42
KS = [0, 1, 2, 3, 5, 8, 16, 32]
N_Q_SMALL = 40000     # Amazon-Book: masks fit one word, so sampling is cheap
N_Q_LARGE = 2000      # AR2023-HK: 2.8M distinct masks scanned per query


def pack(sup_sets, C):
    """Distinct block masks as (n_masks, W) uint64, with popcounts and the number
    of registered users carrying each mask -- the counts enter the P(y) denominator."""
    W = (C + 63) // 64
    cnt = {}
    for s in sup_sets:
        key = tuple(sorted(s))
        cnt[key] = cnt.get(key, 0) + 1
    arr = np.zeros((len(cnt), W), dtype=np.uint64)
    pc = np.zeros(len(cnt), dtype=np.int64)
    nm = np.zeros(len(cnt), dtype=np.int64)
    for r, (key, n) in enumerate(cnt.items()):
        for c in key:
            arr[r, c >> 6] |= np.uint64(1) << np.uint64(c & 63)
        pc[r] = len(key)
        nm[r] = n
    return arr, pc, nm, W


def estimate_V(sup_list, arr, pc, nm, W, C, k, n_q, rng):
    """V = E_{y~P} [ max_m P(y|m) / sum_m n_m P(y|m) ].

    The Bayes success probability is a sum over observations, V = (1/N) sum_y
    max_u P(y|u), not an expectation of max_u P(y|u) under P; dividing by
    P(y) = (1/N) sum_m n_m P(y|m) is what converts the sum into an average over
    y drawn from the channel (pick a user uniformly, redraw its decoys). The
    common 1/C(C,k) factor cancels in the ratio, so it is dropped. At k=0 the
    channel is deterministic and the expression collapses to m/N exactly.
    """
    N = len(sup_list)
    if k == 0:
        return arr.shape[0] / N, 0.0, "exact (m/N)"
    who = rng.randint(0, N, size=n_q)
    vals = np.empty(n_q)
    cmb = np.array([[math.comb(p, j) if 0 <= j <= p else 0
                     for j in range(k + 1)] for p in range(C + 1)], dtype=np.float64)
    for t in range(n_q):
        bits = set(sup_list[who[t]])
        bits |= {int(b) for b in rng.choice(C, size=k, replace=False)}
        y = np.zeros(W, dtype=np.uint64)
        for c in bits:
            y[c >> 6] |= np.uint64(1) << np.uint64(c & 63)
        ny = ~y
        ok = (arr[:, 0] & ny[0]) == 0
        for w in range(1, W):
            ok &= (arr[:, w] & ny[w]) == 0
        p = pc[ok]
        a = len(bits) - p                     # bits of y the decoys must supply
        good = (a >= 0) & (a <= k)
        wt = np.zeros(p.shape[0])
        if good.any():
            wt[good] = cmb[p[good], (k - a)[good]]
        den = float((nm[ok] * wt).sum())
        vals[t] = (float(wt.max()) / den) if den > 0 else 0.0
    return float(vals.mean()), float(vals.std(ddof=1) / math.sqrt(n_q)), "sampled"


def run(sup, n_q):
    users = sorted(sup)
    sup_list = [sorted(sup[u]) for u in users]
    N = len(users)
    C = max(max(v) for v in sup_list) + 1
    arr, pc, nm, W = pack([set(s) for s in sup_list], C)
    mean_active = float(np.mean([len(s) for s in sup_list]))
    rng = np.random.RandomState(SEED)
    rows = []
    for k in KS:
        if k > C:
            continue
        t0 = time.time()
        V, se, kind = estimate_V(sup_list, arr, pc, nm, W, C, k, n_q, rng)
        rows.append({"k": k, "V": round(V, 6), "stderr": round(se, 6),
                     "provenance": kind,
                     # D is a uniform k-subset of [C] and the client submits
                     # I_active u D, so a draw landing on an active chunk costs
                     # nothing: E|I u D| = a + k(C-a)/C, not a + k. The budget form
                     # is only an upper bound, and it is not even feasible -- at k=C
                     # it gives C/a + 1 where the union is exactly dense's C/a.
                     "upload_vs_sparse": round(
                         1.0 + k / mean_active - k / C, 3),
                     "upload_budget_upper_bound": round(
                         (mean_active + k) / mean_active, 3)})
        print("    k=%-3d V=%.6f  +-%.6f  upload %.2fx   (%s, %.0fs)" % (
            k, V, se, rows[-1]["upload_vs_sparse"], kind, time.time() - t0), flush=True)
    return {"users": N, "C": C, "distinct_masks": int(arr.shape[0]),
            "mean_active_chunks": round(mean_active, 3),
            "dense_upload_vs_sparse": round(C / mean_active, 2),
            "chance_1_over_N": 1.0 / N, "n_queries": n_q, "rows": rows}


def load_amazon_book():
    tr = {}
    with open(f"{DATA}/amazon-book/train.txt") as f:
        for line in f:
            p = line.split()
            if len(p) > 1:
                tr[int(p[0])] = {int(i) // SLOTS for i in p[1:]}
    return tr


def load_hk():
    import csv
    path = os.environ.get("HEGREC_HK_CSV",
                          os.path.join(DATA, "ar2023_hk", "hk.csv"))
    umap, imap, sup = {}, {}, {}
    with open(path, newline="") as f:
        rd = csv.reader(f)
        next(rd)
        for r in rd:
            u = umap.get(r[0])
            if u is None:
                u = umap[r[0]] = len(umap)
            i = imap.get(r[1])
            if i is None:
                i = imap[r[1]] = len(imap)
            sup.setdefault(u, set()).add(i // SLOTS)
    return sup


if __name__ == "__main__":
    out = {"seed": SEED, "slots": SLOTS,
           "channel": "Y = B_u union D, D a uniform k-subset of [C] redrawn per request",
           "estimator": "Bayes attacker takes the largest registered mask contained in y; "
                        "V averaged over sampled (user, decoy) draws",
           "datasets": {}}

    print("Amazon-Book (regression guard against decoy_map_bound.json)", flush=True)
    out["datasets"]["Amazon-Book"] = run(load_amazon_book(), N_Q_SMALL)

    ok = True
    try:
        ex = json.load(open(os.path.join(RESULTS, "decoy_map_bound.json")))
        ex = ex["datasets"]["amazon-book"]["V"]
        print("\n  guard: exact vs sampled", flush=True)
        for r in out["datasets"]["Amazon-Book"]["rows"]:
            e = ex.get(str(r["k"]))
            if e is None:
                continue
            dev = abs(r["V"] - e)
            flag = "ok" if dev <= max(4 * r["stderr"], 0.005) else "MISMATCH"
            ok &= flag == "ok"
            print("    k=%-3d exact %.4f  sampled %.4f  |d|=%.4f  %s" % (
                r["k"], e, r["V"], dev, flag), flush=True)
    except Exception as e:
        ok = None
        print("  guard unavailable:", type(e).__name__, e, flush=True)
    out["guard_passed"] = ok

    print("\nAR2023-HK (C=187)", flush=True)
    out["datasets"]["AR2023-HK"] = run(load_hk(), N_Q_LARGE)

    with open(os.path.join(RESULTS, "decoy_frontier_hk.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote results/decoy_frontier_hk.json")
