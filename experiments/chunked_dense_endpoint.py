"""Per-item, chunked and dense linkage on Gowalla and Amazon-Book (Table 3 rows 1-2; App. A.3).
Output: results/chunked_dense_endpoint.json
"""
from _paths import DATA, REPO, RESULTS
import sys, math, json, os
from collections import defaultdict
import numpy as np
sys.path.insert(0, f"{REPO}")
from src.dataset import RecDataset
from src.bucketing import DegreeBucketing

SLOTS = 4096
DATASETS = [("Gowalla", "gowalla", 0.3483, 0.0345, 0.0004),
            ("Amazon-Book", "amazon-book", 0.8962, 0.5878, 0.0003)]


def reid(fps):
    g = defaultdict(int)
    for f in fps:
        g[f] += 1
    return float(np.mean([1.0 / g[f] for f in fps]))


def run(name, sub, g_peritem, g_chunked, g_dense):
    ds = RecDataset(f"{DATA}/{sub}", sub, rating_threshold=0)
    users = [u for u, it in ds.train_dict.items() if it]
    C = math.ceil(ds.n_items / SLOTS)
    blocks = {u: frozenset(i // SLOTS for i in ds.train_dict[u]) for u in users}
    deg = {u: len(ds.train_dict[u]) for u in users}
    op = DegreeBucketing(ds, bucketing="adaptive", privacy_weight=10)
    bucket = {u: b for b, us in op.user_buckets.items() for u in us}

    mean_active = float(np.mean([len(blocks[u]) for u in users]))
    N = len(users)

    rows = {
        "per_item_sparse": reid([(deg[u], blocks[u]) for u in users]),
        "chunked_sparse": reid([(len(blocks[u]), blocks[u]) for u in users]),
        "per_item_bucket_dense": reid([bucket[u] for u in users]),
        "chunked_dense": reid([(C, frozenset(range(C))) for u in users]),
    }
    cost = round(C / mean_active, 3)
    guards = {
        "per_item_sparse": (g_peritem, rows["per_item_sparse"]),
        "chunked_sparse": (g_chunked, rows["chunked_sparse"]),
        "per_item_bucket_dense": (g_dense, rows["per_item_bucket_dense"]),
    }
    ok = all(abs(a - b) <= 6e-4 for a, b in guards.values())
    exact_floor = abs(rows["chunked_dense"] - 1.0 / N) < 1e-12

    print("  %s: N=%d C=%d mean_active=%.2f" % (name, N, C, mean_active), flush=True)
    for k, v in rows.items():
        extra = ""
        if k in guards:
            extra = "  (published %.4f)" % guards[k][0]
        if k == "chunked_dense":
            extra = "  (1/N = %.3e, exact floor: %s)" % (1.0 / N, exact_floor)
        print("    %-24s %.6f%s" % (k, v, extra), flush=True)
    print("    cost of chunked dense: %.3fx sparse upload AND %.3fx chunk-steps"
          % (cost, cost), flush=True)
    print("    guards: %s" % ("ok" if ok else "MISMATCH"), flush=True)

    return {"users": N, "C": C, "mean_active_chunks": round(mean_active, 3),
            "chance_1_over_N": 1.0 / N,
            "reid": {k: round(v, 8) for k, v in rows.items()},
            "chunked_dense_upload_vs_sparse": cost,
            "chunked_dense_chunksteps_vs_sparse": cost,
            "chunked_dense_is_exact_1_over_N": bool(exact_floor),
            "published": {k: g for k, (g, _) in guards.items()},
            "guards_passed": bool(ok)}


if __name__ == "__main__":
    out = {"slots": SLOTS, "bucketing": "degree bucketing adaptive, privacy_weight=10",
           "note": "same graph, chunking and bucket assignment for every row; only "
                   "the observable trace differs. Cost is C/mean-active-chunks, "
                   "which is both the upload and the chunk-step ratio under the "
                   "chunked schedule.",
           "datasets": {}}
    for name, sub, a, b, c in DATASETS:
        out["datasets"][name] = run(name, sub, a, b, c)
    with open(os.path.join(RESULTS, "chunked_dense_endpoint.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote results/chunked_dense_endpoint.json")
