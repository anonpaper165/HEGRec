"""Item-to-chunk layout on the benchmarks (Table 3 item-layout row; App. C.1 Amazon-Book).
Output: results/layout_table.json
"""
from _paths import DATA, REPO, RESULTS
import json, math, os, numpy as np
from collections import defaultdict

SLOTS = 4096
DATA = f"{DATA}"

def load_user_items(path):
    users = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) < 2:
                continue
            users.append([int(x) for x in p[1:]])
    return users

def metrics(user_items, item_to_chunk):
    groups = defaultdict(int)
    fps = []
    for items in user_items:
        fp = frozenset(item_to_chunk[i] for i in items)
        fps.append(fp)
        groups[fp] += 1
    n = len(fps)
    uniq = sum(1 for fp in fps if groups[fp] == 1) / n * 100
    anon = np.mean([groups[fp] for fp in fps])
    reid = np.mean([1.0 / groups[fp] for fp in fps]) * 100
    return uniq, anon, reid

def layout_original(n_items):
    return np.arange(n_items) // SLOTS

def layout_random(n_items, rng):
    perm = rng.permutation(n_items)
    chunk = np.empty(n_items, dtype=int)
    chunk[perm] = np.arange(n_items) // SLOTS
    return chunk

def layout_popularity(n_items, user_items):
    pop = np.zeros(n_items, dtype=int)
    for items in user_items:
        for i in items:
            pop[i] += 1
    order = np.argsort(-pop)
    chunk = np.empty(n_items, dtype=int)
    chunk[order] = np.arange(n_items) // SLOTS
    return chunk

OUT = {}


def run(name, sub, n_items, n_seeds=10, paper_original=None):
    ui = load_user_items(f"{DATA}/{sub}/train.txt")
    C = math.ceil(n_items / SLOTS)
    print(f"\n=== {name} (n_items={n_items}, C={C}, users={len(ui)}) ===")
    rec = OUT.setdefault(name, {"n_items": n_items, "C": C, "users": len(ui)})
    u,a,r = metrics(ui, layout_original(n_items))
    rec["original"] = {"unique_pct": round(u, 2), "anon_set": round(a, 1),
                       "reid": round(r / 100.0, 4)}
    print(f"  Original    : Unique%={u:6.2f}  Anon={a:8.1f}  Re-ID%={r:6.2f}")
    if paper_original is not None:
        pu, pr = paper_original
        assert abs(u - pu) < 0.5 and abs(r - pr) < 0.5, \
            f"Original layout does not reproduce paper ({u:.2f}/{r:.2f} vs {pu}/{pr})"
        print(f"    [validated] reproduces paper Original (Unique% {pu}, Re-ID% {pr})")
    u,a,r = metrics(ui, layout_popularity(n_items, ui))
    rec["popularity"] = {"unique_pct": round(u, 2), "anon_set": round(a, 1),
                         "reid": round(r / 100.0, 4)}
    print(f"  Popularity  : Unique%={u:6.2f}  Anon={a:8.1f}  Re-ID%={r:6.2f}")
    U,A,R = [],[],[]
    for s in range(n_seeds):
        rng = np.random.RandomState(42 + s)
        u,a,r = metrics(ui, layout_random(n_items, rng))
        U.append(u); A.append(a); R.append(r)
    rec["random"] = {"seeds": n_seeds,
                     "unique_pct": round(float(np.mean(U)), 2),
                     "anon_set": round(float(np.mean(A)), 1),
                     "reid": round(float(np.mean(R)) / 100.0, 4),
                     "reid_sd": round(float(np.std(R)) / 100.0, 4)}
    print(f"  Random (n={n_seeds}): "
          f"Unique%={np.mean(U):.2f}+/-{np.std(U):.2f}  "
          f"Anon={np.mean(A):.1f}+/-{np.std(A):.1f}  "
          f"Re-ID%={np.mean(R):.2f}+/-{np.std(R):.2f}")

if __name__ == "__main__":
    run("Gowalla", "gowalla", 40981, paper_original=(0.3, 3.5))
    run("Amazon-Book", "amazon-book", 91599, paper_original=(50.1, 58.8))
    with open(os.path.join(RESULTS, "layout_table.json"), "w") as f:
        json.dump({"seeding": "RandomState(42 + s), so the run is reproducible",
                   "datasets": OUT}, f, indent=1)
    print("\nwrote results/layout_table.json")
