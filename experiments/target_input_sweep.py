"""The 36 padding rules of App. C.2 on Gowalla and Amazon-Book.
Output: results/target_input_sweep.json
"""
from _paths import DATA, REPO, RESULTS
import sys, math, json, os
from collections import defaultdict
import numpy as np
sys.path.insert(0, f"{REPO}")
from src.dataset import RecDataset

SLOTS = 4096
KS = [1, 2, 4, 8, 15, 30, 60]
SEED = 42
DATASETS = [("Gowalla", "gowalla", 0.0345), ("Amazon-Book", "amazon-book", 0.5878)]


def reid(fps):
    g = defaultdict(int)
    for f in fps:
        g[f] += 1
    return float(np.mean([1.0 / g[f] for f in fps]))


def equal_frequency_groups(users, key, k):
    if k == 1:
        return {u: 0 for u in users}, 1
    vals = np.array([key[u] for u in users], dtype=float)
    qs = np.unique(np.quantile(vals, np.linspace(0, 1, k + 1)[1:-1]))
    gid = {u: int(np.searchsorted(qs, key[u], side="right")) for u in users}
    return gid, len(set(gid.values()))


def within_group_cv(work, gid):
    groups = defaultdict(list)
    for u, g in gid.items():
        groups[g].append(u)
    cvs = []
    for us in groups.values():
        if len(us) < 2:
            continue
        w = np.array([work[u] for u in us], dtype=float)
        if w.mean() > 0:
            cvs.append(w.std() / w.mean())
    return float(np.mean(cvs)) if cvs else 0.0


def arm(users, gid, n_groups, nact, blocks, floor, base_steps, fam, label, k, side):
    gmax = defaultdict(int)
    for u in users:
        gmax[gid[u]] = max(gmax[gid[u]], nact[u])
    padded = {u: gmax[gid[u]] for u in users}
    r = reid([(padded[u], blocks[u]) for u in users])
    steps = float(np.mean([padded[u] for u in users]))
    return {"family": fam, "side": side, "target_input": label,
            "k_requested": k, "n_groups": n_groups,
            "reid": round(r, 6), "delta_vs_floor": round(r - floor, 6),
            "at_floor": bool(abs(r - floor) < 1e-9),
            "below_floor": bool(r < floor - 1e-9),
            "within_group_cv_after": round(within_group_cv(padded, gid), 4),
            "mean_chunk_steps": round(steps, 3),
            "overhead": round(steps / base_steps, 3)}


def run(name, sub, published_floor):
    ds = RecDataset(f"{DATA}/{sub}", sub, rating_threshold=0)
    users = [u for u, it in ds.train_dict.items() if it]
    C = math.ceil(ds.n_items / SLOTS)
    blocks = {u: frozenset(i // SLOTS for i in ds.train_dict[u]) for u in users}
    nact = {u: len(blocks[u]) for u in users}
    deg = {u: len(ds.train_dict[u]) for u in users}
    maxix = {u: max(blocks[u]) for u in users}
    span = {u: max(blocks[u]) - min(blocks[u]) for u in users}
    rng = np.random.RandomState(SEED)
    rnd = {u: float(v) for u, v in zip(users, rng.rand(len(users)))}

    floor = reid([blocks[u] for u in users])
    base_steps = float(np.mean([nact[u] for u in users]))
    ok = abs(floor - published_floor) < 5e-4
    print("  %s  N=%d C=%d  blocks-only Re-ID %.4f (published %.4f) %s"
          % (name, len(users), C, floor, published_floor, "ok" if ok else "MISMATCH"),
          flush=True)

    FAMS = [("B_active_chunk", "inside", nact, "|I_active|"),
            ("B_max_index", "inside", maxix, "max(I_active)"),
            ("B_span", "inside", span, "max-min of I_active"),
            ("A_degree", "outside", deg, "degree d_u"),
            ("A_random", "outside", rnd, "a random per-user label")]

    arms = []
    for fam, side, key, label in FAMS:
        print("    %-16s [%s]  target reads %s" % (fam, side, label), flush=True)
        print("      %4s %8s %10s %11s %9s" % ("k", "groups", "Re-ID", "vs floor", "overhead"),
              flush=True)
        for k in KS:
            gid, n = equal_frequency_groups(users, key, k)
            a = arm(users, gid, n, nact, blocks, floor, base_steps, fam, label, k, side)
            arms.append(a)
            print("      %4d %8d %10.4f %+11.6f %8.3fx"
                  % (k, n, a["reid"], a["delta_vs_floor"], a["overhead"]), flush=True)

    bid, seen = {}, {}
    for u in users:
        seen.setdefault(blocks[u], len(seen))
        bid[u] = seen[blocks[u]]
    a = arm(users, bid, len(seen), nact, blocks, floor, base_steps,
            "B_exact", "the block set itself", None, "inside")
    arms.append(a)
    print("    B_exact          [inside]  one group per distinct block set: "
          "%d groups, Re-ID %.4f (%+.6f), %.3fx"
          % (a["n_groups"], a["reid"], a["delta_vs_floor"], a["overhead"]), flush=True)

    inside = [x for x in arms if x["side"] == "inside"]
    outside = [x for x in arms if x["side"] == "outside"]
    deg_arms = [x for x in arms if x["family"] == "A_degree"]
    rnd_arms = [x for x in arms if x["family"] == "A_random"]

    p1 = not any(x["below_floor"] for x in arms)
    p2 = all(x["at_floor"] for x in inside)
    print("    P1 no arm below the floor (%d arms): %s" % (len(arms), p1), flush=True)
    print("    P2 every INSIDE arm exactly at the floor (%d arms, %d statistics): %s"
          % (len(inside), len({x["family"] for x in inside}), p2), flush=True)
    print("    P3 degree max +%.4f; random max +%.6f"
          % (max(x["delta_vs_floor"] for x in deg_arms),
             max(x["delta_vs_floor"] for x in rnd_arms)), flush=True)
    print("       cheapest inside arm %.3fx vs cheapest degree arm %.3fx"
          % (min(x["overhead"] for x in inside), min(x["overhead"] for x in deg_arms)),
          flush=True)

    return {"users": len(users), "C": C,
            "blocks_only_reid": round(floor, 6),
            "published_blocks_only_reid": published_floor,
            "guard_floor_passed": bool(ok),
            "mean_chunk_steps_unpadded": round(base_steps, 3),
            "n_arms": len(arms),
            "n_inside_arms": len(inside), "n_outside_arms": len(outside),
            "n_inside_statistics": len({x["family"] for x in inside}),
            "no_arm_below_floor": bool(p1),
            "all_inside_at_floor": bool(p2),
            "degree_arms_above_floor": sum(1 for x in deg_arms if x["delta_vs_floor"] > 1e-9),
            "degree_max_delta": round(max(x["delta_vs_floor"] for x in deg_arms), 6),
            "random_max_delta": round(max(x["delta_vs_floor"] for x in rnd_arms), 6),
            "cheapest_inside_overhead": min(x["overhead"] for x in inside),
            "cheapest_degree_overhead": min(x["overhead"] for x in deg_arms),
            "arms": arms}


if __name__ == "__main__":
    out = {"slots": SLOTS, "k_sweep": KS, "seed": SEED,
           "shape": "partition users into groups; pad every member's chunk-step count "
                    "to the group's maximum active-chunk count",
           "baseline": "unpadded chunked trace, whose observable is the block set alone",
           "sides": {"inside": "the target is a function of the block set",
                     "outside": "the target reads something the block set does not determine"},
           "datasets": {}}
    for name, sub, g in DATASETS:
        out["datasets"][name] = run(name, sub, g)
    with open(os.path.join(RESULTS, "target_input_sweep.json"), "w") as f:
        json.dump(out, f, indent=1)
    print("\nwrote results/target_input_sweep.json")
