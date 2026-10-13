"""Degree-keyed and chunk-keyed padding under the chunked schedule (Table 3, Sec. 4.1).

What this executes. The client submits its real active chunks (the chunk set is unchanged).
The server then performs chunk-aggregation steps up to the MAXIMUM ACTIVE-CHUNK COUNT
among users in the requester's degree bucket, filling the difference with dummy steps
over all-zero chunks. Cost is therefore per chunk step, which is what chunked
aggregation charges for.

  before:  workload = r_S * |I_active|            trace = (workload, blocks) == blocks
  after :  workload = r_S * maxactive(bucket(u))  trace = (bucket, blocks)

The target objective is within-bucket workload indistinguishability, so we report the
coefficient of variation of per-user chunk-step counts inside each bucket, before and
after, alongside linkage and the compute overhead the dummy steps cost.
"""
from _paths import DATA, HOMO_GCN, REPO, RESULTS  # repo-relative roots
import sys, math, json
from collections import defaultdict
import numpy as np
sys.path.insert(0, f"{HOMO_GCN}")
from src.dataset import RecDataset
from src.bucketing import DegreeBucketing

SLOTS = 4096

def reid(fps):
    g = defaultdict(int)
    for f in fps: g[f] += 1
    return float(np.mean([1.0 / g[f] for f in fps]))

def within_bucket_cv(work, buckets):
    """mean over buckets of (sd/mean) of per-user workload inside the bucket"""
    cvs = []
    for us in buckets.values():
        if len(us) < 2: continue
        w = np.array([work[u] for u in us], dtype=float)
        m = w.mean()
        if m > 0: cvs.append(w.std() / m)
    return float(np.mean(cvs))

out = {"slots": SLOTS,
       "protocol": "client submits real active chunks; server pads chunk-aggregation "
                   "steps to the max active-chunk count within the requester's degree bucket",
       "bucketing": "degree bucketing adaptive, privacy_weight=10 (degree buckets)",
       "datasets": {}}

for name, sub in (("Gowalla", "gowalla"), ("Amazon-Book", "amazon-book")):
    ds = RecDataset(f"{DATA}/{sub}", sub, rating_threshold=0)
    users = [u for u, it in ds.train_dict.items() if it]
    blocks = {u: frozenset(i // SLOTS for i in ds.train_dict[u]) for u in users}
    nact   = {u: len(blocks[u]) for u in users}

    op = DegreeBucketing(ds, bucketing="adaptive", privacy_weight=10)
    bucket  = {u: b for b, us in op.user_buckets.items() for u in us}
    buckets = {b: [u for u in us if u in nact] for b, us in op.user_buckets.items()}
    buckets = {b: us for b, us in buckets.items() if us}

    # server pads chunk steps to the bucket's maximum active-chunk count
    tgt = {b: max(nact[u] for u in us) for b, us in buckets.items()}
    padded = {u: tgt[bucket[u]] for u in users}

    # --- the same objective, a different INPUT to the target rule ----------
    # Sec. 7 argues the reversal is not "a bad padding rule" but a rule whose
    # target reads something the block set does not already determine. The
    # control is a rule with the same shape whose input IS determined by the
    # block set: group users by their active-chunk count into the same number
    # of groups (equal-frequency boundaries over n_act) and pad to the group's
    # maximum. Both rules drive within-group workload variation to zero; only
    # one of them can add an observable, because padded_a = f(n_act) = f(blocks).
    qs = np.quantile(np.array([nact[u] for u in users], dtype=float),
                     np.linspace(0, 1, len(buckets) + 1)[1:-1])
    gid = {u: int(np.searchsorted(qs, nact[u], side="right")) for u in users}
    groups = defaultdict(list)
    for u in users:
        groups[gid[u]].append(u)
    tgt_a = {g: max(nact[u] for u in us) for g, us in groups.items()}
    padded_a = {u: tgt_a[gid[u]] for u in users}
    # a constant target, the degenerate case of the same argument
    padded_c = {u: math.ceil(ds.n_items / SLOTS) for u in users}

    rec = {
        "users": len(users), "C": math.ceil(ds.n_items / SLOTS),
        "n_buckets": len(buckets),
        # Sec. 2.3 says the padded STEP COUNT is coarser than the bucket itself,
        # because several buckets can share one max active-chunk count. That
        # number was quoted in the text with no producer until this line.
        "n_distinct_step_targets": len(set(tgt.values())),
        # --- the objective the intervention is for ---
        "within_bucket_workload_cv_before": round(within_bucket_cv(nact, buckets), 4),
        "within_bucket_workload_cv_after": round(within_bucket_cv(padded, buckets), 4),
        # --- overall linkage, same graph and chunking throughout ---
        "linkage_chunked":          round(reid([(nact[u], blocks[u]) for u in users]), 4),
        "linkage_chunked_bucketed": round(reid([(padded[u], blocks[u]) for u in users]), 4),
        "linkage_blocks_only":      round(reid([blocks[u] for u in users]), 4),
        # --- control: target read off the active-chunk count, not the degree ---
        "n_groups_active_chunk":    len(groups),
        "within_group_cv_before_active": round(within_bucket_cv(nact, groups), 4),
        "within_group_cv_after_active":  round(within_bucket_cv(padded_a, groups), 4),
        "linkage_active_chunk_target": round(reid([(padded_a[u], blocks[u])
                                                   for u in users]), 4),
        "linkage_constant_target":     round(reid([(padded_c[u], blocks[u])
                                                   for u in users]), 4),
        "mean_chunk_steps_active_target": round(float(np.mean([padded_a[u]
                                                              for u in users])), 3),
        # --- what the dummy steps cost ---
        "mean_chunk_steps_before": round(float(np.mean([nact[u] for u in users])), 3),
        "mean_chunk_steps_after":  round(float(np.mean([padded[u] for u in users])), 3),
    }
    rec["chunk_step_overhead"] = round(rec["mean_chunk_steps_after"] /
                                       rec["mean_chunk_steps_before"], 3)
    rec["chunk_step_overhead_active_target"] = round(
        rec["mean_chunk_steps_active_target"] / rec["mean_chunk_steps_before"], 3)
    out["datasets"][name] = rec
    print(name, json.dumps(rec, indent=1), flush=True)

json.dump(out, open(f"{RESULTS}/chunked_bucketed.json", "w"), indent=1)
print("\nwrote results/chunked_bucketed.json")
