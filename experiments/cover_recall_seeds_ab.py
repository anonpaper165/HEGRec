"""
The fixed cover's recall cost over five random splits (Amazon-Book, Gowalla; k = 5, 10, 20).

cover_substitution_recall_ab.py prices the fixed cover on one random split (seed 0): each training
history is split into an epoch-start 80% and a post-codebook 20%, the codebook is built from the
epoch-start chunk sets, and the request carries the full history less the items outside the cover.
This repeats that for split seeds 0..4 with the same model and serving operator (e2e_seal_request.load),
reporting per seed and k the relative change in Recall@20 over all users with a test item, its mean and
s.d. over seeds, and the share of interactions left out.
Output: results/cover_recall_seeds_ab.json
"""
from _paths import RESULTS
import os, sys, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e2e_seal_request import load
from location_padding2 import to_bits, Codebook, cover_bits

SLOTS, K, KS, SPLIT, SEEDS = 4096, 20, (5, 10, 20), 0.8, range(5)


def recall(ebar_s, ehat, sups, full, test, idx):
    out = []
    for b0 in range(0, len(idx), 2048):
        js = idx[b0:b0 + 2048]
        UV = np.stack([(ebar_s[sups[j]] / math.sqrt(len(sups[j]))).sum(0) if len(sups[j]) else np.zeros(ehat.shape[1]) for j in js])
        SC = UV @ ehat.T
        for r, j in enumerate(js):
            sc = SC[r]; sc[full[j]] = -np.inf; top = np.argpartition(-sc, K)[:K]
            out.append(np.isin(top, test[j]).sum() / min(len(test[j]), K))
    return np.array(out)


def main():
    res = {"note": __doc__.strip().split("\n\n")[0], "datasets": {}}
    for ds in ("Amazon-Book", "Gowalla"):
        ebar_s, ehat, hist, test, B, C, nI = load(ds)
        idx = [j for j in range(len(hist)) if len(test[j])]
        base = recall(ebar_s, ehat, hist, hist, test, idx); r = {"full_recall": round(float(base.mean()), 5), "k": {}}
        for k in KS:
            rel, drop = [], []
            for seed in SEEDS:
                rng = np.random.default_rng(seed); start = []
                for s in hist:
                    keep = rng.random(len(s)) < SPLIT
                    if not keep.any(): keep[rng.integers(len(s))] = True
                    start.append(s[keep])
                Bs = to_bits([sorted(set((x // SLOTS).tolist())) for x in start], C)
                cb = Codebook(Bs, k); cov = cover_bits(cb, cb.send(Bs))
                sups = [h[cov[j][h // SLOTS]] for j, h in enumerate(hist)]
                rr = recall(ebar_s, ehat, sups, hist, test, idx)
                rel.append(100 * (rr.mean() / base.mean() - 1))
                drop.append(100 * sum(len(h) - len(x) for h, x in zip(hist, sups)) / sum(len(h) for h in hist))
                print(ds, k, seed, round(rel[-1], 3), round(drop[-1], 3), flush=True)
            r["k"][str(k)] = {"rel_change_pct": [round(x, 3) for x in rel], "mean": round(float(np.mean(rel)), 3),
                              "sd": round(float(np.std(rel)), 3), "dropped_pct_mean": round(float(np.mean(drop)), 3)}
        res["datasets"][ds] = r
        json.dump(res, open(os.path.join(RESULTS, "cover_recall_seeds_ab.json"), "w"), indent=2)
    print("SAVED")


if __name__ == "__main__":
    main()
