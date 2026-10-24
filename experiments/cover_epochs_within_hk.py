"""Interactions deferred within an update period on AR2023-HK (§5.5, 1.0-1.5% at k=10).
Output: results/cover_epochs_within_hk.json
"""
from _paths import RESULTS
import os, sys, json, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from location_padding2 import to_bits, Codebook
import dataset_amazon2023_hk as H
from cover_epochs_hk import QS


def main():
    t0 = time.time()
    U, I, T, nU, nI = H.load_hk(); C = math.ceil(nI / H.SLOTS); ch = I // H.SLOTS
    cuts = [int(np.quantile(T, q)) for q in QS]

    def lists(mask):
        key = np.unique(U[mask] * C + ch[mask]); uu = key // C; cc = key % C
        off = np.searchsorted(uu, np.arange(nU + 1))
        return [cc[off[u]:off[u + 1]].tolist() for u in range(nU)]

    B = [to_bits(lists(T <= c), C) for c in cuts]
    have = B[0].any(1); B = [b[have] for b in B]; n = len(B[0])
    hidx = np.flatnonzero(have); pos = np.full(nU, -1); pos[hidx] = np.arange(n)
    out = {"note": __doc__.strip().split("\n\n")[0], "users": n, "k": {}}
    for k in (5, 10, 20):
        g0 = Codebook(B[0], k).send(B[0])
        order = np.argsort(g0, kind="stable"); gs = g0[order]
        starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]]); sizes = np.diff(np.r_[starts, n])
        rec = {"interactions_outside_previous_cover_pct": [], "users_with_any_left_out_pct": [], "mean_history_left_out_pct": []}
        for e in range(1, len(cuts)):
            Ug = np.logical_or.reduceat(B[e - 1][order], starts, axis=0)
            cov = np.empty_like(B[e - 1]); cov[order] = np.repeat(Ug, sizes, axis=0)
            mm = (T <= cuts[e]) & have[U]; r = pos[U[mm]]
            out_ = ~cov[r, ch[mm]]
            rec["interactions_outside_previous_cover_pct"].append(round(100 * float(out_.mean()), 3))
            per_u_out = np.bincount(r, weights=out_, minlength=n); per_u = np.bincount(r, minlength=n)
            rec["users_with_any_left_out_pct"].append(round(100 * float((per_u_out > 0).mean()), 2))
            rec["mean_history_left_out_pct"].append(round(100 * float(np.mean(per_u_out / np.maximum(per_u, 1))), 3))
        out["k"][str(k)] = rec; print(k, rec, f"[{time.time()-t0:.0f}s]", flush=True)
        json.dump(out, open(os.path.join(RESULTS, "cover_epochs_within_hk.json"), "w"), indent=2)
    print("SAVED")


if __name__ == "__main__":
    main()
