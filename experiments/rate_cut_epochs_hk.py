"""Rate-aware grouping over five codebooks (§5.5: 0.103 by the fifth).
Output: results/rate_cut_epochs_hk.json
"""
from _paths import RESULTS
import os, sys, json, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from location_padding2 import to_bits, Codebook
from dormant_prior_robust import RateCodebook
import dataset_amazon2023_hk as H

KS, QS = (5, 10, 20), (0.875, 0.900, 0.925, 0.950, 0.975, 1.0)
UP_KIB, KEYS_MIB, SCORE_KIB = 158.7, 8.0, 128.1
OUT = os.path.join(RESULTS, "rate_cut_epochs_hk.json")


def success(g, pi):
    s = np.bincount(g, weights=pi); mx = np.zeros(g.max() + 1); np.maximum.at(mx, g, pi)
    return round(float(mx.sum() / s.sum()), 4) if s.sum() > 0 else None


def main():
    t0 = time.time()
    U, I, T, nU, nI = H.load_hk(); C = math.ceil(nI / H.SLOTS); ch = I // H.SLOTS
    cuts = [int(np.quantile(T, q)) for q in QS]

    def lists(mask):
        key = np.unique(U[mask] * C + ch[mask]); uu = key // C; cc = key % C
        off = np.searchsorted(uu, np.arange(nU + 1))
        return [cc[off[u]:off[u + 1]].tolist() for u in range(nU)]

    B = [to_bits(lists(T <= c), C) for c in cuts[1:]]
    have = B[0].any(1); B = [b[have] for b in B]; n = len(B[0]); E = len(B)
    hidx = np.flatnonzero(have)
    cnt = [np.bincount(U[T <= c], minlength=nU)[hidx].astype(float) for c in cuts]
    cum = cnt[1:]; recent = [cnt[e + 1] - cnt[e] for e in range(E)]
    own = [b.sum(1).astype(float) for b in B]; score_mib = C * SCORE_KIB / 1024
    tr = lambda chunks, e: round(float((chunks * UP_KIB / 1024 + KEYS_MIB + score_mib).mean() / (own[e] * UP_KIB / 1024 + KEYS_MIB + score_mib).mean()), 3)
    out = {"note": __doc__.strip().split("\n\n")[0], "users": n, "snapshot_quantiles": QS[1:], "k": {}}
    print(f"loaded n={n} [{time.time()-t0:.0f}s]", flush=True)
    for k in KS:
        res = {}
        for name, cb in (("base", Codebook(B[0], k)), ("rate_cut", RateCodebook(B[0], k, cum[0] / cum[0].sum()))):
            g = cb.send(B[0]); assert (g >= 0).all()
            order = np.argsort(g, kind="stable"); gs = g[order]; starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
            gid = np.empty(n, np.int64); gid[order] = np.repeat(np.arange(len(starts)), np.diff(np.r_[starts, n]))
            rec = {"success_count": [], "success_recent": [], "equal_linkage": [], "traffic_vs_sparse": []}
            size = np.bincount(gid)
            for e in range(E):
                U_g = np.logical_or.reduceat(B[e][order], starts, axis=0)
                rec["success_count"].append(success(gid, cum[e])); rec["success_recent"].append(success(gid, recent[e]))
                rec["equal_linkage"].append(round(float(np.mean(1.0 / size[gid])), 4))
                rec["traffic_vs_sparse"].append(tr(U_g.sum(1)[gid], e))
            res[name] = rec; print(k, name, rec, f"[{time.time()-t0:.0f}s]", flush=True)
        out["k"][str(k)] = res; json.dump(out, open(OUT, "w"), indent=2)
    print("SAVED")


if __name__ == "__main__":
    main()
