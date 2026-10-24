"""Fixed covers when later interactions are not recorded under the account (§5.5, App. C.4).
Output: results/cover_epochs_deviceonly_hk.json
"""
from _paths import RESULTS
import os, sys, json, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from location_padding2 import to_bits, Codebook, cover_bits
import dataset_amazon2023_hk as H
from cover_epochs_hk import classes_of, QS, UP_KIB, KEYS_MIB, SCORE_KIB

K, FS = 10, (0.1, 0.25, 0.5)


def main():
    t0 = time.time()
    U, I, T, nU, nI = H.load_hk(); C = math.ceil(nI / H.SLOTS); ch = I // H.SLOTS
    cuts = [int(np.quantile(T, q)) for q in QS]

    def lists(mask):
        key = np.unique(U[mask] * C + ch[mask]); uu = key // C; cc = key % C
        off = np.searchsorted(uu, np.arange(nU + 1))
        return [cc[off[u]:off[u + 1]].tolist() for u in range(nU)]

    full = [to_bits(lists(T <= c), C) for c in cuts]
    have = full[0].any(1); full = [b[have] for b in full]; n = len(full[0])
    hidx = np.flatnonzero(have); pos = np.full(nU, -1); pos[hidx] = np.arange(n)
    cb0 = Codebook(full[0], K); g0 = cb0.send(full[0])
    order = np.argsort(g0, kind="stable"); gs = g0[order]
    starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]]); sizes = np.diff(np.r_[starts, n])
    score_mib = C * SCORE_KIB / 1024
    out = {"note": __doc__.strip().split("\n\n")[0], "users": n, "k": K, "f": {}}
    rng = np.random.default_rng(0); rnd = rng.random(len(U))
    for f in FS:
        dev = (T > cuts[0]) & (rnd < f)
        covs, rec = [], {"deferred_all_pct": [], "deferred_device_only_pct": [], "traffic_vs_sparse": [], "mean_cover_chunks": []}
        for e in range(len(cuts)):
            op = to_bits(lists((T <= cuts[e]) & ~dev), C)[have]
            Ug = np.logical_or.reduceat(op[order], starts, axis=0)
            cov = np.empty_like(op); cov[order] = np.repeat(Ug, sizes, axis=0)
            covs.append(np.packbits(cov, axis=1))
            mm = (T <= cuts[e]) & have[U]
            inside = cov[pos[U[mm]], ch[mm]]
            rec["deferred_all_pct"].append(round(100 * float(np.mean(~inside)), 3))
            dm = dev[mm]
            rec["deferred_device_only_pct"].append(round(100 * float(np.mean(~inside[dm])), 3) if dm.any() else 0.0)
            s = cov.sum(1); own = full[e].sum(1)
            tr = (s * UP_KIB / 1024 + KEYS_MIB + score_mib).mean() / (own * UP_KIB / 1024 + KEYS_MIB + score_mib).mean()
            rec["traffic_vs_sparse"].append(round(float(tr), 3)); rec["mean_cover_chunks"].append(round(float(s.mean()), 2))
        rec["linkage_5_linked"] = round(float(np.mean(1.0 / classes_of(covs))), 5)
        rec["smallest_class"] = int(classes_of(covs).min())
        out["f"][str(f)] = rec; print(f, rec, f"[{time.time()-t0:.0f}s]", flush=True)
        json.dump(out, open(os.path.join(RESULTS, "cover_epochs_deviceonly_hk.json"), "w"), indent=2)
    print("SAVED")


if __name__ == "__main__":
    main()
