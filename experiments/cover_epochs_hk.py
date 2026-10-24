"""Five linked codebooks on AR2023-HK: regrouped, fixed and never-refreshed covers (Table 5; §5.5; App. C.4).
Output: results/cover_epochs_hk.json
"""
from _paths import RESULTS
import os, sys, json, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from location_padding2 import to_bits, Codebook, cover_bits
import dataset_amazon2023_hk as H

KS, QS = (5, 10, 20), (0.900, 0.925, 0.950, 0.975, 1.0)
UP_KIB, KEYS_MIB, SCORE_KIB = 326.5, 8.46, 128.1


def classes_of(cols):
    X = np.concatenate([c.reshape(len(c), -1).view(np.uint8) for c in cols], axis=1)
    _, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    return cnt[inv.ravel()]


def main():
    t0 = time.time()
    U, I, T, nU, nI = H.load_hk(); C = math.ceil(nI / H.SLOTS); ch = I // H.SLOTS
    cuts = [int(np.quantile(T, q)) for q in QS]

    def lists(mask):
        key = np.unique(U[mask] * C + ch[mask]); uu = key // C; cc = key % C
        off = np.searchsorted(uu, np.arange(nU + 1))
        return [cc[off[u]:off[u + 1]].tolist() for u in range(nU)]

    B = [to_bits(lists(T <= c), C) for c in cuts]
    have = B[0].any(1); B = [b[have] for b in B]; n = len(B[0]); E = len(B)
    own = [b.sum(1).astype(np.float64) for b in B]
    hidx = np.flatnonzero(have); pos = np.full(nU, -1); pos[hidx] = np.arange(n)
    score_mib = C * SCORE_KIB / 1024

    def traffic_ratio(chunks, e):
        tr = chunks * UP_KIB / 1024 + KEYS_MIB + score_mib
        sp = own[e] * UP_KIB / 1024 + KEYS_MIB + score_mib
        return round(float(tr.mean() / sp.mean()), 3)

    out = {"note": __doc__.strip().split("\n\n")[0], "users": n, "C": C, "snapshot_quantiles": QS, "k": {}}
    print(f"loaded n={n} [{time.time()-t0:.0f}s]", flush=True)
    for k in KS:
        res = {}
        cb0 = Codebook(B[0], k); g0 = cb0.send(B[0])
        covs = []
        for e in range(E):
            cb = cb0 if e == 0 else Codebook(B[e], k)
            covs.append(np.packbits(cover_bits(cb, cb.send(B[e])), axis=1))
        rec = {"linkage_R": [round(float(np.mean(1.0 / classes_of(covs[E - R:]))), 5) for R in range(1, E + 1)],
               "upload_vs_own": [round(float(np.unpackbits(covs[e], axis=1)[:, :C].sum(1).mean() / own[e].mean()), 6) for e in range(E)]}
        rec["traffic_vs_sparse"] = [traffic_ratio(np.unpackbits(covs[e], axis=1)[:, :C].sum(1), e) for e in range(E)]
        res["regroup"] = rec; print(k, "regroup", rec, f"[{time.time()-t0:.0f}s]", flush=True)
        order = np.argsort(g0, kind="stable"); gs = g0[order]
        starts = np.flatnonzero(np.r_[True, gs[1:] != gs[:-1]])
        pcovs, sizes = [], []
        for e in range(E):
            U_g = np.logical_or.reduceat(B[e][order], starts, axis=0)
            cov = np.empty_like(B[e]); cov[order] = np.repeat(U_g, np.diff(np.r_[starts, n]), axis=0)
            pcovs.append(np.packbits(cov, axis=1)); sizes.append(cov.sum(1))
        rec = {"linkage_R": [round(float(np.mean(1.0 / classes_of(pcovs[E - R:]))), 5) for R in range(1, E + 1)],
               "smallest_class_all_R": int(classes_of(pcovs).min()),
               "upload_vs_own": [round(float(sizes[e].mean() / own[e].mean()), 6) for e in range(E)],
               "traffic_vs_sparse": [traffic_ratio(sizes[e], e) for e in range(E)],
               "mean_cover_chunks": [round(float(sizes[e].mean()), 5) for e in range(E)]}
        res["persistent"] = rec; print(k, "persistent", rec, f"[{time.time()-t0:.0f}s]", flush=True)
        c0 = cover_bits(cb0, g0); s0 = c0.sum(1)
        m = T <= cuts[-1]
        defer = []
        for e in range(E):
            mm = (T <= cuts[e]) & have[U]
            r = pos[U[mm]]; inside = c0[r, ch[mm]]
            defer.append(round(100 * float(np.mean(~inside)), 3))
        cs = classes_of([np.packbits(c0, axis=1)])
        rec = {"linkage_any_R": round(float(np.mean(1.0 / cs)), 5),
               "upload_vs_own": [round(float(s0.mean() / own[e].mean()), 6) for e in range(E)],
               "traffic_vs_sparse": [traffic_ratio(s0, e) for e in range(E)],
               "interactions_deferred_pct": defer}
        res["static"] = rec; print(k, "static", rec, f"[{time.time()-t0:.0f}s]", flush=True)
        res["dense"] = {"linkage_any_R": round(1.0 / n, 8), "upload_vs_own": [round(float(C / own[e].mean()), 3) for e in range(E)],
                        "traffic_vs_sparse": [traffic_ratio(np.full(n, C), e) for e in range(E)]}
        out["k"][str(k)] = res
        json.dump(out, open(os.path.join(RESULTS, "cover_epochs_hk.json"), "w"), indent=2)
    print("SAVED")


if __name__ == "__main__":
    main()
