"""
Chunk bundling against linked requests (AR2023-HK, current profiles).

The alternative to a larger ring: keep the ring-8192 chunks (4,096 items each) but group b consecutive chunks into a
bundle and send every chunk of a bundle that holds an interaction. The server then sees the bundle set, the same
partition a b-times larger chunk gives, so linkage equals coarser packing's; traffic is the base chunks sent.
Per bundle size b = 1, 2, 4, 8, 16, 32 and the three dates of unified_comparison_hk_v2.py, over all users with a
year-old history: mean linkage of the last R = 1, 2, 3 linked requests (1 / class size), the share of users
whose class has fewer than 10 members (no k-anonymity floor), and LightGCN traffic relative to sparse at the end
date from the measured ring-8192 sizes (8.01 MiB keys, 158.6 KiB per seeded chunk, 23.4 MiB scores).
Output: results/hk_bundle_linked.json
"""
from _paths import RESULTS
import os, sys, json, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from location_padding2 import to_bits
import dataset_amazon2023_hk as H
KEYS, CHUNK, SCORES = 8.01, 158.6 / 1024, 23.4


def seq_classes(cols):
    X = np.concatenate([c.reshape(len(c), -1).view(np.uint8) for c in cols], axis=1)
    _, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    return cnt[inv.ravel()]


def main():
    U, I, T, nU, nI = H.load_hk(); C = math.ceil(nI / 4096); ch = I // 4096
    cut10 = int(np.quantile(T, 0.90)); cut5 = int(np.quantile(T, 0.95))
    def lists(mask):
        key = np.unique(U[mask] * C + ch[mask]); uu = key // C; cc = key % C
        off = np.searchsorted(uu, np.arange(nU + 1))
        return [cc[off[u]:off[u + 1]].tolist() for u in range(nU)]
    B = [to_bits(lists(T <= c), C) for c in (cut10, cut5, int(T.max()))]
    have = B[0].any(1); B = [b[have] for b in B]
    base = KEYS + B[2].sum(1).mean() * CHUNK + SCORES
    out = {"note": __doc__.strip().split("\n\n")[0], "sparse_MiB": round(float(base), 2), "rows": []}
    for b in (1, 2, 4, 8, 16, 32):
        t0 = time.time(); nb = math.ceil(C / b)
        sizes = np.array([min(b, C - j * b) for j in range(nb)])
        Bb = [np.stack([x[:, j * b:(j + 1) * b].any(1) for j in range(nb)], 1) for x in B]
        P = [np.packbits(x, axis=1) for x in Bb]
        sent = (Bb[2] * sizes[None, :]).sum(1)
        row = {"bundle": b, "bundles": nb, "mean_bundles_end": round(float(Bb[2].sum(1).mean()), 3),
               "traffic_vs_sparse": round(float((KEYS + sent.mean() * CHUNK + SCORES) / base), 3)}
        for R in (1, 2, 3):
            cs = seq_classes(P[3 - R:])
            row[f"R{R}"] = round(float(np.mean(1.0 / cs)), 6); row[f"R{R}_share_class_lt10"] = round(float(np.mean(cs < 10)), 4)
            row[f"R{R}_share_unique"] = round(float(np.mean(cs == 1)), 4)
        out["rows"].append(row); print(row, f"[{time.time()-t0:.0f}s]", flush=True)
        json.dump(out, open(os.path.join(RESULTS, "hk_bundle_linked.json"), "w"), indent=2)
    print("SAVED")


if __name__ == "__main__":
    main()
