"""Sorted versus random grouping on AR2023-HK (§5.5: 1.12x versus 1.22x traffic).
Output: results/codebook_order_ablation.json
"""
from _paths import RESULTS
import os, sys, json, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_amazon2023_hk as H
import location_padding2 as LP
from location_padding2 import to_bits, Codebook

SLOTS = 4096
KEYS_MIB, SCORES_MIB, CHUNK_KIB = 8.0, 23.4, 158.7


class OrderedCodebook(Codebook):
    def __init__(self, B, k, order):
        self._fixed = order
        super().__init__(B, k)

    def keys(self, B):
        self.order = self._fixed
        return super().keys(B)


def main():
    U, I, T, nU, nI = H.load_hk()
    key = np.unique(U * nI + I); U, I = key // nI, key % nI
    C = math.ceil(nI / SLOTS)
    off = np.searchsorted(U, np.arange(nU + 1))
    B = to_bits([np.unique(I[off[u]:off[u + 1]] // SLOTS) for u in range(nU)], C)
    traffic = lambda ch: KEYS_MIB + SCORES_MIB + ch * CHUNK_KIB / 1024
    base = float(np.mean(traffic(B.sum(1))))
    freq = B.sum(0)
    orders = {"rarest_first": np.argsort(freq, kind="stable"),
              "most_frequent_first": np.argsort(-freq, kind="stable"),
              "catalogue": np.arange(C),
              "random": np.random.default_rng(0).permutation(C)}
    out = {"note": __doc__.strip().split("\n\n")[0], "users": int(nU), "C": C,
           "sparse_mean_chunks": round(float(B.sum(1).mean()), 3), "rows": []}
    for k in (5, 10, 20):
        for name, order in orders.items():
            t0 = time.time()
            cb = OrderedCodebook(B, k, order)
            sent = cb.send(B)
            assert (sent >= 0).all()
            cls = sent
            sizes = cb.covers[sent].sum(1)
            cnt = np.bincount(cls)
            r = {"k": k, "order": name, "mean_cover_chunks": round(float(sizes.mean()), 3),
                 "added_chunks": round(float(sizes.mean() - B.sum(1).mean()), 3),
                 "traffic_vs_sparse": round(float(np.mean(traffic(sizes))) / base, 3),
                 "linkage": round(float(np.mean(1.0 / cnt[cls])), 4), "smallest_class": int(cnt[cnt > 0].min())}
            out["rows"].append(r); print(r, f"[{time.time()-t0:.0f}s]", flush=True)
        perm = np.random.default_rng(0).permutation(nU); ng = nU // k
        g = np.empty(nU, dtype=np.int64); g[perm] = np.minimum(np.arange(nU) // k, ng - 1)
        cov = np.zeros((ng, C), dtype=bool)
        np.logical_or.at(cov, g, B)
        sizes = cov[g].sum(1); cnt = np.bincount(g)
        r = {"k": k, "order": "unsorted_random_groups", "mean_cover_chunks": round(float(sizes.mean()), 3),
             "added_chunks": round(float(sizes.mean() - B.sum(1).mean()), 3),
             "traffic_vs_sparse": round(float(np.mean(traffic(sizes))) / base, 3),
             "linkage": round(float(np.mean(1.0 / cnt[g])), 4), "smallest_class": int(cnt.min())}
        out["rows"].append(r); print(r, flush=True)
        json.dump(out, open(os.path.join(RESULTS, "codebook_order_ablation.json"), "w"), indent=2)
    print("SAVED")


if __name__ == "__main__":
    main()
