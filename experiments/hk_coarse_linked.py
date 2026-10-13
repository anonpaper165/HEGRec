"""
Coarser packing against linked requests (AR2023-HK, current profiles).

For CKKS slot counts 4,096 (the paper's), 8,192, 16,384 and 32,768 (ring degrees 8192 to 65536), the chunk set of
each user's history at the three dates of unified_comparison_hk_v2.py (a year before the end, four months later,
the end), and the linkage of the last R = 1, 2, 3 linked sparse requests over all users (deterministic traces:
1 / class size, Observation 1). Ring degree 65536 is beyond the largest that Microsoft SEAL supports at the
128-bit level (32768), so its row is a counterfactual.
Output: results/hk_coarse_linked.json
"""
from _paths import RESULTS
import os, sys, json, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from location_padding2 import to_bits
import dataset_amazon2023_hk as H


def seq_classes(cols):
    X = np.concatenate([c.reshape(len(c), -1).view(np.uint8) for c in cols], axis=1)
    _, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    return cnt[inv.ravel()]


def main():
    U, I, T, nU, nI = H.load_hk()
    cut10 = int(np.quantile(T, 0.90)); cut5 = int(np.quantile(T, 0.95))
    out = {"note": __doc__.strip().split("\n\n")[0], "rows": []}
    for slots in (4096, 8192, 16384, 32768):
        t0 = time.time(); C = math.ceil(nI / slots); ch = I // slots
        def lists(mask):
            key = np.unique(U[mask] * C + ch[mask]); uu = key // C; cc = key % C
            off = np.searchsorted(uu, np.arange(nU + 1))
            return [cc[off[u]:off[u + 1]].tolist() for u in range(nU)]
        B = [to_bits(lists(T <= c), C) for c in (cut10, cut5, int(T.max()))]
        have = B[0].any(1); B = [b[have] for b in B]
        P = [np.packbits(b, axis=1) for b in B]
        row = {"slots": slots, "n_poly": 2 * slots, "C": C, "users": int(have.sum()),
               "mean_chunks_end": round(float(B[2].sum(1).mean()), 3)}
        for R in (1, 2, 3):
            cs = seq_classes(P[3 - R:]); row[f"R{R}"] = round(float(np.mean(1.0 / cs)), 6)
        out["rows"].append(row); print(row, f"[{time.time()-t0:.0f}s]", flush=True)
        json.dump(out, open(os.path.join(RESULTS, "hk_coarse_linked.json"), "w"), indent=2)
    print("SAVED")


if __name__ == "__main__":
    main()
