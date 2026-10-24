"""Three linked requests against current profiles on AR2023-HK (Table 4 current-profile columns; Fig. 3a).
Run with HEGREC_NQ=10000. Output: results/unified_comparison_hk_v2.json
"""
from _paths import RESULTS
import os, sys, json, math, time
from math import lgamma
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from location_padding2 import to_bits, Codebook, cover_bits
import dataset_amazon2023_hk as H

K_DECOY, K_COVER, SEED = 32, 10, 7
NQ = int(os.environ.get("HEGREC_NQ", 1000))
UP_KIB, KEYS_MIB, SCORE_KIB = 326.5, 8.46, 128.1
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def pack(B):
    return np.packbits(B, axis=1)


def popc(P):
    return POP[P].sum(1)


def seq_classes(cols):
    X = np.concatenate([c.reshape(len(c), -1).view(np.uint8) for c in cols], axis=1)
    _, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    return cnt[inv.ravel()]


def lcomb(n, r):
    out = np.full(n.shape, -np.inf)
    ok = (r >= 0) & (r <= n)
    out[ok] = [lgamma(a + 1) - lgamma(b + 1) - lgamma(a - b + 1) for a, b in zip(n[ok], r[ok])]
    return out


def boot(x, B=2000):
    r = np.random.default_rng(123); x = np.asarray(x, dtype=float)
    m = np.array([x[r.integers(0, len(x), len(x))].mean() for _ in range(B)])
    return [round(float(np.percentile(m, 2.5)), 4), round(float(np.percentile(m, 97.5)), 4)]


def main():
    U, I, T, nU, nI = H.load_hk(); C = math.ceil(nI / H.SLOTS); ch = I // H.SLOTS

    def lists(mask):
        key = np.unique(U[mask] * C + ch[mask]); uu = key // C; cc = key % C
        off = np.searchsorted(uu, np.arange(nU + 1))
        return [cc[off[u]:off[u + 1]].tolist() for u in range(nU)]

    cut10 = int(np.quantile(T, 0.90)); cut5 = int(np.quantile(T, 0.95))
    B = [to_bits(lists(T <= c), C) for c in (cut10, cut5, int(T.max()))]
    have = B[0].any(1); B = [b[have] for b in B]; n = len(B[0])
    P = [pack(b) for b in B]; A = [popc(p) for p in P]
    rng = np.random.default_rng(SEED); q = np.sort(rng.choice(n, NQ, replace=False))
    out = {"note": __doc__.strip().split("\n\n")[0], "users": n, "queries": NQ, "C": C, "methods": {}}

    def traffic(chunks):
        return round(float(np.mean(chunks)) * UP_KIB / 1024 + KEYS_MIB + C * SCORE_KIB / 1024, 2)

    def det(name, labels, chunks, full):
        rec = {}
        for R in (1, 2, 3):
            cs = seq_classes(labels[3 - R:])
            rec[f"R{R}"] = round(float(np.mean(1.0 / cs[q])), 4)
            rec[f"R{R}_population"] = round(float(np.mean(1.0 / cs)), 6)
            rec[f"R{R}_ci95"] = boot(1.0 / cs[q])
        rec["traffic_MiB_per_request"] = [traffic(c[q]) for c in chunks]
        rec["full_history"] = full
        out["methods"][name] = rec; print(name, rec, flush=True)

    t0 = time.time()
    det("sparse", P, A, True)
    out["methods"]["dense"] = {"R1": round(1 / n, 7), "R2": round(1 / n, 7), "R3": round(1 / n, 7),
                               "traffic_MiB_per_request": [traffic(np.array([C]))] * 3, "full_history": True}
    cb = Codebook(B[0], K_COVER)
    g = [cb.send(b) for b in B]
    det("cover_rechosen", [x[:, None] for x in g], [cover_bits(cb, x).sum(1) for x in g], True)
    g0 = g[0]; cov0 = cover_bits(cb, g0)
    det("cover_fixed", [g0[:, None]] * 3, [cov0.sum(1)] * 3, False)
    print(f"[{time.time()-t0:.0f}s] decoys", flush=True)

    allc = np.arange(C)
    if os.environ.get("UNIFIED_SKIP_DECOYS"):
        json.dump(out, open(os.path.join(RESULTS, "unified_comparison_hk_v2_covers.json"), "w"), indent=2); print("SAVED"); return
    from _par import pmap
    for mi, mode in enumerate(("redrawn", "fixed")):
        def one(u):
            rng = np.random.default_rng([SEED, mi, int(u)])
            sc = {}; up = []
            D0 = rng.choice(allc, K_DECOY, replace=False)
            O = []
            for t in range(3):
                D = D0 if mode == "fixed" else (D0 if t == 0 else rng.choice(allc, K_DECOY, replace=False))
                o = B[t][u].copy(); o[D] = True; O.append(np.packbits(o)); up.append(int(o.sum()))
            for R in (1, 2, 3):
                ts = list(range(3 - R, 3))
                feas = np.ones(n, dtype=bool)
                for t in ts:
                    feas &= ~((P[t] & ~O[t][None, :]).any(1))
                cand = np.flatnonzero(feas)
                if mode == "redrawn":
                    ll = np.zeros(len(cand))
                    for t in ts:
                        a = A[t][cand]; o = int(POP[O[t]].sum())
                        ll += lcomb(a, K_DECOY - (o - a))
                else:
                    inter = O[ts[0]].copy()
                    for t in ts[1:]: inter &= O[t]
                    uni = np.zeros((len(cand), P[0].shape[1]), dtype=np.uint8)
                    for t in ts: uni |= (O[t][None, :] & ~P[t][cand])
                    m = popc(uni); ii = int(POP[inter].sum())
                    ok = ~((uni & ~inter[None, :]).any(1))
                    ll = np.where(ok, lcomb(np.full(len(cand), ii) - m, K_DECOY - m), -np.inf)
                best = ll.max(); top = cand[ll >= best - 1e-9]
                sc[R] = 1.0 / len(top) if u in set(top.tolist()) else 0.0
            return sc, up
        res = pmap(one, q)
        succ = {R: [r[0][R] for r in res] for R in (1, 2, 3)}; ups = [[r[1][t] for r in res] for t in range(3)]
        rec = {f"R{R}": round(float(np.mean(v)), 4) for R, v in succ.items()}
        rec.update({f"R{R}_ci95": boot(np.array(v)) for R, v in succ.items()})
        rec["traffic_MiB_per_request"] = [traffic(np.array(x)) for x in ups]; rec["full_history"] = True
        out["methods"][f"decoys_{mode}"] = rec; print(mode, rec, flush=True)
        json.dump(out, open(os.path.join(RESULTS, "unified_comparison_hk_v2.json"), "w"), indent=2)
    json.dump(out, open(os.path.join(RESULTS, "unified_comparison_hk_v2.json"), "w"), indent=2)
    print("SAVED")


if __name__ == "__main__":
    main()
