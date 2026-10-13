"""
Every defence against the realistic attacker: year-old profiles (AR2023-HK, growing histories).

unified_comparison_hk.py scores the defences at the upper bound: the operator knows every account's
history at every request. Here the operator holds only the signed-in profiles of the one-year-old
snapshot (as in Sec. 5.1) and matches requests made after it: at the 5% snapshot (about four months on)
and at the end of the data, each carrying the history to date; R = 1 is the end request, R = 2 links
both. Same 1,000 users and seed as unified_comparison_hk.py.
Attacker (the stale attacker of App. D.1, extended to linked requests): a profile is a candidate if its
chunk set lies inside every observed set (histories only grow, and decoys and covers only add chunks);
candidates are ranked by Jaccard between their chunk set and the intersection of the observed sets, or,
for covers, by Jaccard between the cover the profile itself would send and that intersection; the rule
with the better overall rate is reported (both are kept), ties split 1/#. Rates are over the 1,000 users with a snapshot profile
(multiply by the 98.8% who have one for the population rate, as in Sec. 5.1).
Output: results/unified_comparison_stale_hk.json
"""
from _paths import RESULTS
import os, sys, json, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from location_padding2 import to_bits, Codebook, cover_bits
import dataset_amazon2023_hk as H
from _par import pmap

K_DECOY, K_COVER, SEED = 32, 10, 7
NQ = int(os.environ.get("HEGREC_NQ", 1000))
POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def popc(P):
    return POP[P].sum(1)


def main():
    t0 = time.time()
    U, I, T, nU, nI = H.load_hk(); C = math.ceil(nI / H.SLOTS); ch = I // H.SLOTS

    def lists(mask):
        key = np.unique(U[mask] * C + ch[mask]); uu = key // C; cc = key % C
        off = np.searchsorted(uu, np.arange(nU + 1))
        return [cc[off[u]:off[u + 1]].tolist() for u in range(nU)]

    cut10 = int(np.quantile(T, 0.90)); cut5 = int(np.quantile(T, 0.95))
    B = [to_bits(lists(T <= c), C) for c in (cut10, cut5, int(T.max()))]
    have = B[0].any(1); B = [b[have] for b in B]; n = len(B[0])
    rng = np.random.default_rng(SEED); q = np.sort(rng.choice(n, NQ, replace=False))   # = unified_comparison_hk.py
    ref = np.packbits(B[0], axis=1); rn = popc(ref)
    cb = Codebook(B[0], K_COVER); g = [cb.send(b) for b in B]
    refcov = np.packbits(cover_bits(cb, g[0]), axis=1); rcn = popc(refcov)
    allc = np.arange(C)
    out = {"note": __doc__.strip().split("\n\n")[0], "users_with_profile": n, "queries": NQ, "methods": {}}

    def attack(obs_list, u, use_cover):
        inter = obs_list[0].copy()
        for o in obs_list[1:]: inter &= o
        feas = ~((ref & ~inter[None, :]).any(1))
        cand = np.flatnonzero(feas)
        nr = 2 if use_cover else 1
        if u not in set(cand.tolist()): return [0.0] * nr
        res = []
        ranks = [(ref, rn)] + ([(refcov, rcn)] if use_cover else [])
        ni = int(POP[inter].sum())
        for R, Rn in ranks:
            it = popc(R[cand] & inter[None, :]).astype(float)
            j = it / np.maximum(Rn[cand] + ni - it, 1)
            top = cand[j >= j.max() - 1e-12]
            res.append(1.0 / len(top) if u in set(top.tolist()) else 0.0)
        return res

    methods = {"sparse": False, "decoys_redrawn": False, "decoys_fixed": False,
               "cover_rechosen": True, "cover_fixed": True}
    for name, use_cover in methods.items():
        mi = list(methods).index(name)

        def one(u):
            rng = np.random.default_rng([SEED, mi, int(u)])          # per user: parallel-safe
            D0 = rng.choice(allc, K_DECOY, replace=False)
            obs = []
            for t in (1, 2):
                if name == "sparse": o = B[t][u].copy()
                elif name == "decoys_redrawn":
                    o = B[t][u].copy(); o[rng.choice(allc, K_DECOY, replace=False)] = True
                elif name == "decoys_fixed":
                    o = B[t][u].copy(); o[D0] = True
                elif name == "cover_rechosen":
                    o = cover_bits(cb, g[t][u:u + 1])[0]
                else:
                    o = cover_bits(cb, g[0][u:u + 1])[0]
                obs.append(np.packbits(o))
            return attack([obs[1]], u, use_cover), attack(obs, u, use_cover)
        res = pmap(one, q); s1 = [r[0] for r in res]; s2 = [r[1] for r in res]
        m1 = np.mean(np.array(s1), 0); m2 = np.mean(np.array(s2), 0)      # per ranking rule
        rec = {"R1_end": round(float(m1.max()), 4), "R2_linked": round(float(m2.max()), 4),
               "per_rule_R1": [round(float(x), 4) for x in m1], "per_rule_R2": [round(float(x), 4) for x in m2]}
        out["methods"][name] = rec; print(name, rec, f"[{time.time()-t0:.0f}s]", flush=True)
        json.dump(out, open(os.path.join(RESULTS, "unified_comparison_stale_hk.json"), "w"), indent=2)
    out["methods"]["dense"] = {"R1_end": round(1 / n, 7), "R2_linked": round(1 / n, 7)}
    json.dump(out, open(os.path.join(RESULTS, "unified_comparison_stale_hk.json"), "w"), indent=2)
    print("SAVED")


if __name__ == "__main__":
    main()
