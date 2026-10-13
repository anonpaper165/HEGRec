"""
How much of the request must the operator's profile share? (AR2023-HK, year-old profiles.)

hk_openworld_partial.py measured three keep probabilities and a disjoint split with the overlap
rankings. This sweeps the share of the history the request carries, p = 0.1, 0.25, 0.5, 0.75, 0.9, 1
(each interaction kept independently, seed 7), against the year-old signed-in profiles, and adds an
attacker that needs no shared interaction, only shared chunks: IDF-weighted cosine between the
request's chunk set and each profile's (idf(c) = log(N / users with chunk c in their profile)), next to
the two overlap rankings of hk_openworld_partial.py (Jaccard; share of the request's chunks held, ties to
fewer chunks). The disjoint split (signed-in half up to the snapshot as profile, the other half of all
time as request, no shared interaction) is scored with all three.
Every rule is reported separately (no maximum over rules); rates are over the same 1,000 queries and
rescaled by the share of users with a profile, as in the paper.
Output: results/hk_overlap_curve.json
"""
from _paths import RESULTS
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_amazon2023_hk as H
from hk_openworld_partial import chunk_lists, index, score
from _par import pmean
NQ = int(os.environ.get("HEGREC_NQ", 1000))

PS = (0.1, 0.25, 0.5, 0.75, 0.9, 1.0)


def idf_score(Q, true_u, idx, idf, rnorm, acc):
    us, st, en, nact = idx
    touched = [us[st[c]:en[c]] for c in Q]
    accf = acc
    for c, sl in zip(Q, touched): accf[sl] += idf[c] ** 2
    cand = np.unique(np.concatenate(touched)) if touched else np.empty(0, dtype=np.int64)
    hit = 0.0
    if len(cand):
        qn = np.sqrt(sum(idf[c] ** 2 for c in Q))
        s = accf[cand] / (rnorm[cand] * qn + 1e-12); top = s >= s.max() - 1e-9
        pos = np.flatnonzero(cand == true_u)
        if len(pos) and top[pos[0]]: hit = 1.0 / int(top.sum())
    for sl in touched: accf[sl] = 0
    return hit


def main():
    t0 = time.time()
    U, I, T, nU, nI = H.load_hk(); C = int(np.ceil(nI / H.SLOTS)); ch = I // H.SLOTS
    cut = int(np.quantile(T, 0.90)); old = T <= cut
    ref = chunk_lists(U, ch, nU, C, old)
    have = np.array([len(t) > 0 for t in ref]); f = float(have.mean()); hv = np.flatnonzero(have)
    rng = np.random.default_rng(7); acc = np.zeros(nU, dtype=np.int32); accf = np.zeros(nU)
    q_idx = rng.choice(hv, NQ, replace=False)
    idx = index(ref, have, nU, C)
    dfc = np.bincount(np.concatenate([np.array(t, dtype=np.int64) for t in ref if len(t)]), minlength=C)
    idf = np.log(have.sum() / np.maximum(dfc, 1))
    rnorm = np.array([np.sqrt(sum(idf[c] ** 2 for c in t)) if len(t) else 1.0 for t in ref])
    out = {"note": __doc__.strip().split("\n\n")[0], "attackable_fraction": round(f, 4), "p": {}}
    for p in PS:
        keep = rng.random(len(U)) < p
        dev = chunk_lists(U, ch, nU, C, keep)
        qs = [u for u in q_idx if len(dev[u])]
        r = {"jaccard": pmean(lambda u: score(dev[u], u, idx, nU, acc, False, "jaccard")[0], qs),
             "share": pmean(lambda u: score(dev[u], u, idx, nU, acc, False, "cover")[0], qs),
             "idf_cosine": pmean(lambda u: idf_score(dev[u], u, idx, idf, rnorm, accf), qs)}
        out["p"][str(p)] = {k: round(v * f, 4) for k, v in r.items()}
        print(p, out["p"][str(p)], f"[{time.time()-t0:.0f}s]", flush=True)
        json.dump(out, open(os.path.join(RESULTS, "hk_overlap_curve.json"), "w"), indent=2)
    half = rng.random(len(U)) < 0.5
    ref_h = chunk_lists(U, ch, nU, C, half & old); dev_h = chunk_lists(U, ch, nU, C, ~half)
    have_h = np.array([len(t) > 0 for t in ref_h]); fh = float(have_h.mean())
    idx_h = index(ref_h, have_h, nU, C)
    dfh = np.bincount(np.concatenate([np.array(t, dtype=np.int64) for t in ref_h if len(t)]), minlength=C)
    idfh = np.log(have_h.sum() / np.maximum(dfh, 1))
    rnh = np.array([np.sqrt(sum(idfh[c] ** 2 for c in t)) if len(t) else 1.0 for t in ref_h])
    qh = rng.choice(np.flatnonzero(have_h & np.array([len(t) > 0 for t in dev_h])), NQ, replace=False)
    r = {"jaccard": pmean(lambda u: score(dev_h[u], u, idx_h, nU, acc, False, "jaccard")[0], qh),
         "share": pmean(lambda u: score(dev_h[u], u, idx_h, nU, acc, False, "cover")[0], qh),
         "idf_cosine": pmean(lambda u: idf_score(dev_h[u], u, idx_h, idfh, rnh, accf), qh)}
    out["disjoint_halves"] = {k: round(v * fh, 4) for k, v in r.items()} | {"attackable_fraction": round(fh, 4)}
    print("disjoint", out["disjoint_halves"], f"[{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open(os.path.join(RESULTS, "hk_overlap_curve.json"), "w"), indent=2)
    print("SAVED")


if __name__ == "__main__":
    main()
