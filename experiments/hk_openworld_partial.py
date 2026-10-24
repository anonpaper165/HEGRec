"""Open-world test on AR2023-HK (App. E; §5.2). Paper run: HEGREC_NQ=10000 SKIP_PARTIAL=1.
Output: results/hk_openworld_partial.json, results/hk_openworld_scores.json
"""
from _paths import RESULTS
import os, sys, json, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_amazon2023_hk as H
from _par import pmap
NQ = int(os.environ.get("HEGREC_NQ", 1000))


def chunk_lists(U, ch, nU, C, mask):
    key = np.unique(U[mask] * C + ch[mask]); uu = key // C; cc = key % C
    off = np.searchsorted(uu, np.arange(nU + 1))
    return [tuple(cc[off[u]:off[u + 1]].tolist()) for u in range(nU)]


def index(ref_tuples, valid, nU, C):
    uc, users = [], []
    for u in np.flatnonzero(valid):
        for c in ref_tuples[u]:
            uc.append(c); users.append(u)
    uc = np.array(uc, dtype=np.int64); users = np.array(users, dtype=np.int64)
    order = np.argsort(uc, kind="stable"); cs, us = uc[order], users[order]
    st = np.searchsorted(cs, np.arange(C), "left"); en = np.searchsorted(cs, np.arange(C), "right")
    nact = np.array([len(t) for t in ref_tuples], dtype=np.int64)
    return us, st, en, nact


def score(Q, true_u, idx, nU, acc, subset, rule):
    us, st, en, nact = idx
    touched = [us[st[c]:en[c]] for c in Q]
    for sl in touched: acc[sl] += 1
    cand = np.unique(np.concatenate(touched)) if touched else np.empty(0, dtype=np.int64)
    out = (0.0, -1.0, 0.0)
    if len(cand):
        inter = acc[cand].astype(np.float64); rn = nact[cand].astype(np.float64)
        if rule == "jaccard":
            s = inter / np.maximum(rn + len(Q) - inter, 1)
            if subset: s[inter < rn] = -1.0
            best = s.max(); top = s == best
        else:
            s = inter; best0 = s.max(); top = s == best0
            m = rn[top].min(); top = top & (rn == m); best = best0 / max(len(Q), 1)
        srt = np.sort(s)[::-1]; second = srt[1] if len(srt) > 1 else -1.0
        hit = 0.0
        if true_u is not None:
            pos = np.flatnonzero(cand == true_u)
            if len(pos) and top[pos[0]]: hit = 1.0 / int(np.count_nonzero(top))
        out = (hit, float(best), float(best - (second if rule == "jaccard" else second / max(len(Q), 1))))
    for sl in touched: acc[sl] = 0
    return out


def exact_thresholds(R, X, eps=1e-9):
    R = np.array(R); X = np.array(X); h = R[:, 0]; out = {}
    def m(k, t, op):
        sr = R[:, k] >= t - eps if op == "ge" else R[:, k] > t + eps
        sx = X[:, k] >= t - eps if op == "ge" else X[:, k] > t + eps
        rec = float(np.mean(h * sr)); fc = float(np.mean(sx))
        return {"recall": round(rec, 4), "false_claim": round(fc, 4), "precision_equal_mix": round(rec / max(sr.mean() + fc, 1e-12), 4)}
    out["margin_ge_0.2"] = m(2, 0.2, "ge"); out["margin_gt_0.2"] = m(2, 0.2, "gt")
    out["margin_ge_0.3333"] = m(2, 1 / 3, "ge"); out["margin_gt_0.3333"] = m(2, 1 / 3, "gt"); out["top_ge_1.0"] = m(1, 1.0, "ge")
    best = max(((m(1, t, "ge"), t) for t in np.unique(np.concatenate([R[:, 1], X[:, 1]]))), key=lambda z: z[0]["precision_equal_mix"])
    out["top_score_max_precision"] = dict(best[0], t=round(float(best[1]), 4))
    return out


def main():
    U, I, T, nU, nI = H.load_hk(); C = int(np.ceil(nI / H.SLOTS)); ch = I // H.SLOTS
    cut = int(np.quantile(T, 0.90)); old = T <= cut
    full = chunk_lists(U, ch, nU, C, np.ones(len(U), dtype=bool)); ref = chunk_lists(U, ch, nU, C, old)
    have = np.array([len(t) > 0 for t in ref]); f = float(have.mean())
    rng = np.random.default_rng(7); acc = np.zeros(nU, dtype=np.int32); out = {"note": __doc__.strip().split("\n\n")[0]}
    t0 = time.time()
    hv = np.flatnonzero(have); held = np.zeros(nU, dtype=bool); held[rng.choice(hv, int(0.1 * len(hv)), replace=False)] = True
    reg = rng.choice(np.flatnonzero(have & ~held), NQ, replace=False); unreg = rng.choice(np.flatnonzero(held), NQ, replace=False)
    idx = index(ref, have & ~held, nU, C)
    R = pmap(lambda u: score(full[u], u, idx, nU, acc, True, "jaccard"), reg)
    X = pmap(lambda u: score(full[u], None, idx, nU, acc, True, "jaccard"), unreg)
    ow = {"closed_world_recall_registered": round(float(np.mean([r[0] for r in R])), 4), "curves": {}}
    for conf, k in (("top_score", 1), ("margin", 2)):
        rs = np.array([r[k] for r in R]); xs = np.array([x[k] for x in X]); hits = np.array([r[0] for r in R])
        rows = []
        for t in np.unique(np.quantile(np.concatenate([rs, xs]), np.linspace(0, 0.99, 34))):
            rec = float(np.mean(hits * (rs >= t))); fc = float(np.mean(xs >= t))
            claims_r = float(np.mean(rs >= t)); prec = rec / max(claims_r + fc, 1e-12)
            rows.append({"t": round(float(t), 4), "recall": round(rec, 4), "false_claim": round(fc, 4), "precision_equal_mix": round(prec, 4)})
        ow["curves"][conf] = rows
    json.dump({"registered":[list(r) for r in R],"unregistered":[list(x) for x in X]},open(os.path.join(RESULTS,"hk_openworld_scores.json"),"w"))
    if os.environ.get("OPEN_WORLD_ONLY"): print("SCORES SAVED"); return
    ow["exact_thresholds"] = exact_thresholds(R, X)
    out["open_world"] = ow
    print("open world", ow["closed_world_recall_registered"], f"[{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open(os.path.join(RESULTS, "hk_openworld_partial.json"), "w"), indent=2)
    if os.environ.get("SKIP_PARTIAL"): print("SAVED"); return
    q_idx = rng.choice(hv, 1000, replace=False)
    idx_all = index(ref, have, nU, C)
    part = {}
    for p in (0.25, 0.5, 0.75):
        keep = rng.random(len(U)) < p
        dev = chunk_lists(U, ch, nU, C, keep)
        res = {r: float(np.mean([score(dev[u], u, idx_all, nU, acc, False, r)[0] for u in q_idx if len(dev[u])])) for r in ("jaccard", "cover")}
        part[f"device_p{p}"] = {k: round(v * f, 4) for k, v in res.items()} | {"best": round(max(res.values()) * f, 4)}
        print("device", p, part[f"device_p{p}"], f"[{time.time()-t0:.0f}s]", flush=True)
    half = rng.random(len(U)) < 0.5
    ref_h = chunk_lists(U, ch, nU, C, half & old); dev_h = chunk_lists(U, ch, nU, C, ~half)
    have_h = np.array([len(t) > 0 for t in ref_h]); fh = float(have_h.mean())
    idx_h = index(ref_h, have_h, nU, C)
    qh = rng.choice(np.flatnonzero(have_h & np.array([len(t) > 0 for t in dev_h])), 1000, replace=False)
    res = {r: float(np.mean([score(dev_h[u], u, idx_h, nU, acc, False, r)[0] for u in qh])) for r in ("jaccard", "cover")}
    part["disjoint_halves"] = {k: round(v * fh, 4) for k, v in res.items()} | {"best": round(max(res.values()) * fh, 4), "attackable_fraction": round(fh, 4)}
    print("disjoint", part["disjoint_halves"], f"[{time.time()-t0:.0f}s]", flush=True)
    out["partial_overlap"] = part
    out["chance"] = 1.0 / int(have.sum())
    json.dump(out, open(os.path.join(RESULTS, "hk_openworld_partial.json"), "w"), indent=2)
    print("SAVED")


if __name__ == "__main__":
    main()
