"""Recall@20 lost on AR2023-HK under fixed covers (Table 8; HK_SPLIT=temporal|temporal_early) or partial histories (HK_MODE=removal; App. C.4).
Output: results/hk_cover_recall{,_temporal,_temporal_early}.json, results/hk_removal_recall.json
"""
from _paths import RESULTS
import os, sys, json, math, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from location_padding2 import to_bits, Codebook, cover_bits

DEV = os.environ.get("HK_DEVICE", "cuda:0")
RUN = os.environ.get("HK_RUN", "hk_lightgcn_lr5e3")
SLOTS, K, KS, SPLIT, SEEDS = 4096, 20, (5, 10, 20), 0.8, range(5)
MODE = os.environ.get("HK_MODE", "cover"); PS = (0.75, 0.5, 0.25)
SPLIT_MODE = os.environ.get("HK_SPLIT", "random")
if SPLIT_MODE.startswith("temporal"): SEEDS = range(1)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    d = np.load(os.path.join(RESULTS, RUN, "split.npz")); U, I, tr, te = d["U"].astype(np.int64), d["I"].astype(np.int64), d["tr"], d["te"]
    nU, nI = int(U.max()) + 1, int(I.max()) + 1; C = math.ceil(nI / SLOTS)
    eb = torch.tensor(np.load(os.path.join(RESULTS, RUN, "ebar_s.npy")), device=DEV)
    eh = torch.tensor(np.load(os.path.join(RESULTS, RUN, "ehat.npy")), device=DEV)
    off = np.searchsorted(U, np.arange(nU + 1))
    hist = [I[off[u]:off[u + 1]][tr[off[u]:off[u + 1]]] for u in range(nU)]
    test = [I[off[u]:off[u + 1]][te[off[u]:off[u + 1]]] for u in range(nU)]
    idx = np.array([u for u in range(nU) if len(test[u]) and len(hist[u])])
    if SPLIT_MODE.startswith("temporal"):
        import dataset_amazon2023_hk as H
        U0, I0, T0, _, _ = H.load_hk(); k0 = U0.astype(np.int64) * nI + I0
        o = np.lexsort((T0, k0)); k0, T0 = k0[o], T0[o]; first = np.r_[True, k0[1:] != k0[:-1]]
        kk, tt = k0[first], T0[first]; key = U * nI + I
        pos = np.searchsorted(kk, key); assert np.all(kk[pos] == key)
        Tall = tt[pos]; thist = [Tall[off[u]:off[u + 1]][tr[off[u]:off[u + 1]]] for u in range(nU)]
        log("temporal hold-out: latest 20% of each training history by first-interaction time")
    log(f"{nU:,} users, {nI:,} items, C={C}, {len(idx):,} scored")

    @torch.no_grad()
    def recall(sups, bs=1024):
        out = np.empty(len(idx))
        for b0 in range(0, len(idx), bs):
            us = idx[b0:b0 + bs]
            r = np.concatenate([np.full(len(sups[u]), k) for k, u in enumerate(us)]); c = np.concatenate([sups[u] for u in us])
            w = np.concatenate([np.full(len(sups[u]), 1 / math.sqrt(max(len(sups[u]), 1))) for u in us])
            S = torch.sparse_coo_tensor(torch.tensor(np.vstack([r, c])), torch.tensor(w, dtype=torch.float32), (len(us), nI)).to(DEV)
            sc = torch.sparse.mm(S, eb) @ eh.T
            r2 = np.concatenate([np.full(len(hist[u]), k) for k, u in enumerate(us)]); c2 = np.concatenate([hist[u] for u in us])
            sc[torch.tensor(r2, device=DEV), torch.tensor(c2, device=DEV)] = -float("inf")
            top = torch.topk(sc, K, 1).indices.cpu().numpy()
            for k, u in enumerate(us):
                out[b0 + k] = np.isin(top[k], test[u]).sum() / len(test[u])
        return out

    base = recall(hist); log(f"full history recall {base.mean():.5f}")
    res = {"note": __doc__.strip().split("\n\n")[0], "run": RUN, "users_scored": int(len(idx)),
           "full_recall": round(float(base.mean()), 5), "k": {}}
    if MODE == "removal":
        res["note"] = "What sending part of the history costs the recommendation (AR2023-HK), as removal_recall_ab.py."; res["p"] = {}
        for p in PS:
            rel = []
            for seed in SEEDS:
                r2 = np.random.default_rng(100 + seed); sups = []
                for s in hist:
                    if len(s) == 0: sups.append(s); continue
                    keep = r2.random(len(s)) < p
                    if not keep.any(): keep[r2.integers(len(s))] = True
                    sups.append(s[keep])
                rel.append(100 * (recall(sups).mean() / base.mean() - 1)); log(f"p={p} seed={seed} rel {rel[-1]:.3f}%")
            res["p"][str(p)] = {"rel_change_pct": [round(x, 3) for x in rel], "mean": round(float(np.mean(rel)), 3), "sd": round(float(np.std(rel)), 3)}
            json.dump(res, open(os.path.join(RESULTS, "hk_removal_recall.json"), "w"), indent=2)
        log("SAVED"); return
    for k in KS:
        rel, drop = [], []
        for seed in SEEDS:
            rng = np.random.default_rng(seed); start = []
            for u, s in enumerate(hist):
                if len(s) == 0: start.append(s); continue
                if SPLIT_MODE.startswith("temporal"):
                    n_late = int(round((1 - SPLIT) * len(s))); n_keep = max(1, len(s) - n_late)
                    o = np.argsort(thist[u], kind="stable"); keep = np.zeros(len(s), bool)
                    keep[o[-n_keep:] if SPLIT_MODE == "temporal_early" else o[:n_keep]] = True
                else:
                    keep = rng.random(len(s)) < SPLIT
                    if not keep.any(): keep[rng.integers(len(s))] = True
                start.append(s[keep])
            Bs = to_bits([sorted(set((x // SLOTS).tolist())) for x in start], C)
            cb = Codebook(Bs, k); cov = cover_bits(cb, cb.send(Bs))
            sups = [h[cov[j][h // SLOTS]] if len(h) else h for j, h in enumerate(hist)]
            rr = recall(sups)
            rel.append(100 * (rr.mean() / base.mean() - 1))
            drop.append(100 * sum(len(h) - len(x) for h, x in zip(hist, sups)) / sum(len(h) for h in hist))
            log(f"k={k} seed={seed} rel {rel[-1]:.3f}% dropped {drop[-1]:.3f}%")
        if SPLIT_MODE.startswith("temporal"): res["split"] = SPLIT_MODE + " (latest, or for temporal_early earliest, 20% of each training history held out of the codebook)"
        res["k"][str(k)] = {"rel_change_pct": [round(x, 3) for x in rel], "mean": round(float(np.mean(rel)), 3),
                            "sd": round(float(np.std(rel)), 3), "dropped_pct_mean": round(float(np.mean(drop)), 3)}
        json.dump(res, open(os.path.join(RESULTS, f"hk_cover_recall_{SPLIT_MODE}.json" if SPLIT_MODE.startswith("temporal") else "hk_cover_recall.json"), "w"), indent=2)
    log("SAVED")


if __name__ == "__main__":
    main()
