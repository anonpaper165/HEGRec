"""Fixed covers on Amazon-Book: share of users whose input changes (App. C.4) and linked-request linkage (Fig. 3b).
Output: results/cover_substitution_recall_ab.json
"""
from _paths import DATA, REPO, RESULTS
import os, sys, json, math, time
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, f"{REPO}")
import torch
from src.dataset import RecDataset
from src.model import LightGCN
from location_padding2 import to_bits, Codebook, cover_bits

SLOTS, K, KS, SPLIT, SEED = 4096, 20, (5, 10, 20), 0.8, 0


def reid(labels):
    if labels.ndim == 1: labels = labels[:, None]
    _, inv, cnt = np.unique(labels, axis=0, return_inverse=True, return_counts=True)
    return float(np.mean(1.0 / cnt[inv.ravel()]))


def run(sub):
    ds = RecDataset(f"{DATA}/{sub}", sub, rating_threshold=0)
    ck = torch.load(f"{REPO}/models/{sub}_best.pt", map_location="cpu")
    m = LightGCN(ck["n_users"], ck["n_items"], ck["args"].get("emb_dim", 64), ck["args"].get("n_layers", 3))
    m.load_state_dict(ck["model_state_dict"]); m.eval(); L = m.n_layers; M = ck["n_items"]
    with torch.no_grad():
        E = torch.cat([m.user_embedding.weight, m.item_embedding.weight], 0); ls = [E]
        for _ in range(L):
            E = torch.sparse.mm(ds.adj_matrix, E); ls.append(E)
        items = [l[ck["n_users"]:] for l in ls]
        ebar = torch.stack(items[:L], 0).mean(0).numpy().astype(np.float32); ehat = torch.stack(items, 0).mean(0).numpy().astype(np.float32)
    deg = np.zeros(M)
    for it in ds.train_dict.values():
        for i in it: deg[i] += 1
    inv = np.where(deg > 0, deg ** -0.5, 0.0)
    en = ebar / np.maximum(np.linalg.norm(ebar, axis=1, keepdims=True), 1e-9)
    C = math.ceil(M / SLOTS)
    users = [u for u in sorted(ds.train_dict) if ds.train_dict[u]]
    rng = np.random.default_rng(SEED)
    full, start = [], []
    for u in users:
        s = np.array(sorted(ds.train_dict[u])); keep = rng.random(len(s)) < SPLIT
        if not keep.any(): keep[rng.integers(len(s))] = True
        full.append(s); start.append(s[keep])
    Bs = to_bits([sorted(set((x // SLOTS).tolist())) for x in start], C)
    Bf = to_bits([sorted(set((x // SLOTS).tolist())) for x in full], C)
    chunk_items = [np.arange(c * SLOTS, min((c + 1) * SLOTS, M)) for c in range(C)]
    test = [np.array(sorted(ds.test_dict.get(u, []))) for u in users]
    disc = 1 / np.log2(np.arange(2, K + 2))

    def uvec(sup):
        if len(sup) == 0: return np.zeros(ebar.shape[1], dtype=np.float32)
        w = inv[sup] / math.sqrt(len(sup)); return (ebar[sup] * w[:, None]).sum(0)

    def score_many(sups, idx):
        out = np.zeros((len(idx), 2))
        for b0 in range(0, len(idx), 2048):
            UV = np.stack([uvec(x) for x in sups[b0:b0 + 2048]]).astype(np.float32)
            SC = UV @ ehat.T
            for r, j in enumerate(idx[b0:b0 + 2048]):
                sc = SC[r]; sc[full[j]] = -np.inf
                top = np.argpartition(-sc, K)[:K]; top = top[np.argsort(-sc[top])]
                t = test[j]; hit = np.isin(top, t)
                out[b0 + r] = (hit.sum() / min(len(t), K), float((hit * disc).sum() / disc[:min(len(t), K)].sum()))
        return out

    def substitute(sup, covrow):
        inside = covrow[sup // SLOTS]
        if inside.all(): return sup, False
        pool = np.concatenate([chunk_items[c] for c in np.flatnonzero(covrow)])
        pool = pool[deg[pool] > 0]
        sub = pool[(en[pool] @ en[sup[~inside]].T).argmax(0)]
        return np.unique(np.concatenate([sup[inside], sub])), True

    res = {"users": len(users), "C": C, "split_epoch_start": SPLIT,
           "share_chunk_set_grew": round(float(np.mean((Bf != Bs).any(1))), 4), "k": {}}
    tu = [j for j in range(len(users)) if len(test[j])]
    base = score_many([full[j] for j in tu], tu)
    res["full"] = {"recall@20": round(float(base[:, 0].mean()), 5), "ndcg@20": round(float(base[:, 1].mean()), 5)}
    print(sub, "full", res["full"], flush=True)
    for k in KS:
        t0 = time.time(); cb = Codebook(Bs, k)
        g0 = cb.send(Bs); cov0 = cover_bits(cb, g0); gf = cb.send(Bf)
        drops, subs, chg = [], [], []
        for j in tu:
            sup = full[j]; row = cov0[j]
            drops.append(sup[row[sup // SLOTS]])
            s2, c = substitute(sup, row); subs.append(s2); chg.append(c)
        arms = {"fixed_drop": score_many(drops, tu), "fixed_substitute": score_many(subs, tu)}
        chg = np.array(chg); rec = {"changed_pct": round(100 * float(chg.mean()), 2)}
        for a, v in arms.items():
            v = np.array(v)
            rec[a] = {"recall@20": round(float(v[:, 0].mean()), 5), "ndcg@20": round(float(v[:, 1].mean()), 5),
                      "recall@20_changed": round(float(v[chg, 0].mean()), 5),
                      "ndcg@20_changed": round(float(v[chg, 1].mean()), 5)}
        rec["full_on_changed"] = {"recall@20": round(float(base[chg, 0].mean()), 5), "ndcg@20": round(float(base[chg, 1].mean()), 5)}
        rec["linkage"] = {"fixed_any_R": round(reid(g0), 5),
                          "rechosen_R1": round(reid(gf), 5),
                          "rechosen_R2": round(reid(np.stack([g0, gf], 1)), 5)}
        rec["upload_chunks_end_request"] = {"own": round(float(Bf.sum(1).mean()), 3),
                                            "fixed": round(float(cov0.sum(1).mean()), 3),
                                            "rechosen": round(float(cover_bits(cb, gf).sum(1).mean()), 3)}
        rec["rechosen_sends_all_C_pct"] = round(100 * float(np.mean(gf < 0)), 2)
        res["k"][str(k)] = rec; print(sub, k, json.dumps(rec), f"[{time.time()-t0:.0f}s]", flush=True)
    return res


if __name__ == "__main__":
    out = {"note": __doc__.strip().split("\n\n")[0], "datasets": {}}
    for name, sub in (("Amazon-Book", "amazon-book"), ("Gowalla", "gowalla")):
        out["datasets"][name] = run(sub)
        json.dump(out, open(os.path.join(RESULTS, "cover_substitution_recall_ab.json"), "w"), indent=2)
    print("SAVED")
