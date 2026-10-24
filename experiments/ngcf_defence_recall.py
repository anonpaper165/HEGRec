"""Recall@20 lost by partial histories and fixed covers, poly-NGCF (Table 8 NGCF row; Table 9 NGCF row).
Output: results/ngcf_defence_recall.json
"""
from _paths import DATA, REPO, RESULTS
import os, sys, json, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, str(REPO))
import torch
from src.dataset import RecDataset
from poly_ngcf_serve import PolyNGCF, build_adj, p_eval
from location_padding2 import to_bits, Codebook, cover_bits

SLOTS, K, SPLIT, PS, KS = 4096, 20, 0.8, (0.75, 0.5, 0.25), (5, 10, 20)
DEV = os.environ.get("POLY_DEVICE", "cuda:0")


def run(ds_name, key):
    d = RecDataset(f"{DATA}/{key}", key, rating_threshold=0); nU, nI = d.n_users, d.n_items
    mk = torch.load(os.path.join(RESULTS, "poly_ngcf", f"{key}.pt"), map_location=DEV, weights_only=False)
    ck = torch.load(f"{REPO}/models/{key}_ngcf_3l.pt", map_location=DEV, weights_only=False)
    m = PolyNGCF(ck, nU, nI, mk["coef"]).to(DEV); m.load_state_dict({k: v.to(DEV) for k, v in mk["state"].items()}); m.eval()
    with torch.no_grad():
        adj = build_adj(d.train_dict, nU, nI).to(DEV)
        E = torch.cat([torch.zeros(nU, 64, device=DEV), m.item]); items = [m.item]
        for l in range(3):
            a = torch.sparse.mm(adj, E); E = p_eval(m.coef, m.W1[l](E + a) + m.W2[l](E * a)); items.append(E[nU:])
    deg = np.zeros(nI)
    for it in d.train_dict.values():
        for i in it: deg[i] += 1
    inv = torch.tensor(np.where(deg > 0, deg ** -0.5, 0.0), dtype=torch.float32, device=DEV)
    users = [u for u in sorted(d.train_dict) if d.train_dict[u]]
    full = [np.array(sorted(d.train_dict[u])) for u in users]
    test = [np.array(sorted(d.test_dict.get(u, []))) for u in users]
    tu = [j for j in range(len(users)) if len(test[j])]
    C = math.ceil(nI / SLOTS)

    @torch.no_grad()
    def recall(sups, idx):
        out = []
        for b0 in range(0, len(idx), 4096):
            ss = sups[b0:b0 + 4096]; js = idx[b0:b0 + 4096]
            r = np.concatenate([np.full(len(s), k) for k, s in enumerate(ss)]); c = np.concatenate(ss)
            cw = torch.tensor(c, device=DEV)
            w = inv[cw] / torch.tensor(np.concatenate([np.full(len(s), math.sqrt(max(len(s), 1))) for s in ss]), dtype=torch.float32, device=DEV)
            S = torch.sparse_coo_tensor(torch.tensor(np.vstack([r, c]), device=DEV), w, (len(ss), nI))
            e = torch.zeros(len(ss), 64, device=DEV); sc = 0
            for l in range(3):
                a = torch.sparse.mm(S, items[l]); e = p_eval(m.coef, m.W1[l](e + a) + m.W2[l](e * a)); sc = sc + e @ items[l + 1].T
            for k, j in enumerate(js): sc[k, full[j]] = -float("inf")
            top = torch.topk(sc, K, 1).indices.cpu().numpy()
            out += [np.isin(top[k], test[j]).sum() / len(test[j]) for k, j in enumerate(js)]
        return np.array(out)

    base = recall([full[j] for j in tu], tu); res = {"users_scored": len(tu), "full_recall": round(float(base.mean()), 5), "removal": {}, "cover": {}}
    print(ds_name, "full", res["full_recall"], flush=True)
    for p in PS:
        rs = []
        for seed in range(5):
            r2 = np.random.default_rng(100 + seed); sups = []
            for j in tu:
                s0 = full[j]; keep = r2.random(len(s0)) < p
                if not keep.any(): keep[r2.integers(len(s0))] = True
                sups.append(s0[keep])
            rs.append(recall(sups, tu).mean())
        res["removal"][str(p)] = {"recall@20": round(float(np.mean(rs)), 5), "recall_rel_change_pct": round(100 * (float(np.mean(rs)) / base.mean() - 1), 2)}
        print(ds_name, "removal", p, res["removal"][str(p)], flush=True)
    for k in KS:
        rel = []
        for seed in range(5):
            rng = np.random.default_rng(seed); start = []
            for s in full:
                keep = rng.random(len(s)) < SPLIT
                if not keep.any(): keep[rng.integers(len(s))] = True
                start.append(s[keep])
            Bs = to_bits([sorted(set((x // SLOTS).tolist())) for x in start], C)
            cb = Codebook(Bs, k); cov = cover_bits(cb, cb.send(Bs))
            sups = [full[j][cov[j][full[j] // SLOTS]] for j in tu]
            rel.append(100 * (recall(sups, tu).mean() / base.mean() - 1))
        res["cover"][str(k)] = {"rel_change_pct": [round(x, 3) for x in rel], "mean": round(float(np.mean(rel)), 3), "sd": round(float(np.std(rel)), 3)}
        print(ds_name, "cover", k, res["cover"][str(k)], flush=True)
    return res


if __name__ == "__main__":
    out = {"note": __doc__.strip().split("\n\n")[0], "datasets": {}}
    for name, key in (("Amazon-Book", "amazon-book"), ("Gowalla", "gowalla")):
        out["datasets"][name] = run(name, key)
        json.dump(out, open(os.path.join(RESULTS, "ngcf_defence_recall.json"), "w"), indent=2)
    print("SAVED")
