"""Table 6 for Gowalla, Amazon-Book and AR2023-HK (LightGCN), by inference from the trained models.
Output: results/accuracy_table_eval.json
"""
from _paths import DATA, REPO, RESULTS
import os, sys, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, str(REPO))
import torch
from src.dataset import RecDataset
from src.model import LightGCN
from serving_formulation import variants, recall_at_k
import poly_ngcf_serve as P
import hk_lightgcn_train as T

out = {"note": __doc__.strip().split("\n\n")[0], "datasets": {}}
for name, key in (("Gowalla", "gowalla"), ("Amazon-Book", "amazon-book")):
    ds = RecDataset(f"{DATA}/{key}", key, rating_threshold=0)
    ck = torch.load(f"{REPO}/models/{key}_best.pt", map_location="cpu", weights_only=False)
    m = LightGCN(ck["n_users"], ck["n_items"], ck["args"].get("emb_dim", 64), ck["args"].get("n_layers", 3))
    m.load_state_dict(ck["model_state_dict"]); m.eval(); v = variants(m, ds, ck["n_users"])
    res = {"lightgcn_standard": round(recall_at_k(*v["vanilla"], ds), 4), "lightgcn_account_free": round(recall_at_k(*v["drop_user_L0"], ds), 4)}
    nck = torch.load(f"{REPO}/models/{key}_ngcf_3l.pt", map_location=P.DEV, weights_only=False)
    users = [u for u in ds.test_dict if ds.test_dict[u] and ds.train_dict.get(u)]
    res["ngcf_original"] = round(P.ngcf_recall(nck, ds, ds.adj_matrix.to(P.DEV), users), 4)
    mk = torch.load(os.path.join(RESULTS, "poly_ngcf", f"{key}.pt"), map_location=P.DEV, weights_only=False)
    pm = P.PolyNGCF(nck, ds.n_users, ds.n_items, mk["coef"]).to(P.DEV); pm.load_state_dict(mk["state"]); pm.eval()
    with torch.no_grad():
        ue, ie = pm(P.build_adj(ds.train_dict, ds.n_users, ds.n_items).to(P.DEV))
        res["ngcf_poly"] = round(P.recall(ue, ie, ds.train_dict, ds.test_dict, users), 4)
    out["datasets"][name] = res; print(name, res, flush=True)

z = np.load(os.path.join(RESULTS, "hk_lightgcn_lr5e3", "split.npz"))
U, I, tr, te = z["U"].astype(np.int64), z["I"].astype(np.int64), z["tr"], z["te"]
nU, nI = int(U.max()) + 1, int(I.max()) + 1; off = np.searchsorted(U, np.arange(nU + 1))
tr_o, tr_I = T.lists(U, I, off, tr, nU); te_o, te_I = T.lists(U, I, off, te, nU)
A, du, di = T.adj(U[tr], I[tr], nU, nI); test_users = np.flatnonzero(np.diff(te_o) > 0)
ck = torch.load(os.path.join(RESULTS, "hk_lightgcn_lr5e3", "ckpt.pt"))
with torch.no_grad():
    ls = T.propagate(A, ck["Eu"].to(T.DEV), ck["Ei"].to(T.DEV)); eb, eh = T.served_vectors(ls, nU, di); Ef = torch.stack(ls).mean(0)
    served = T.recall(T.served_scores(eb, eh, tr_o, tr_I, du), test_users, tr_o, tr_I, te_o, te_I)
    account = T.recall(lambda us: Ef[torch.tensor(us, device=T.DEV)] @ Ef[nU:].T, test_users, tr_o, tr_I, te_o, te_I)
out["datasets"]["AR2023-HK"] = {"lightgcn_standard": round(account, 5), "lightgcn_account_free": round(served, 5)}
print("AR2023-HK", out["datasets"]["AR2023-HK"], flush=True)
json.dump(out, open(os.path.join(RESULTS, "accuracy_table_eval.json"), "w"), indent=2)
