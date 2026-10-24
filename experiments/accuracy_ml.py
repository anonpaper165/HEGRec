"""Table 6 for ML-100K and ML-1M (LightGCN, NGCF and poly-NGCF; App. D.2 passes).
Output: results/accuracy_ml.json
"""
from _paths import REPO, DATA, RESULTS
import os, sys, json
import numpy as np
sys.path.insert(0, str(REPO))
import torch
from src.dataset import RecDataset
from src.model import LightGCN
from serving_formulation import recall_at_k, variants
import poly_ngcf_serve as P

K = 20
DATASETS = [("ML-100K", "ml-100k", f"{DATA}/ml-100k"),
            ("ML-1M", "ml-1m", os.environ.get("ML1M_DIR", f"{DATA}/ml-1m"))]


def mostpop(ds):
    deg = np.zeros(ds.n_items)
    for it in ds.train_dict.values():
        for i in it: deg[i] += 1
    order = np.argsort(-deg, kind="stable"); rs = []
    for u, tr in ds.train_dict.items():
        te = ds.test_dict.get(u, [])
        if not tr or not len(te): continue
        trs = set(tr); top = []
        for i in order:
            if i not in trs:
                top.append(i)
                if len(top) == K: break
        rs.append(len(set(top) & set(te)) / len(te))
    return float(np.mean(rs))


out = {"note": __doc__.strip().split("\n\n")[0], "datasets": {}}
for name, key, root in DATASETS:
    ds = RecDataset(root, key, rating_threshold=4)
    ck = torch.load(f"{REPO}/models/{key}_best.pt", map_location="cpu", weights_only=False)
    assert (ck["n_users"], ck["n_items"]) == (ds.n_users, ds.n_items), name
    m = LightGCN(ck["n_users"], ck["n_items"], ck["args"].get("emb_dim", 64), ck["args"].get("n_layers", 3))
    m.load_state_dict(ck["model_state_dict"]); m.eval()
    v = variants(m, ds, ck["n_users"])
    res = {"mostpop_recall@20": round(mostpop(ds), 4),
           "lightgcn_standard_recall@20": round(recall_at_k(*v["vanilla"], ds), 4),
           "lightgcn_account_free_recall@20": round(recall_at_k(*v["drop_user_L0"], ds), 4)}
    print(name, res, flush=True)
    log = lambda s: print(s, flush=True)
    rng = np.random.RandomState(0); tr, ho = {}, {}
    for u, its in ds.train_dict.items():
        its = list(its)
        if len(its) >= 2:
            j = rng.randint(len(its)); ho[u] = [its[j]]; tr[u] = its[:j] + its[j + 1:]
        else:
            tr[u] = its
    os.environ["POLY_EVAL"] = "5"; P.EPOCHS = 300
    _, _, dbest, _ = P.train(key, tr, ho, ds.n_users, ds.n_items, 2, log)
    res["poly_passes"] = P.EPOCHS = dbest[2]
    pm, adj, best, users = P.train(key, ds.train_dict, ds.test_dict, ds.n_users, ds.n_items, 2, log)
    nck = torch.load(f"{REPO}/models/{key}_ngcf_3l.pt", map_location=P.DEV, weights_only=False)
    res["ngcf_original_recall@20"] = round(P.ngcf_recall(nck, ds, ds.adj_matrix.to(P.DEV), users), 4)
    pm.eval()
    with torch.no_grad():
        ue, ie = pm(adj); res["ngcf_poly_recall@20"] = round(P.recall(ue, ie, ds.train_dict, ds.test_dict, users), 4)
    res["users_scored"] = len(users)
    out["datasets"][name] = res
    print(name, res, flush=True)
json.dump(out, open(os.path.join(RESULTS, "accuracy_ml.json"), "w"), indent=2)
