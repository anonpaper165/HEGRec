"""
Poly-NGCF: the second graph recommender we serve under CKKS.

NGCF (Wang et al., SIGIR 2019) with the two changes encrypted serving needs:
  * the activation is a polynomial p(x) (least-squares fit of LeakyReLU(0.2) on [-B, B], degree DEG),
    evaluated exactly as the encrypted operator does; a soft range penalty
    LAM * mean(relu(|x| - B)^2) keeps pre-activations where the fit holds; no L2 normalisation;
  * the user side has no ID embedding (E_u^(0) = 0), as for our LightGCN operator: the server cannot look
    up the account's vector without naming it. The user's layers are then computed from the submitted
    history alone,
        a_u^(l)   = sum_{i in S_u} E_i^(l) / sqrt(d_u d_i)
        E_u^(l+1) = p( W1_l (E_u^(l) + a_u^(l)) + W2_l (E_u^(l) * a_u^(l)) + b_l ),
    and the score is sum_l <E_u^(l), E_i^(l)> over l = 1..L. Training uses the same recursion, so the
    served model is the trained model (no self-term to drop).
Initialised from the NGCF checkpoint (items and weights), trained with BPR, Adam 1e-3, batch 4096, L2 1e-5.
Recall@20 over every test user of the released LightGCN splits, for poly-NGCF and for the NGCF checkpoint.

  POLY_DEV=1   select the activation degree: hold out one training item per user (users with >= 2), train
               DEG = 2 and 3, report hold-out Recall@20 only (test split untouched).
Output: results/poly_ngcf_serve.json, models in results/poly_ngcf/{ds}.pt (item layers, W, b, p)
"""
from _paths import HOMO_GCN, RESULTS
import sys, os, json, time, math
import numpy as np
sys.path.insert(0, str(HOMO_GCN))
import torch, torch.nn.functional as F
from src.dataset import RecDataset

K, L, D = 20, 3, 64
DEV = os.environ.get("POLY_DEVICE", "cuda:0")
B, LAM = 4.0, 0.05
EPOCHS = int(os.environ.get("POLY_EPOCHS", 300))


def poly_fit(deg):
    xs = np.linspace(-B, B, 4001)
    return [float(c) for c in np.polyfit(xs, np.where(xs >= 0, xs, 0.2 * xs), deg)]  # highest power first


def p_eval(c, x):
    y = torch.zeros_like(x)
    for a in c:
        y = y * x + a
    return y


def build_adj(train, nU, nI):
    import scipy.sparse as sp
    r = np.array([u for u in train for _ in train[u]]); c = np.array([i for u in train for i in train[u]])
    A = sp.coo_matrix((np.ones(len(r)), (r, nU + c)), shape=(nU + nI, nU + nI)); A = (A + A.T).tocsr()
    d = np.asarray(A.sum(1)).ravel(); di = np.where(d > 0, d ** -0.5, 0)
    A = sp.diags(di) @ A @ sp.diags(di); A = A.tocoo()
    return torch.sparse_coo_tensor(np.vstack([A.row, A.col]), A.data, A.shape, dtype=torch.float32).coalesce()


class PolyNGCF(torch.nn.Module):
    def __init__(self, ck, nU, nI, coef):
        super().__init__()
        st = ck["model_state_dict"]; self.nU = nU; self.coef = coef
        self.item = torch.nn.Parameter(st["item_embedding.weight"].clone().float())
        self.W1 = torch.nn.ModuleList([torch.nn.Linear(D, D) for _ in range(L)])
        self.W2 = torch.nn.ModuleList([torch.nn.Linear(D, D) for _ in range(L)])
        for l in range(L):
            for nm, mod in (("W1", self.W1[l]), ("W2", self.W2[l])):
                mod.weight.data.copy_(st[f"{nm}.{l}.weight"]); mod.bias.data.copy_(st[f"{nm}.{l}.bias"])

    def forward(self, adj, collect=False):
        E = torch.cat([torch.zeros(self.nU, D, device=self.item.device), self.item]); out = []; pen = 0.0
        for l in range(L):
            a = torch.sparse.mm(adj, E); x = self.W1[l](E + a) + self.W2[l](E * a)
            if collect: pen = pen + (F.relu(x.abs() - B) ** 2).mean()
            E = p_eval(self.coef, x); out.append(E)
        Ef = torch.cat(out, 1)
        return (Ef[:self.nU], Ef[self.nU:], pen) if collect else (Ef[:self.nU], Ef[self.nU:])


@torch.no_grad()
def recall(ue, ie, train, test, users):
    rs = []
    for j0 in range(0, len(users), 4096):
        us = users[j0:j0 + 4096]; S = ue[us] @ ie.T
        for r, u in enumerate(us): S[r, train[u]] = -1e9
        top = torch.topk(S, K, 1).indices.cpu().numpy()
        rs += [len(set(top[r]) & set(test[u])) / min(len(test[u]), K) for r, u in enumerate(us)]
    return float(np.mean(rs))


@torch.no_grad()
def ngcf_recall(ck, ds, adj, users):
    from src.ngcf_model import NGCF
    m = NGCF(ds.n_users, ds.n_items, emb_dim=ck["emb_dim"], n_layers=ck["n_layers"]).to(DEV)
    m.load_state_dict(ck["model_state_dict"]); m.eval(); ue, ie = m(adj)
    return recall(ue, ie, ds.train_dict, ds.test_dict, users)


def train(ds_key, train_d, test_d, nU, nI, deg, log):
    ck = torch.load(f"{HOMO_GCN}/models/{ds_key}_ngcf_3l.pt", map_location=DEV, weights_only=False)
    torch.manual_seed(42); rng = np.random.RandomState(42)
    adj = build_adj(train_d, nU, nI).to(DEV); m = PolyNGCF(ck, nU, nI, poly_fit(deg)).to(DEV)
    tu = np.array([u for u in train_d for _ in train_d[u]]); ti = np.array([i for u in train_d for i in train_d[u]])
    pos = {u: set(v) for u, v in train_d.items()}
    opt = torch.optim.Adam(m.parameters(), lr=1e-3); users = [u for u in test_d if test_d[u] and train_d.get(u)]
    bs = 4096; t0 = time.time(); best = (-1, None, 0)
    for ep in range(1, EPOCHS + 1):
        m.train(); perm = rng.permutation(len(tu))
        for s in range(0, len(perm), bs):
            idx = perm[s:s + bs]; u = tu[idx]; neg = rng.randint(0, nI, len(idx))
            ue, ie, pen = m(adj, collect=True)
            us, ps, ns = ue[u], ie[ti[idx]], ie[neg]
            reg = (m.item[ti[idx]].pow(2).sum() + m.item[neg].pow(2).sum()) / (2 * len(idx))
            loss = -F.logsigmoid((us * ps).sum(1) - (us * ns).sum(1)).mean() + 1e-5 * reg + LAM * pen
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
        if ep % int(os.environ.get("POLY_EVAL", 20)) == 0 or ep == EPOCHS:
            m.eval(); ue, ie = m(adj); r = recall(ue, ie, train_d, test_d, users)
            log(f"  {ds_key} deg{deg} ep{ep} loss {loss.item():.4f} pen {float(pen):.4f} recall {r:.4f} ({time.time()-t0:.0f}s)")
            if r > best[0]: best = (r, {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}, ep)
    return m, adj, best, users


def main():
    names = {"gowalla": "Gowalla", "amazon-book": "Amazon-Book"}
    dss = os.environ.get("POLY_DS", "gowalla,amazon-book").split(",")
    dev = os.environ.get("POLY_DEV") == "1"
    outp = os.path.join(RESULTS, "poly_ngcf_select.json" if dev else "poly_ngcf_serve.json")
    out = json.load(open(outp)) if os.path.exists(outp) else {"note": __doc__.strip().split("\n\n")[0], "datasets": {}}
    log = lambda s: print(s, flush=True)
    for key in dss:
        ds = RecDataset(f"{HOMO_GCN}/data/{key}", key, rating_threshold=0); nU, nI = ds.n_users, ds.n_items
        if dev:
            rng = np.random.RandomState(0); tr, ho = {}, {}
            for u, its in ds.train_dict.items():
                if len(its) >= 2:
                    j = rng.randint(len(its)); ho[u] = [its[j]]; tr[u] = its[:j] + its[j + 1:]
                else:
                    tr[u] = list(its)
            res = {}
            for deg in (2, 3):
                _, _, best, _ = train(key, tr, ho, nU, nI, deg, log)
                res[f"deg{deg}"] = {"holdout_recall@20": round(best[0], 4), "best_epoch": best[2]}
            out["datasets"][names[key]] = res
        else:
            deg = int(os.environ.get("POLY_DEG", 3))
            m, adj, best, users = train(key, ds.train_dict, ds.test_dict, nU, nI, deg, log)
            ck = torch.load(f"{HOMO_GCN}/models/{key}_ngcf_3l.pt", map_location=DEV, weights_only=False)
            r0 = ngcf_recall(ck, ds, ds.adj_matrix.to(DEV), users)
            m.eval()  # the model after EPOCHS (chosen on the hold-out run), not the best test epoch
            with torch.no_grad():
                ue, ie = m(adj); r = recall(ue, ie, ds.train_dict, ds.test_dict, users)
            os.makedirs(os.path.join(RESULTS, "poly_ngcf"), exist_ok=True)
            torch.save({"state": {k: v.cpu() for k, v in m.state_dict().items()}, "coef": m.coef, "B": B, "deg": deg, "epoch": EPOCHS},
                       os.path.join(RESULTS, "poly_ngcf", f"{key}.pt"))
            out["datasets"][names[key]] = {"users": len(users), "deg": deg, "coef": m.coef, "B": B, "epochs": EPOCHS,
                                           "poly_ngcf_recall@20": round(r, 4), "ngcf_checkpoint_recall@20": round(r0, 4)}
        log(json.dumps(out["datasets"][names[key]]))
        cur = json.load(open(outp)) if os.path.exists(outp) else out          # other datasets may run concurrently
        cur["datasets"][names[key]] = out["datasets"][names[key]]; json.dump(cur, open(outp, "w"), indent=2)


if __name__ == "__main__":
    main()
