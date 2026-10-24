"""NGCF and poly-NGCF on AR2023-HK (Table 6, App. D.2); settings from HK_* variables, see hk_ngcf_final_run.sh.
Output: results/hk_ngcf{HK_TAG}.json, checkpoints in $HK_CKPT_DIR/hk_ngcf{HK_TAG}/ (default results/)
"""
from _paths import RESULTS, REPO
import os, sys, json, time
import numpy as np
import torch, torch.nn.functional as F
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, str(REPO))
import hk_lightgcn_train as T
import poly_ngcf_serve as P
from src.ngcf_model import NGCF

DEV = T.DEV
E = os.environ.get
STAGE = E("HK_STAGE", "both"); TEST = E("HK_TEST", "1") == "1"
EPOCHS = int(E("HK_EPOCHS", 60)); POLY_EPOCHS = int(E("HK_POLY_EPOCHS", 30))
EVAL = int(E("HK_EVAL", 2)); PATIENCE = int(E("HK_PATIENCE", 4))
BATCH = int(E("HK_BATCH", 65536)); LR = float(E("HK_LR", 1e-3)); REG = float(E("HK_REG", 1e-5)); DROPOUT = float(E("HK_DROPOUT", 0.1))
POLY_LR = float(E("HK_POLY_LR", 1e-3))
TAG = E("HK_TAG", "")
OUT = os.path.join(E("HK_CKPT_DIR", RESULTS), "hk_ngcf" + TAG); os.makedirs(OUT, exist_ok=True)
JSON = os.path.join(RESULTS, "hk_ngcf" + TAG + ".json")
log = T.log


def scorer(ue, ie):
    return lambda us: ue[torch.as_tensor(us, device=DEV)] @ ie.T


def fit(model, step, epochs, every, ckpt, save, V, t0, name, ep0=0, best=(-1.0, 0)):
    g = torch.Generator(device=DEV); g.manual_seed(ep0); hist, flat = [], 0
    for ep in range(ep0 + 1, ep0 + epochs + 1):
        model.train(); perm = torch.randperm(V["n"], device=DEV, generator=g)
        for b0 in range(0, len(perm), BATCH):
            loss = step(perm[b0:b0 + BATCH], torch.randint(0, V["nI"], (min(BATCH, len(perm) - b0),), device=DEV, generator=g))
        if ep % every and ep != ep0 + epochs:
            continue
        model.eval()
        with torch.no_grad():
            ue, ie = model(V["A"])[:2]; r = T.recall(scorer(ue, ie), V["val_users"], *V["val"])
        hist.append({"epoch": ep, "val_recall@20": round(r, 5)})
        log(f"{name} ep {ep} loss {loss:.4f} val R@20 {r:.5f} ({time.time() - t0:.0f}s)")
        if r > best[0]:
            best, flat = (r, ep), 0; torch.save(save(), ckpt)
        else:
            flat += 1
            if flat >= PATIENCE: break
    return best, hist


def main():
    t0 = time.time()
    z = np.load(os.path.join(RESULTS, "hk_lightgcn_lr5e3", "split.npz"))
    U, I, tr, te, va = z["U"].astype(np.int64), z["I"].astype(np.int64), z["tr"], z["te"], z["va"]
    nU, nI = int(U.max()) + 1, int(I.max()) + 1
    off = np.searchsorted(U, np.arange(nU + 1))
    tr_o, tr_I = T.lists(U, I, off, tr, nU); te_o, te_I = T.lists(U, I, off, te, nU); va_o, va_I = T.lists(U, I, off, va, nU)
    Ut, It = U[tr], I[tr]; A, du, di = T.adj(Ut, It, nU, nI)
    val_users = np.flatnonzero(np.diff(va_o) > 0); test_users = np.flatnonzero(np.diff(te_o) > 0)
    log(f"{nU:,} users, {nI:,} items, {tr.sum():,} train, {len(test_users):,} test users")
    Ut_d = torch.tensor(Ut, device=DEV); It_d = torch.tensor(It, device=DEV)
    V = {"A": A, "n": len(Ut_d), "nI": nI, "val_users": val_users, "val": (tr_o, tr_I, va_o, va_I)}
    test = lambda ue, ie: round(T.recall(scorer(ue, ie), test_users, tr_o, tr_I, te_o, te_I), 5)
    res = json.load(open(JSON)) if os.path.exists(JSON) else {"note": __doc__.strip().split("\n\n")[0]}
    res.update({"batch": BATCH, "eval_every": EVAL, "patience": PATIENCE})
    torch.manual_seed(0)

    if STAGE in ("1", "both"):
        m = NGCF(nU, nI, 64, 3, dropout=DROPOUT).to(DEV); ep0, best0 = 0, (-1.0, 0)
        if E("HK_RESUME"):
            rk = torch.load(E("HK_RESUME"), map_location=DEV, weights_only=False); m.load_state_dict(rk["model_state_dict"])
            ep0 = int(E("HK_RESUME_EPOCH")); best0 = (float(E("HK_RESUME_VAL")), ep0); res["resumed_from"] = [E("HK_RESUME"), ep0]
        opt = torch.optim.Adam(m.parameters(), lr=LR)
        def step(idx, n):
            ue, ie = m(A); bpr, reg = m.bpr_loss(ue, ie, Ut_d[idx], It_d[idx], n)
            opt.zero_grad(); (bpr + REG * reg).backward(); opt.step(); return bpr.item()
        ck_path = os.path.join(OUT, "ngcf.pt")
        if E("HK_RESUME"):
            torch.save({"model_state_dict": m.state_dict()}, ck_path)
        best, hist = fit(m, step, EPOCHS, EVAL, ck_path, lambda: {"model_state_dict": m.state_dict()}, V, t0, "ngcf", ep0, best0)
        res.update({"lr": LR, "reg": REG, "dropout": DROPOUT, "ngcf_best_epoch": best[1],
                    "ngcf_best_val_recall@20": round(best[0], 5), "ngcf_val_history": hist})
        if TEST:
            m.load_state_dict(torch.load(ck_path, map_location=DEV, weights_only=False)["model_state_dict"]); m.eval()
            with torch.no_grad():
                res["ngcf_original_recall@20"] = test(*m(A))
        json.dump(res, open(JSON, "w"), indent=2); del m, opt; torch.cuda.empty_cache()

    if STAGE in ("2", "both"):
        init = E("HK_INIT", os.path.join(OUT, "ngcf.pt"))
        ck = torch.load(init, map_location=DEV, weights_only=False)
        pm = P.PolyNGCF(ck, nU, nI, P.poly_fit(2)).to(DEV); opt = torch.optim.Adam(pm.parameters(), lr=POLY_LR)
        def pstep(idx, n):
            u, p = Ut_d[idx], It_d[idx]; ue, ie, pen = pm(A, collect=True)
            reg = (pm.item[p].pow(2).sum() + pm.item[n].pow(2).sum()) / (2 * len(idx))
            loss = -F.logsigmoid((ue[u] * ie[p]).sum(1) - (ue[u] * ie[n]).sum(1)).mean() + REG * reg + P.LAM * pen
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(pm.parameters(), 5.0); opt.step()
            return loss.item()
        pk = os.path.join(OUT, "poly.pt")
        best, hist = fit(pm, pstep, POLY_EPOCHS, 1, pk, lambda: {"state": pm.state_dict(), "coef": pm.coef}, V, t0, "poly")
        res.update({"init": init, "poly_lr": POLY_LR, "poly_passes": best[1], "poly_best_val_recall@20": round(best[0], 5),
                    "poly_val_history": hist})
        if TEST:
            pm.load_state_dict(torch.load(pk, map_location=DEV, weights_only=False)["state"]); pm.eval()
            with torch.no_grad():
                res["ngcf_poly_recall@20"] = test(*pm(A)[:2])
    if STAGE == "eval":
        ck = torch.load(E("HK_INIT"), map_location=DEV, weights_only=False)
        m = NGCF(nU, nI, 64, 3, dropout=DROPOUT).to(DEV); m.load_state_dict(ck["model_state_dict"]); m.eval()
        with torch.no_grad():
            res["ngcf_original_recall@20"] = test(*m(A))
        del m; torch.cuda.empty_cache()
        pm = P.PolyNGCF(ck, nU, nI, P.poly_fit(2)).to(DEV)
        pm.load_state_dict(torch.load(E("HK_POLY"), map_location=DEV, weights_only=False)["state"]); pm.eval()
        with torch.no_grad():
            res["ngcf_poly_recall@20"] = test(*pm(A)[:2])
        res.update({"init": E("HK_INIT"), "poly": E("HK_POLY")})
        log(f"test: NGCF {res['ngcf_original_recall@20']}, poly-NGCF {res['ngcf_poly_recall@20']}")
    res["seconds"] = round(time.time() - t0)
    json.dump(res, open(JSON, "w"), indent=2)


if __name__ == "__main__":
    main()
