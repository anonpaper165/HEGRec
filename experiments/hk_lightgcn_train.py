"""LightGCN on AR2023-HK (App. A.1). Paper run: HK_TAG=_lr5e3 HK_LR=5e-3 HK_BATCH=65536.
Output: results/hk_lightgcn{TAG}.json, results/hk_lightgcn{TAG}/ (checkpoint, served item vectors, split)
"""
from _paths import RESULTS
import os, sys, json, math, time
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_amazon2023_hk as H

DEV = os.environ.get("HK_DEVICE", "cuda:0")
EMB, L, K = 64, 3, 20
EPOCHS = int(os.environ.get("HK_EPOCHS", 60)); EVAL_EVERY = 5
BATCH = int(os.environ.get("HK_BATCH", 16384)); LR = float(os.environ.get("HK_LR", 2e-3)); REG = float(os.environ.get("HK_REG", 1e-5))
VAL_USERS = 100_000
TAG = os.environ.get("HK_TAG", "")
OUT = os.path.join(RESULTS, "hk_lightgcn" + TAG); os.makedirs(OUT, exist_ok=True)


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def split():
    U, I, T, nU, nI = H.load_hk()
    key = np.unique(U * nI + I); U, I = key // nI, key % nI
    off = np.searchsorted(U, np.arange(nU + 1))
    rng = np.random.default_rng(0)
    is_test = np.zeros(len(U), bool)
    for u in range(nU):
        a, b = off[u], off[u + 1]; n = b - a
        if n < 2: continue
        t = max(1, int(round(0.2 * n)))
        is_test[a + rng.choice(n, t, replace=False)] = True
    is_val = np.zeros(len(U), bool)
    for u in rng.choice(nU, VAL_USERS, replace=False):
        idx = np.flatnonzero(~is_test[off[u]:off[u + 1]])
        if len(idx) > 1: is_val[off[u] + rng.choice(idx)] = True
    tr = ~is_test & ~is_val
    return U, I, off, tr, is_test, is_val, nU, nI


def lists(U, I, off, mask, nU):
    c = np.cumsum(np.concatenate([[0], mask]))
    o = c[off]; Im = I[mask]
    return o, Im


def adj(Ut, It, nU, nI):
    du = np.bincount(Ut, minlength=nU).astype(np.float32); di = np.bincount(It, minlength=nI).astype(np.float32)
    w = 1.0 / np.sqrt(du[Ut] * di[It])
    r = np.concatenate([Ut, It + nU]); c = np.concatenate([It + nU, Ut]); v = np.concatenate([w, w])
    A = torch.sparse_coo_tensor(torch.tensor(np.vstack([r, c])), torch.tensor(v), (nU + nI, nU + nI)).coalesce()
    return A.to(DEV), du, di


def propagate(A, Eu, Ei):
    E = torch.cat([Eu, Ei]); ls = [E]
    for _ in range(L):
        E = torch.sparse.mm(A, E); ls.append(E)
    return ls


@torch.no_grad()
def recall(score_fn, users, tr_o, tr_I, tgt_o, tgt_I, bs=1024):
    out = []
    for b0 in range(0, len(users), bs):
        us = users[b0:b0 + bs]; S = score_fn(us)
        rows = np.concatenate([np.full(tr_o[u + 1] - tr_o[u], k) for k, u in enumerate(us)])
        cols = np.concatenate([tr_I[tr_o[u]:tr_o[u + 1]] for u in us])
        S[torch.tensor(rows, device=DEV), torch.tensor(cols, device=DEV)] = -float("inf")
        top = torch.topk(S, K, 1).indices.cpu().numpy()
        for k, u in enumerate(us):
            t = tgt_I[tgt_o[u]:tgt_o[u + 1]]
            out.append(np.isin(top[k], t).sum() / len(t))
    return float(np.mean(out))


def served_vectors(ls, nU, di):
    items = [l[nU:] for l in ls]
    ebar = torch.stack(items[:L]).mean(0); ehat = torch.stack(items).mean(0)
    inv = torch.tensor(np.where(di > 0, di ** -0.5, 0.0), dtype=torch.float32, device=DEV)
    return ebar * inv[:, None], ehat


def served_scores(ebar_s, ehat, tr_o, tr_I, du):
    def f(us):
        rows = np.concatenate([np.full(tr_o[u + 1] - tr_o[u], k) for k, u in enumerate(us)])
        cols = np.concatenate([tr_I[tr_o[u]:tr_o[u + 1]] for u in us])
        w = np.concatenate([np.full(tr_o[u + 1] - tr_o[u], 1 / math.sqrt(max(tr_o[u + 1] - tr_o[u], 1))) for u in us])
        S = torch.sparse_coo_tensor(torch.tensor(np.vstack([rows, cols])), torch.tensor(w, dtype=torch.float32),
                                    (len(us), ebar_s.shape[0])).to(DEV)
        return torch.sparse.mm(S, ebar_s) @ ehat.T
    return f


def main():
    t0 = time.time()
    U, I, off, tr, te, va, nU, nI = split()
    log(f"split: {tr.sum():,} train / {te.sum():,} test / {va.sum():,} val interactions, {nU:,} users, {nI:,} items")
    np.savez_compressed(os.path.join(OUT, "split.npz"), U=U.astype(np.int32), I=I.astype(np.int32), tr=tr, te=te, va=va)
    tr_o, tr_I = lists(U, I, off, tr, nU); te_o, te_I = lists(U, I, off, te, nU); va_o, va_I = lists(U, I, off, va, nU)
    Ut, It = U[tr], I[tr]
    A, du, di = adj(Ut, It, nU, nI)
    val_users = np.flatnonzero(np.diff(va_o) > 0); test_users = np.flatnonzero(np.diff(te_o) > 0)
    Eu = torch.nn.Parameter(torch.empty(nU, EMB, device=DEV)); Ei = torch.nn.Parameter(torch.empty(nI, EMB, device=DEV))
    torch.nn.init.xavier_uniform_(Eu); torch.nn.init.xavier_uniform_(Ei)
    opt = torch.optim.Adam([Eu, Ei], lr=LR)
    Ut_d = torch.tensor(Ut, device=DEV); It_d = torch.tensor(It, device=DEV)
    g = torch.Generator(device=DEV); g.manual_seed(0)
    hist, best = [], (-1.0, 0)
    for ep in range(1, EPOCHS + 1):
        perm = torch.randperm(len(Ut_d), device=DEV, generator=g); tot = 0.0
        for b0 in range(0, len(perm), BATCH):
            idx = perm[b0:b0 + BATCH]; u = Ut_d[idx]; p = It_d[idx]
            n = torch.randint(0, nI, (len(idx),), device=DEV, generator=g)
            ls = propagate(A, Eu, Ei); Ef = torch.stack(ls).mean(0)
            ue, pe, ne = Ef[u], Ef[nU + p], Ef[nU + n]
            loss = -torch.nn.functional.logsigmoid((ue * pe).sum(1) - (ue * ne).sum(1)).mean()
            reg = (Eu[u].pow(2).sum() + Ei[p].pow(2).sum() + Ei[n].pow(2).sum()) / (2 * len(idx))
            opt.zero_grad(); (loss + REG * reg).backward(); opt.step(); tot += loss.item() * len(idx)
        msg = f"ep {ep} loss {tot / len(perm):.4f} ({time.time() - t0:.0f}s)"
        if ep % EVAL_EVERY == 0 or ep == EPOCHS:
            with torch.no_grad():
                ls = propagate(A, Eu.detach(), Ei.detach()); eb, eh = served_vectors(ls, nU, di)
                r = recall(served_scores(eb, eh, tr_o, tr_I, du), val_users, tr_o, tr_I, va_o, va_I)
            hist.append({"epoch": ep, "val_recall@20_served": round(r, 5)}); msg += f" val served R@20 {r:.5f}"
            if r > best[0]:
                best = (r, ep); torch.save({"Eu": Eu.detach().cpu(), "Ei": Ei.detach().cpu(), "epoch": ep}, os.path.join(OUT, "ckpt.pt"))
        log(msg)
    ck = torch.load(os.path.join(OUT, "ckpt.pt"))
    with torch.no_grad():
        Eu_b, Ei_b = ck["Eu"].to(DEV), ck["Ei"].to(DEV); ls = propagate(A, Eu_b, Ei_b)
        eb, eh = served_vectors(ls, nU, di); Ef = torch.stack(ls).mean(0)
        np.save(os.path.join(OUT, "ebar_s.npy"), eb.cpu().numpy()); np.save(os.path.join(OUT, "ehat.npy"), eh.cpu().numpy())
        served = recall(served_scores(eb, eh, tr_o, tr_I, du), test_users, tr_o, tr_I, te_o, te_I)
        log(f"test served {served:.5f}")
        account = recall(lambda us: Ef[torch.tensor(us, device=DEV)] @ Ef[nU:].T, test_users, tr_o, tr_I, te_o, te_I)
        log(f"test account-embedding {account:.5f}")
        pop = torch.tensor(np.bincount(It, minlength=nI), dtype=torch.float32, device=DEV)
        mostpop = recall(lambda us: pop.expand(len(us), nI).clone(), test_users, tr_o, tr_I, te_o, te_I)
        log(f"test mostpop {mostpop:.5f}")
    out = {"note": __doc__.strip().split("\n\n")[0], "users": int(nU), "items": int(nI),
           "train": int(tr.sum()), "test": int(te.sum()), "val": int(va.sum()), "test_users": int(len(test_users)),
           "epochs": EPOCHS, "batch": BATCH, "lr": LR, "reg": REG, "best_epoch": best[1], "val_history": hist,
           "recall@20": {"served": round(served, 5), "account_embedding": round(account, 5), "mostpop": round(mostpop, 5)},
           "seconds": round(time.time() - t0)}
    json.dump(out, open(os.path.join(RESULTS, "hk_lightgcn" + TAG + ".json"), "w"), indent=2)
    log("SAVED", json.dumps(out["recall@20"]))


if __name__ == "__main__":
    main()
