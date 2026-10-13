"""
One whole poly-NGCF request in Microsoft SEAL (via tenseal.sealapi), for sparse, cover and dense: the
nonlinear graph recommender of poly_ngcf_serve.py served under CKKS in one round, from the same request.

Model (poly_ngcf_serve.py): E_u^(0) = 0; a^(l) = sum_{i in S_u} E_i^(l) / sqrt(d_u d_i);
E_u^(l+1) = p(W1_l (E_u^(l) + a^(l)) + W2_l (E_u^(l) * a^(l)) + b_l), p(x) = c2 x^2 + c1 x + c0;
scores sum_{l=1..3} <E_u^(l), E_i^(l)>.  The server folds c2 into its plaintexts and evaluates
e' = x^2 + (c1/c2) x + c0/c2, one multiplicative level per activation.
Parameters: N = 16384 (8192 slots), moduli [44, 34 x 10, 54] (438 bits, the 128-bit budget; the 54-bit special
prime keeps key-switching noise small), every rotation done before the rescale that follows it (lazy rescaling),
scale 2^34, ten levels: aggregation 1, per layer mixing 1 + activation 1 (+ the bi-interaction product
1 from layer 2), scoring 1. One thread per process.
Request: the same chunks of 4,096 items as the LightGCN request; the client encrypts each sent chunk as
8,192 slots, item k at slots 2k and 2k+1 (1/sqrt(d_u) at the user's items), seeded, under its secret key.
Server, per layer l (user vector kept as 64 coordinates repeated with period 64 over the slots):
  aggregation  Y_j = sum_c ct_c * Q_{l,c,j} (j < 32), A = sum_j rot(Y_j, 2j) (Horner, 31 rotations),
               then a stride-64 rotate-and-sum (22 rotations): slot s holds a_(s mod 64);
  mixing       W v by the diagonal method with 8 baby and 8 giant steps (7 + 7 rotations per input);
  activation   one square, relinearised.
Scoring: score ciphertext q holds items 8192q..8192q+8191, one per slot, sum over l and t of
rot(e'_l, t) * F_{l,q,t} (baby-step rotations shared over q, 7 giant per q).
Galois keys for steps 1, 2, 8, 64 and 512 (5 keys; the stride-64 sum composes larger strides from them) and a relinearisation key, fresh per request.
Item-side plaintexts are encoded once per model version (a warm-up dense request; timed apart).
Usage: python e2e_seal_ngcf.py DATASET REGIME SHARD NSHARDS ; --merge
Output: results/<E2E_OUTDIR>/<dataset>_<regime>_<shard>.json, merged into results/e2e_seal_ngcf.json
"""
from _paths import HOMO_GCN, RESULTS
import os, sys, json, math, time, tempfile
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tenseal.sealapi as sa

N, SLOTS, CH, EMB, L, K, KCOV = 16384, 8192, 4096, 64, 3, 20, 10
DELTA = 2.0 ** 34
NUSERS = int(os.environ.get("E2E_N", 200))
OUTDIR = os.path.join(RESULTS, os.environ.get("E2E_OUTDIR", "e2e_seal_ngcf"))
STEPS = [1, 2, 8, 64, 512]                 # Galois keys; larger strides are composed from 64 and 512
s_idx = np.arange(SLOTS)


def fsz(o):
    f = tempfile.mktemp(); o.save(f); n = os.path.getsize(f); os.remove(f); return n


def load(ds):
    """poly-NGCF item layers E^(0..3) (M x 64 each), folded weights, histories, test sets, chunk sets"""
    import torch
    from location_padding2 import to_bits
    from poly_ngcf_serve import PolyNGCF, build_adj
    if ds == "AR2023-HK":
        # no trained model on AR2023-HK (as for LightGCN): item layers random with the per-layer spread of the
        # Gowalla model, weights and activation of the Gowalla model; timings and sizes are exact, accuracy is not reported
        import dataset_amazon2023_hk as H
        G = load("Gowalla")
        U, I, T, nU, nI = H.load_hk(); rng = np.random.default_rng(0)
        items = [rng.standard_normal((nI, EMB)) * G["items"][l].std() for l in range(L + 1)]
        deg = np.bincount(I, minlength=nI).astype(float)
        order = np.argsort(U, kind="stable"); Us, Is = U[order], I[order]; off = np.searchsorted(Us, np.arange(nU + 1))
        hist = [np.unique(Is[off[u]:off[u + 1]]) for u in range(nU)]; hist = [h for h in hist if len(h)]
        C = math.ceil(nI / CH)
        B = to_bits([sorted(set((h // CH).tolist())) for h in hist], C)
        return dict(items=items, W1=G["W1"], W2=G["W2"], bias=G["bias"], coef=G["coef"],
                    inv=np.where(deg > 0, deg ** -0.5, 0.0), hist=hist, test=[None] * len(hist), B=B, C=C, nI=nI)
    sys.path.insert(0, str(HOMO_GCN))
    from src.dataset import RecDataset
    sub = {"Gowalla": "gowalla", "Amazon-Book": "amazon-book"}[ds]
    dsx = RecDataset(f"{HOMO_GCN}/data/{sub}", sub, rating_threshold=0); nU, nI = dsx.n_users, dsx.n_items
    path = os.environ.get("E2E_NGCF_MODEL", os.path.join(RESULTS, "poly_ngcf", f"{sub}.pt"))
    mk = torch.load(path, map_location="cpu", weights_only=False)
    ck = torch.load(f"{HOMO_GCN}/models/{sub}_ngcf_3l.pt", map_location="cpu", weights_only=False)
    m = PolyNGCF(ck, nU, nI, mk["coef"]); m.load_state_dict(mk["state"]); m.eval()
    adj = build_adj(dsx.train_dict, nU, nI)
    with torch.no_grad():
        E = torch.cat([torch.zeros(nU, EMB), m.item]); items = [m.item.numpy().astype(np.float64)]
        for l in range(L):
            a = torch.sparse.mm(adj, E); x = m.W1[l](E + a) + m.W2[l](E * a)
            from poly_ngcf_serve import p_eval
            E = p_eval(m.coef, x); items.append(E[nU:].numpy().astype(np.float64))
    W1 = [m.W1[l].weight.detach().numpy().astype(np.float64) for l in range(L)]
    W2 = [m.W2[l].weight.detach().numpy().astype(np.float64) for l in range(L)]
    bias = [(m.W1[l].bias + m.W2[l].bias).detach().numpy().astype(np.float64) for l in range(L)]
    deg = np.zeros(nI)
    for it in dsx.train_dict.values():
        for i in it: deg[i] += 1
    users = [u for u in sorted(dsx.train_dict) if dsx.train_dict[u]]
    hist = [np.array(sorted(dsx.train_dict[u])) for u in users]
    test = [np.array(sorted(dsx.test_dict.get(u, []))) for u in users]
    C = math.ceil(nI / CH)
    B = to_bits([sorted(set((h // CH).tolist())) for h in hist], C)
    inv = np.where(deg > 0, deg ** -0.5, 0.0)
    return dict(items=items, W1=W1, W2=W2, bias=bias, coef=list(mk["coef"]), inv=inv, hist=hist, test=test,
                B=B, C=C, nI=nI)


def plain_scores(M, h):
    """the served model in float64 (no folding): reference for the decrypted scores"""
    c2, c1, c0 = M["coef"]; w = 1 / math.sqrt(len(h)); e = np.zeros(EMB); sc = 0
    for l in range(L):
        a = (M["items"][l][h] * M["inv"][h, None]).sum(0) * w
        x = M["W1"][l] @ (e + a) + M["W2"][l] @ (e * a) + M["bias"][l]
        e = c2 * x * x + c1 * x + c0
        sc = sc + M["items"][l + 1] @ e
    return sc


def main(ds, regime, shard, nsh):
    from location_padding2 import Codebook, cover_bits
    t0 = time.time()
    M = load(ds); hist, test, B, C, nI = M["hist"], M["test"], M["B"], M["C"], M["nI"]
    c2, c1, c0 = M["coef"]; beta, gamma = c1 / c2, c0 / c2
    Cs = math.ceil(nI / SLOTS)
    rng = np.random.default_rng(1)
    cand = [j for j in range(len(hist)) if test[j] is None or len(test[j])]
    sample = rng.choice(np.array(cand), NUSERS, replace=False); mine = sample[shard::nsh]
    sends = {}
    if regime in ("sparse", "interleaved"): sends["sparse"] = {j: B[j] for j in mine}
    if regime in ("dense", "interleaved"): sends["dense"] = {j: np.ones(C, dtype=bool) for j in mine}
    if regime in ("cover", "interleaved"):
        cb = Codebook(B, KCOV); g = cb.send(B[mine]); cov = cover_bits(cb, g); sends["cover"] = {j: cov[r] for r, j in enumerate(mine)}
    regs = list(sends)
    pairs = [(j, regs[(t + i) % len(regs)]) for t, j in enumerate(mine) for i in range(len(regs))]
    print(f"{ds} {regime} loaded [{time.time()-t0:.0f}s]", flush=True)

    p = sa.EncryptionParameters(sa.SCHEME_TYPE.CKKS); p.set_poly_modulus_degree(N)
    p.set_coeff_modulus(sa.CoeffModulus.Create(N, [44] + [34] * 10 + [54]))
    ctx = sa.SEALContext(p, True, sa.SEC_LEVEL_TYPE.TC128)
    ecd, ev = sa.CKKSEncoder(ctx), sa.Evaluator(ctx)
    pids = []; cd = ctx.first_context_data()
    while cd is not None: pids.append(cd.parms_id()); cd = cd.next_context_data()
    lvl = lambda ct: [i for i, q in enumerate(pids) if q == ct.parms_id()][0]
    qs = [m.value() for m in ctx.first_context_data().parms().coeff_modulus()]   # q_0 .. q_10
    drop = lambda lev: qs[len(qs) - 1 - lev]       # the prime a rescale at level lev removes

    # ---- item-side plaintext generators (values), encoded lazily at the scale the chain needs ----
    pad = [np.zeros((C * CH, EMB)) for _ in range(L)]
    for l in range(L):
        pad[l][:nI] = M["items"][l] * M["inv"][:, None] * (1.0 if l == 0 else 1.0 / c2)
    def Qvals(l, c, j):                     # aggregation plaintext: slot u -> item (u//2), column (u-2j) mod 64
        it = c * CH + (s_idx // 2) % CH; return pad[l][it, (s_idx - 2 * j) % EMB]
    def Dvals(W, g, b):                     # mixing diagonal, pre-rotated by -8g
        return W[(s_idx - 8 * g) % EMB, (s_idx + b) % EMB]
    padF = [np.zeros((Cs * SLOTS, EMB)) for _ in range(L)]
    for l in range(L): padF[l][:nI] = c2 * M["items"][l + 1]
    def Fvals(l, q, g, b):                  # scoring: slot s -> item 8192q + s, column (s + 8g + b) mod 64, pre-rotated
        s0 = (s_idx - 8 * g) % SLOTS; return padF[l][q * SLOTS + s0, (s0 + 8 * g + b) % EMB]
    cache = {}; enc_t = [0.0]
    def P(key, vals_fn, pid, scale):
        if key not in cache:
            te = time.perf_counter(); pt = sa.Plaintext(); ecd.encode(vals_fn().tolist(), pid, scale, pt)
            cache[key] = pt; enc_t[0] += time.perf_counter() - te
        return cache[key]
    def Pconst(v, pid, scale):              # constants are encoded per use (cheap)
        pt = sa.Plaintext(); ecd.encode(float(v), pid, scale, pt); return pt

    def rot(ct, k, gk):
        r = sa.Ciphertext(); ev.rotate_vector(ct, k, gk, r); return r
    def add(a, b):
        if a is None: return b
        ev.add_inplace(a, b); return a
    def at_level(ct, l):                    # out of place: the input is left as it is
        if lvl(ct) == l: return ct
        r = sa.Ciphertext(); ev.mod_switch_to(ct, pids[l], r); return r

    def aggregate(l, cts, chunks, lev, target_scale, gk):
        """a'^(l) at level lev+1 with scale target_scale (period-64 layout)"""
        src = [at_level(ct, lev) for ct in cts]
        sQ = target_scale * drop(lev) / src[0].scale if src else DELTA
        Z = None
        for j in reversed(range(32)):
            Y = None
            for c, ct in zip(chunks, src):
                pr = sa.Ciphertext(); ev.multiply_plain(ct, P(("Q", l, int(c), j), lambda: Qvals(l, int(c), j), pids[lev], sQ), pr); Y = add(Y, pr)
            Z = Y if Z is None else add(rot(Z, 2, gk), Y)
        for k in range(7):                              # stride 64 * 2^k, composed from the 64 and 512 keys
            st_, r = 64 * 2 ** k, Z
            base = 512 if st_ >= 512 else 64
            for _ in range(st_ // base): r = rot(r, base, gk)
            ev.add_inplace(Z, r)
        ev.rescale_to_next_inplace(Z)                   # rescale after the rotations (lazy): their noise stays small
        Z.scale = target_scale                          # exact by construction of sQ (guards float rounding)
        return Z

    def mix(l, ins, Ws, gk):
        """sum_k W_k v_k + bias for inputs v_k at one level; output one level down"""
        lev = lvl(ins[0]); base = ins[-1].scale
        babies = []
        for v in ins:
            bs = [v]
            for b in range(1, 8): bs.append(rot(bs[-1], 1, gk))
            babies.append(bs)
        Z = None
        for g in reversed(range(8)):
            X = None
            for k, (bs, W) in enumerate(zip(babies, Ws)):
                sc = DELTA * base / ins[k].scale
                for b in range(8):
                    pr = sa.Ciphertext(); ev.multiply_plain(bs[b], P(("D", l, k, g, b), lambda: Dvals(W, g, b), pids[lev], sc), pr); X = add(X, pr)
            Z = X if Z is None else add(rot(Z, 8, gk), X)
        ev.rescale_to_next_inplace(Z)
        bias = np.asarray(M["bias"][l])[s_idx % EMB]
        ev.add_plain_inplace(Z, P(("b", l), lambda: bias, Z.parms_id(), Z.scale))
        return Z

    def act(x, rk):
        """x^2 + beta x + gamma, one level"""
        sq = sa.Ciphertext(); ev.square(x, sq); ev.relinearize_inplace(sq, rk)
        bx = sa.Ciphertext(); ev.multiply_plain(x, Pconst(beta, x.parms_id(), x.scale), bx)
        ev.add_inplace(sq, bx); ev.rescale_to_next_inplace(sq)
        ev.add_plain_inplace(sq, Pconst(gamma, sq.parms_id(), sq.scale)); return sq

    def serve(cts, chunks, gk, rk, rec):
        t_a = 0.0; ta = time.perf_counter()
        # layer 1 (E_u^(0) = 0): x = W1 a + b
        a = aggregate(0, cts, chunks, 0, DELTA, gk); t_a += time.perf_counter() - ta
        e = act(mix(0, [a], [M["W1"][0]], gk), rk); es = [e]
        for l in (1, 2):
            ta = time.perf_counter(); a = aggregate(l, cts, chunks, lvl(e) - 1, e.scale, gk); t_a += time.perf_counter() - ta
            s = sa.Ciphertext(); ev.add(e, a, s)
            pp = sa.Ciphertext(); ev.multiply(e, a, pp); ev.relinearize_inplace(pp, rk); ev.rescale_to_next_inplace(pp)
            ev.mod_switch_to_next_inplace(s)
            e = act(mix(l, [s, pp], [c2 * M["W1"][l], c2 * c2 * M["W2"][l]], gk), rk); es.append(e)
        rec["server_aggregate_s"] = t_a
        # scoring
        ts = time.perf_counter(); top = lvl(es[-1]); T = DELTA * es[-1].scale
        babies = []
        for e in es:
            e = at_level(e, top); bs = [e]
            for b in range(1, 8): bs.append(rot(bs[-1], 1, gk))
            babies.append(bs)
        out = []
        for q in range(Cs):
            Z = None
            for g in reversed(range(8)):
                X = None
                for l in range(L):
                    sc = T / es[l].scale
                    for b in range(8):
                        pr = sa.Ciphertext(); ev.multiply_plain(babies[l][b], P(("F", l, q, g, b), lambda: Fvals(l, q, g, b), pids[top], sc), pr); X = add(X, pr)
                Z = X if Z is None else add(rot(Z, 8, gk), X)
            ev.rescale_to_next_inplace(Z); out.append(Z)
        rec["server_score_s"] = time.perf_counter() - ts
        return out

    rows = []; enc_warm = None
    for it, (j, reg) in enumerate([(pairs[0][0], "warmup")] + pairs):
        send = sends.get(reg, {j: np.ones(C, dtype=bool)}); rec = {"user": int(j), "regime": reg}
        tk = time.perf_counter()
        kg = sa.KeyGenerator(ctx); sk = kg.secret_key()
        elts = [pow(3, s, 2 * N) for s in STEPS]
        gk_seeded = kg.create_galois_keys(elts); gk = sa.GaloisKeys(); kg.create_galois_keys(elts, gk)
        rk_seeded = kg.create_relin_keys(); rk = sa.RelinKeys(); kg.create_relin_keys(rk)
        rec["client_keygen_s"] = time.perf_counter() - tk
        if it <= 1:
            rec["galois_keys_seeded_KiB"] = fsz(gk_seeded) / 1024; rec["relin_key_seeded_KiB"] = fsz(rk_seeded) / 1024
        enc, dec = sa.Encryptor(ctx, sk), sa.Decryptor(ctx, sk)
        chunks = np.flatnonzero(send[j]); h = hist[j]; w = 1.0 / math.sqrt(len(h))
        tc = time.perf_counter(); up = 0; cts = []
        for c in chunks:
            v = np.zeros(SLOTS); hh = h[(h >= c * CH) & (h < (c + 1) * CH)] - c * CH
            v[2 * hh] = w; v[2 * hh + 1] = w
            pt = sa.Plaintext(); ecd.encode(v.tolist(), DELTA, pt)
            if it <= 1: up_one = fsz(enc.encrypt_symmetric(pt))
            ct = sa.Ciphertext(); enc.encrypt_symmetric(pt, ct); cts.append(ct)
        rec["client_encrypt_s"] = time.perf_counter() - tc
        if it <= 1: rec["upload_seeded_KiB_per_chunk"] = up_one / 1024
        rec["chunks_sent"] = int(len(chunks))
        e0 = enc_t[0]; tsv = time.perf_counter(); scores = serve(cts, chunks, gk, rk, rec); enc_req = enc_t[0] - e0
        rec["server_total_s"] = time.perf_counter() - tsv
        rec["server_layers_s"] = rec["server_total_s"] - rec["server_aggregate_s"] - rec["server_score_s"]
        if reg == "warmup":
            enc_warm = enc_req
        elif enc_req > 0:
            rec["plaintext_encode_s_in_request"] = enc_req
        if it <= 1: rec["score_ct_KiB"] = fsz(scores[0]) / 1024
        td = time.perf_counter(); allsc = np.empty(Cs * SLOTS)
        for q, sc in enumerate(scores):
            pt = sa.Plaintext(); dec.decrypt(sc, pt); allsc[q * SLOTS:(q + 1) * SLOTS] = ecd.decode_double(pt)
        allsc = allsc[:nI]; allsc[h] = -np.inf
        top = np.argpartition(-allsc, K)[:K]; top = top[np.argsort(-allsc[top])]
        rec["client_decrypt_rank_s"] = time.perf_counter() - td
        plain = plain_scores(M, h); plain[h] = -np.inf; fin = np.isfinite(plain)
        ptop = np.argpartition(-plain, K)[:K]
        rec["max_abs_score_err"] = float(np.max(np.abs(allsc[fin] - plain[fin]))); rec["max_abs_score"] = float(np.max(np.abs(plain[fin])))
        rec["top20_overlap_with_plaintext"] = len(set(top.tolist()) & set(ptop.tolist())) / K
        srt = np.sort(plain[fin])[::-1]; rec["plain_gap_20_21"] = float(srt[K - 1] - srt[K])
        t = test[j]
        if t is not None:
            rec["recall20_encrypted"] = float(np.isin(top, t).sum() / min(len(t), K))
            rec["recall20_plaintext"] = float(np.isin(ptop, t).sum() / min(len(t), K))
        if reg != "warmup": rows.append(rec)
        print(ds, reg, json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    pt_bytes = sum(int(pt.coeff_count()) * 8 for pt in cache.values())
    json.dump({"dataset": ds, "regime": regime, "C": C, "C_score": Cs, "plaintext_encode_warmup_s": enc_warm,
               "item_plaintexts": len(cache), "item_plaintexts_GiB": round(pt_bytes / 2 ** 30, 3), "rows": rows},
              open(os.path.join(OUTDIR, f"{ds}_{regime}_{shard}.json"), "w"), indent=1)


def merge():
    import glob, statistics as st
    out = {"note": __doc__.strip().split("\n\n")[0], "datasets": {}}
    for f in sorted(glob.glob(os.path.join(OUTDIR, "*.json"))):
        d = json.load(open(f)); x = out["datasets"].setdefault(d["dataset"], {"C": d["C"], "C_score": d["C_score"],
                                                                             "item_plaintexts_GiB": d["item_plaintexts_GiB"]})
        for r in d["rows"]:
            x.setdefault(r["regime"], {"rows": []})["rows"].append(r)
    for ds, x in out["datasets"].items():
        for reg, v in x.items():
            if not isinstance(v, dict): continue
            rs = v["rows"]; v["n"] = len(rs)
            for k in ("client_keygen_s", "client_encrypt_s", "server_aggregate_s", "server_layers_s", "server_score_s", "server_total_s", "client_decrypt_rank_s", "chunks_sent"):
                v[f"median_{k}"] = st.median(r[k] for r in rs)
            v["top20_equal"] = sum(r["top20_overlap_with_plaintext"] == 1 for r in rs)
            v["max_score_err"] = max(r["max_abs_score_err"] for r in rs)
    json.dump(out, open(os.path.join(RESULTS, os.environ.get("E2E_MERGED", "e2e_seal_ngcf.json")), "w"), indent=1)
    print(json.dumps({ds: {r: {k: v for k, v in y.items() if k != "rows"} for r, y in x.items() if isinstance(y, dict)} for ds, x in out["datasets"].items()}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "--merge": merge()
    else: main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
