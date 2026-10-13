"""
One whole request, measured end to end in Microsoft SEAL (via tenseal.sealapi), for sparse, cover and dense.

Validation chain of App. A.1: N = 8192, moduli [60,40,40,60], scale 2^40, one thread per process.
Serving operator of App. A.1: user vector u = sum over the submitted items of e-bar_i / sqrt(d_u d_i);
scores <u, e-hat_i> over the whole catalogue.
  client   key generation (secret key; the 12 power-of-two Galois keys, seeded) and, per sent chunk, a
           4096-slot vector with 1/sqrt(d_u) at the user's items (zeros elsewhere), encrypted
           symmetrically in seeded form (what a client holding the secret key sends; the full
           public-key ciphertext size is also recorded);
  server   per embedding column j (64): multiply each received chunk by its plaintext column
           e-bar_{c,j} / sqrt(d_i), add over chunks, rescale once, rotate-and-sum (12 rotations, shared
           over chunks); then, per catalogue chunk c, score_c = sum_j u_j * e-hat_{c,j}, rescale, and
           switch to the last level before returning (C ciphertexts);
  client   decrypt and decode the C score ciphertexts, rank, exclude the training items.
Item-side plaintexts are encoded once per model version (timed separately, not per request).
Regimes: sparse (own chunk set), cover (k = 10, the codebook of location_padding2.py built from every
user's stored chunk set; the client sends its cover), dense (all C chunks).
AR2023-HK runs 16 users by default (E2E_NHK).
Gowalla and Amazon-Book use the trained LightGCN of App. A.1 and report Recall@20 from the decrypted
scores against the plaintext ranking; AR2023-HK has no trained model we use (App. D), so its item
vectors are random of the same shape: its timings and sizes are exact, its accuracy is not reported.
Usage: python e2e_seal_request.py DATASET REGIME SHARD NSHARDS  (REGIME = sparse | cover | dense | interleaved;
interleaved runs all three per user back to back in one process, for paired timing)
        (one process each; merge with --merge)
Output: results/e2e_seal/<dataset>_<regime>_<shard>.json, merged into results/e2e_seal_request.json
"""
from _paths import DATA, HOMO_GCN, RESULTS
import os, sys, json, math, time, tempfile, statistics
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import tenseal.sealapi as sa

N = int(os.environ.get("E2E_NPOLY", 8192))                      # coarser packing: 16384 or 32768 (Fig. 2a ring x4)
SLOTS, EMB, SCALE, K, KCOV = N // 2, 64, 2.0 ** 40, 20, 10
NUSERS = {"Gowalla": 40, "Amazon-Book": 40, "AR2023-HK": int(os.environ.get("E2E_NHK", 16))}
if os.environ.get("E2E_N"): NUSERS = {k: int(os.environ["E2E_N"]) for k in NUSERS}
if os.environ.get("E2E_SMOKE"): NUSERS = {k: 1 for k in NUSERS}
STEPS = [2 ** j for j in range(int(math.log2(SLOTS)))]
OUTDIR = os.path.join(RESULTS, os.environ.get("E2E_OUTDIR", "e2e_seal"))


def fsz(o):
    f = tempfile.mktemp(); o.save(f); n = os.path.getsize(f); os.remove(f); return n


def load(ds):
    """item vectors (ebar folded with 1/sqrt(d_i), ehat), per-user item arrays, test sets, all chunk sets"""
    from location_padding2 import to_bits
    if ds == "AR2023-HK":
        import dataset_amazon2023_hk as H
        U, I, T, nU, nI = H.load_hk(); C = math.ceil(nI / SLOTS)
        rng = np.random.default_rng(0)
        ebar = (rng.standard_normal((nI, EMB)) * 0.05).astype(np.float64); ehat = ebar.copy()
        deg = np.bincount(I, minlength=nI).astype(float)
        order = np.argsort(U, kind="stable"); Us, Is = U[order], I[order]; off = np.searchsorted(Us, np.arange(nU + 1))
        hist = [np.unique(Is[off[u]:off[u + 1]]) for u in range(nU)]
        have = np.array([len(h) > 0 for h in hist]); users = np.flatnonzero(have)
        hist = [hist[u] for u in users]; test = [None] * len(users)
    else:
        sys.path.insert(0, f"{HOMO_GCN}")
        import torch
        from src.dataset import RecDataset
        from src.model import LightGCN
        sub = {"Gowalla": "gowalla", "Amazon-Book": "amazon-book"}[ds]
        dsx = RecDataset(f"{DATA}/{sub}", sub, rating_threshold=0)
        ck = torch.load(f"{HOMO_GCN}/models/{sub}_best.pt", map_location="cpu")
        m = LightGCN(ck["n_users"], ck["n_items"], ck["args"].get("emb_dim", 64), ck["args"].get("n_layers", 3))
        m.load_state_dict(ck["model_state_dict"]); m.eval(); L = m.n_layers; nI = ck["n_items"]
        with torch.no_grad():
            E = torch.cat([m.user_embedding.weight, m.item_embedding.weight], 0); ls = [E]
            for _ in range(L):
                E = torch.sparse.mm(dsx.adj_matrix, E); ls.append(E)
            items = [l[ck["n_users"]:] for l in ls]
            ebar = torch.stack(items[:L], 0).mean(0).numpy().astype(np.float64)
            ehat = torch.stack(items, 0).mean(0).numpy().astype(np.float64)
        deg = np.zeros(nI)
        for it in dsx.train_dict.values():
            for i in it: deg[i] += 1
        users = [u for u in sorted(dsx.train_dict) if dsx.train_dict[u]]
        hist = [np.array(sorted(dsx.train_dict[u])) for u in users]
        test = [np.array(sorted(dsx.test_dict.get(u, []))) for u in users]
        C = math.ceil(nI / SLOTS)
    inv = np.where(deg > 0, deg ** -0.5, 0.0)
    B = to_bits([sorted(set((h // SLOTS).tolist())) for h in hist], C)
    return ebar * inv[:, None], ehat, hist, test, B, C, nI


def main(ds, regime, shard, nsh):
    from location_padding2 import Codebook, cover_bits
    t0 = time.time()
    ebar_s, ehat, hist, test, B, C, nI = load(ds)
    rng = np.random.default_rng(1)
    cand = [j for j in range(len(hist)) if test[j] is None or len(test[j])]
    sample = rng.choice(np.array(cand), NUSERS[ds], replace=False)
    mine = sample[shard::nsh]
    sends = {}
    if regime in ("sparse", "interleaved"): sends["sparse"] = {j: B[j] for j in mine}
    if regime in ("dense", "interleaved"): sends["dense"] = {j: np.ones(C, dtype=bool) for j in mine}
    if regime in ("cover", "interleaved"):
        cb = Codebook(B, KCOV); g = cb.send(B[mine]); cov = cover_bits(cb, g); sends["cover"] = {j: cov[r] for r, j in enumerate(mine)}
    regs = list(sends)
    # interleaved: every user runs all three regimes back to back, the order rotating per user, so that
    # machine load affects the three alike and per-user ratios are paired
    pairs = [(j, regs[(t + i) % len(regs)]) for t, j in enumerate(mine) for i in range(len(regs))]
    send = {}
    print(f"{ds} {regime} loaded [{time.time()-t0:.0f}s]", flush=True)

    p = sa.EncryptionParameters(sa.SCHEME_TYPE.CKKS); p.set_poly_modulus_degree(N)
    p.set_coeff_modulus(sa.CoeffModulus.Create(N, [60, 40, 40, 60]))
    ctx = sa.SEALContext(p, True, sa.SEC_LEVEL_TYPE.TC128)
    ecd, ev = sa.CKKSEncoder(ctx), sa.Evaluator(ctx)
    pad = np.zeros((C * SLOTS, EMB)); pad[:nI] = ebar_s
    padh = np.zeros((C * SLOTS, EMB)); padh[:nI] = ehat
    # item side, once per model version: aggregation plaintexts (top level) for the chunks any sampled user
    # sends, and scoring plaintexts at the level after one rescale for every chunk
    need = sorted(set(int(c) for d in sends.values() for v in d.values() for c in np.flatnonzero(v)))
    te = time.perf_counter()
    agg_pt = {}
    for c in need:
        blk = pad[c * SLOTS:(c + 1) * SLOTS]
        for j in range(EMB):
            pt = sa.Plaintext(); ecd.encode(blk[:, j].tolist(), SCALE, pt); agg_pt[(c, j)] = pt
    t_agg_enc = time.perf_counter() - te
    agg_pt_bytes = int(next(iter(agg_pt.values())).coeff_count()) * 8      # in-memory size, NTT form, top level
    lvl1 = ctx.first_context_data().next_context_data().parms_id()
    lvl2 = ctx.first_context_data().next_context_data().next_context_data().parms_id()
    rows = []
    for j, reg in pairs:
        send = sends[reg]; rec = {"user": int(j), "regime": reg}
        # client: keys
        tk = time.perf_counter()
        kg = sa.KeyGenerator(ctx); sk = kg.secret_key()
        elts = [pow(3, s, 2 * N) for s in STEPS]
        gk_seeded = kg.create_galois_keys(elts); gk = sa.GaloisKeys(); kg.create_galois_keys(elts, gk)
        rec["client_keygen_s"] = time.perf_counter() - tk
        rec["galois_keys_seeded_KiB"] = fsz(gk_seeded) / 1024
        enc, dec = sa.Encryptor(ctx, sk), sa.Decryptor(ctx, sk)
        # client: encrypt
        chunks = np.flatnonzero(send[j]); h = hist[j]; w = 1.0 / math.sqrt(len(h))
        tc = time.perf_counter(); up_seeded = 0; up_full = 0; cts = []
        for c in chunks:
            v = np.zeros(SLOTS); hh = h[(h >= c * SLOTS) & (h < (c + 1) * SLOTS)] - c * SLOTS; v[hh] = w
            pt = sa.Plaintext(); ecd.encode(v.tolist(), SCALE, pt)
            s = enc.encrypt_symmetric(pt); up_seeded += fsz(s)
            ct = sa.Ciphertext(); enc.encrypt_symmetric(pt, ct); cts.append(ct)
        rec["client_encrypt_s"] = time.perf_counter() - tc
        up_full = len(chunks) * fsz(cts[0]) if cts else 0
        rec.update({"chunks_sent": int(len(chunks)), "upload_seeded_KiB": up_seeded / 1024, "upload_full_KiB": up_full / 1024})
        # server: aggregation
        ts_ = time.perf_counter(); u = []
        for jj in range(EMB):
            acc = None
            for c, ct in zip(chunks, cts):
                pr = sa.Ciphertext(); ev.multiply_plain(ct, agg_pt[(int(c), jj)], pr)
                if acc is None: acc = pr
                else: ev.add_inplace(acc, pr)
            ev.rescale_to_next_inplace(acc)
            k = 1
            while k < SLOTS:
                r = sa.Ciphertext(); ev.rotate_vector(acc, k, gk, r); ev.add_inplace(acc, r); k *= 2
            u.append(acc)
        rec["server_aggregate_s"] = time.perf_counter() - ts_
        # server: scoring (plaintexts encoded per model version; the encode is timed apart)
        ts2 = time.perf_counter(); scores = []; enc_sc = 0.0
        for c in range(C):
            blk = padh[c * SLOTS:(c + 1) * SLOTS]; acc = None
            for jj in range(EMB):
                te2 = time.perf_counter()
                pt = sa.Plaintext(); ecd.encode(blk[:, jj].tolist(), lvl1, u[jj].scale, pt)
                score_pt_bytes = int(pt.coeff_count()) * 8
                enc_sc += time.perf_counter() - te2
                pr = sa.Ciphertext(); ev.multiply_plain(u[jj], pt, pr)
                if acc is None: acc = pr
                else: ev.add_inplace(acc, pr)
            ev.rescale_to_next_inplace(acc); ev.mod_switch_to_inplace(acc, lvl2) if acc.parms_id() != lvl2 else None
            scores.append(acc)
        rec["server_score_s"] = time.perf_counter() - ts2 - enc_sc
        rec["scoring_plaintext_encode_s"] = enc_sc
        rec["download_KiB"] = len(scores) * fsz(scores[0]) / 1024
        # client: decrypt and rank
        td = time.perf_counter(); allsc = np.empty(C * SLOTS)
        for c, sc in enumerate(scores):
            pt = sa.Plaintext(); dec.decrypt(sc, pt); allsc[c * SLOTS:(c + 1) * SLOTS] = ecd.decode_double(pt)
        allsc = allsc[:nI]; allsc[h] = -np.inf
        top = np.argpartition(-allsc, K)[:K]; top = top[np.argsort(-allsc[top])]
        rec["client_decrypt_rank_s"] = time.perf_counter() - td
        uv = (ebar_s[h] * w).sum(0); plain = ehat @ uv; plain[h] = -np.inf
        ptop = np.argpartition(-plain, K)[:K]
        fin = np.isfinite(plain)
        rec["max_abs_score_err"] = float(np.max(np.abs(allsc[fin] - plain[fin])))
        rec["top20_overlap_with_plaintext"] = len(set(top.tolist()) & set(ptop.tolist())) / K
        if test[j] is not None and len(test[j]):
            t = test[j]; rec["recall20_encrypted"] = float(np.isin(top, t).sum() / min(len(t), K))
            rec["recall20_plaintext"] = float(np.isin(ptop, t).sum() / min(len(t), K))
        rows.append(rec); print(ds, regime, json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rec.items()}), flush=True)
    os.makedirs(OUTDIR, exist_ok=True)
    json.dump({"dataset": ds, "regime": regime, "C": C, "agg_plaintext_encode_s": t_agg_enc,
               "agg_plaintexts": len(agg_pt), "agg_plaintext_bytes": agg_pt_bytes, "score_plaintext_bytes": score_pt_bytes,
               "item_plaintexts_all_chunks_GiB": round(C * EMB * (agg_pt_bytes + score_pt_bytes) / 2**30, 3), "rows": rows},
              open(os.path.join(OUTDIR, f"{ds}_{regime}_{shard}.json"), "w"), indent=1)
    print("SAVED", flush=True)


def merge():
    import glob
    out = {"note": __doc__.strip().split("\n\n")[0], "datasets": {}}
    mem = {}
    for f in sorted(glob.glob(os.path.join(OUTDIR, "*.json"))):
        d = json.load(open(f))
        if d["regime"] == "interleaved": continue                       # summarised by e2e_interleaved_table.py
        if "item_plaintexts_all_chunks_GiB" in d: mem[d["dataset"]] = d["item_plaintexts_all_chunks_GiB"]
        e = out["datasets"].setdefault(d["dataset"], {"C": d["C"]}).setdefault(d["regime"], {"rows": []})
        e["rows"] += d["rows"]
    for ds, r in out["datasets"].items():
        for reg, e in r.items():
            if reg == "C": continue
            keys = [k for k in e["rows"][0] if k not in ("user", "regime")]
            e["n"] = len(e["rows"])
            e["median"] = {k: round(float(statistics.median([x[k] for x in e["rows"] if k in x])), 4) for k in keys}
            e["mean"] = {k: round(float(np.mean([x[k] for x in e["rows"] if k in x])), 4) for k in keys}
            e["p90"] = {k: round(float(np.percentile([x[k] for x in e["rows"] if k in x], 90)), 4) for k in keys}
    out["item_plaintexts_GiB"] = mem
    json.dump(out, open(os.path.join(RESULTS, os.environ.get("E2E_MERGED", "e2e_seal_request.json")), "w"), indent=2)
    print(json.dumps({ds: {reg: e.get("median") for reg, e in r.items() if reg != "C"} for ds, r in out["datasets"].items()}, indent=1))


if __name__ == "__main__":
    if sys.argv[1] == "--merge": merge()
    else: main(sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
