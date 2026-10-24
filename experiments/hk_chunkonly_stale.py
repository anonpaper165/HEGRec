"""Linkage against profiles stored at four dates on AR2023-HK (Table 2 top; Table 10 overlap and share-with-profile rows).
Run with NN_QUERIES=10000. Output: results/hk_chunkonly_stale.json
"""
from _paths import RESULTS
import os, sys, json, time, datetime as dt
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dataset_amazon2023_hk as H
from _par import pmap


def par_rate(fn, q_idx, *args):
    parts = np.array_split(q_idx, max(1, 8 * int(os.environ.get("HEGREC_NPROC", 1))))
    parts = [p for p in parts if len(p)]
    vals = pmap(lambda p: fn(*args, p), parts)
    return float(sum(v * len(p) for v, p in zip(vals, parts)) / len(q_idx))


def reid_nn_chunk(ref_us, ref_starts, ref_ends, ref_nact, qry_tup, q_idx, nU, subset):
    acc = np.zeros(nU, dtype=np.int16)
    tot = 0.0
    for qi in q_idx:
        Q = qry_tup[qi]
        touched = []
        for c in Q:
            sl = ref_us[ref_starts[c]:ref_ends[c]]
            acc[sl] += 1
            touched.append(sl)
        cand = np.unique(np.concatenate(touched)) if touched else np.empty(0, dtype=np.int64)
        inter = acc[cand].astype(np.float64)
        union = ref_nact[cand] + len(Q) - inter
        j = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        if subset:
            j[inter < ref_nact[cand]] = -1.0
        best = j.max() if len(j) else -1.0
        pos = np.flatnonzero(cand == qi)
        if len(pos) and j[pos[0]] >= best:
            tot += 1.0 / int(np.count_nonzero(j == best))
        for sl in touched:
            acc[sl] = 0
    return tot / len(q_idx)


def main():
    U, I, T, nU, nI = H.load_hk()
    C = int(np.ceil(nI / H.SLOTS)); ch = I // H.SLOTS
    deg = np.bincount(U, minlength=nU).astype(np.int64)
    tup, uc, off, nact = H.user_chunksets(U, ch, nU, C)
    published = json.load(open(os.path.join(RESULTS, "dataset_amazon2023_hk.json")))["temporal_staleness"]
    rng = np.random.default_rng(H.SEED)
    out = {"note": __doc__.strip().split("\n\n")[0], "arms": {}}
    for frac in H.UNSEEN_TARGETS:
        t0 = time.time()
        Tcut = int(np.quantile(T, 1.0 - frac)); keep = T <= Tcut
        rU, rch = U[keep], ch[keep]
        rdeg = np.bincount(rU, minlength=nU).astype(np.int64)
        rtup, ruc, roff, rnact = H.user_chunksets(rU, rch, nU, C)
        have = np.flatnonzero(rdeg > 0)
        q_idx = np.sort(rng.choice(have, min(H.NN_QUERIES, len(have)), replace=False))
        ru, rs, re_ = H.inverted_index(ruc, roff, nU, C)
        f = len(have) / nU
        deg_filt = float("nan") if os.environ.get("SKIP_DEG") else H.reid_nn(ru, rs, re_, rnact.astype(np.int64), rdeg, tup, nact, deg, q_idx, nU)
        rna = rnact.astype(np.int64)
        nof = par_rate(lambda *a: reid_nn_chunk(*a[:-1], a[-1], nU, subset=False), q_idx, ru, rs, re_, rna, tup)
        sub = par_rate(lambda *a: reid_nn_chunk(*a[:-1], a[-1], nU, subset=True), q_idx, ru, rs, re_, rna, tup)
        key = f"unseen_{int(frac*100)}pct"
        a = {"snapshot_date": str(dt.datetime.utcfromtimestamp(Tcut / 1000).date()),
             "attackable_fraction": round(f, 6), "n_queries": int(len(q_idx)),
             "degree_filtered_sample": round(deg_filt, 6),
             "degree_filtered_population": round(deg_filt * f, 6),
             "published_population": published[key]["full_reid_nn_jaccard_population"],
             "guard_reproduces_published": (abs(deg_filt * f - published[key]["full_reid_nn_jaccard_population"]) < 5e-6) if deg_filt == deg_filt else None,
             "chunk_nofilter_sample": round(nof, 6), "chunk_nofilter_population": round(nof * f, 6),
             "chunk_subset_sample": round(sub, 6), "chunk_subset_population": round(sub * f, 6),
             "chunk_exact_population": published[key]["ch2_reid_exact"]}
        out["arms"][key] = a
        print(key, a, f"[{time.time()-t0:.0f}s]", flush=True)
    json.dump(out, open(os.path.join(RESULTS, "hk_chunkonly_stale.json"), "w"), indent=2)
    print("SAVED")


if __name__ == "__main__":
    main()
