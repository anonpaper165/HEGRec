"""
Covers across codebook refreshes with a codebook anyone can check (AR2023-HK, upper bound).

cover_epochs_hk.py builds every codebook from the operator's stored histories, which gives each group
exactly k members but asks clients to trust the operator's construction. Here every codebook is built
by the same public algorithm (location_padding2.Codebook) from a fixed random 50% sample S of users
(seed 0, a stand-in for a public dataset), so anyone holding S can rebuild and check it; as in
location_padding_public_outside.py we report users OUTSIDE S, whom a public dataset would not hold.
Snapshots, upper bound and linkage are those of cover_epochs_hk.py: five snapshots (quantiles 0.900,
0.925, 0.950, 0.975, end), each user with a history at the first sends one request per snapshot, the
operator knows every current history, and R linked requests are the last R snapshots;
linkage = mean 1/|class| of identical sent sequences over all users with a history (S and outside),
averaged over the users outside S.
Strategies (k = 10, 20):
  regroup     a new public codebook at every snapshot from S's histories then; each client sends the
              first cover containing its set (its key's group and the next 63), else all C;
  persistent  the groups of the first public codebook are kept; at every snapshot each group's cover is
              recomputed from its S members' histories then, and every client assigned to the group
              sends it; a client whose set no cover contained at the first codebook sends all C
              throughout; interactions of an outside client that fall outside its group's current
              cover wait (deferred), since the client cannot add chunks without leaving its group;
  dense       every chunk.
Cost: upload relative to the user's own sparse request, and total traffic with the measured sizes of
traffic_measured.json (158.7 KiB per seeded chunk, 8.009 MiB of keys, C score ciphertexts of
128.12 KiB), relative to sparse, both over users outside S.
Output: results/cover_epochs_public_hk.json
"""
from _paths import RESULTS
import os, sys, json, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from location_padding2 import to_bits, Codebook, cover_bits
import dataset_amazon2023_hk as H

KS, QS, FRAC = (10, 20), (0.900, 0.925, 0.950, 0.975, 1.0), 0.5
UP_KIB, KEYS_MIB, SCORE_KIB = 158.7, 8.009, 128.12


def classes_of(cols):
    X = np.concatenate([c.reshape(len(c), -1).view(np.uint8) for c in cols], axis=1)
    _, inv, cnt = np.unique(X, axis=0, return_inverse=True, return_counts=True)
    return cnt[inv.ravel()]


def main():
    t0 = time.time()
    U, I, T, nU, nI = H.load_hk(); C = math.ceil(nI / H.SLOTS); ch = I // H.SLOTS
    cuts = [int(np.quantile(T, q)) for q in QS]

    def lists(mask):
        key = np.unique(U[mask] * C + ch[mask]); uu = key // C; cc = key % C
        off = np.searchsorted(uu, np.arange(nU + 1))
        return [cc[off[u]:off[u + 1]].tolist() for u in range(nU)]

    B = [to_bits(lists(T <= c), C) for c in cuts]
    have = B[0].any(1); hidx = np.flatnonzero(have); B = [b[have] for b in B]; n = len(B[0]); E = len(B)
    pos = np.full(nU, -1); pos[hidx] = np.arange(n)
    S = np.sort(np.random.RandomState(0).choice(n, int(round(FRAC * n)), replace=False))
    inS = np.zeros(n, bool); inS[S] = True; out = ~inS
    own = [b.sum(1).astype(np.float64) for b in B]
    score_mib = C * SCORE_KIB / 1024

    def traffic_ratio(chunks, e):
        tr = chunks[out] * UP_KIB / 1024 + KEYS_MIB + score_mib
        sp = own[e][out] * UP_KIB / 1024 + KEYS_MIB + score_mib
        return round(float(tr.mean() / sp.mean()), 4)

    def sent_bits(cb, g):
        return cover_bits(cb, g)                                   # all-C rows for g == -1

    res = {"note": __doc__.strip().split("\n\n")[0], "users": n, "users_outside": int(out.sum()),
           "C": C, "snapshot_quantiles": QS, "sample_fraction": FRAC, "k": {}}
    print(f"loaded n={n}, outside={int(out.sum())} [{time.time()-t0:.0f}s]", flush=True)
    for k in KS:
        rk = {}
        # regroup: a new public codebook at every snapshot
        covs, sizes, fallback = [], [], []
        for e in range(E):
            cb = Codebook(B[e][S], k); g = cb.send(B[e])
            cv = sent_bits(cb, g); covs.append(np.packbits(cv, axis=1)); sizes.append(cv.sum(1))
            fallback.append(round(100 * float(np.mean(g[out] < 0)), 3))
            print(k, "regroup snapshot", e, f"[{time.time()-t0:.0f}s]", flush=True)
        cls = [classes_of(covs[E - R:]) for R in range(1, E + 1)]
        rk["regroup"] = {"linkage_R_outside": [round(float(np.mean(1.0 / c[out])), 5) for c in cls],
                         "smallest_class_outside_all_R": int(cls[-1][out].min()),
                         "sends_all_C_outside_pct": fallback,
                         "upload_vs_own_outside": [round(float(sizes[e][out].mean() / own[e][out].mean()), 4) for e in range(E)],
                         "traffic_vs_sparse_outside": [traffic_ratio(sizes[e], e) for e in range(E)]}
        print(k, "regroup", rk["regroup"], f"[{time.time()-t0:.0f}s]", flush=True)
        # persistent: groups of the first public codebook kept, covers recomputed from S members
        cb0 = Codebook(B[0][S], k); g0 = cb0.send(B[0]); G = len(cb0.covers)
        memS = cb0.group_of                                        # group of each S member (index into S)
        pcovs, psizes, defer_users = [], [], []
        for e in range(E):
            Ug = np.zeros((G, C), dtype=bool)
            np.logical_or.at(Ug, memS, B[e][S])                    # union of S members' current histories
            cv = np.ones((n, C), dtype=bool); ok = g0 >= 0; cv[ok] = Ug[g0[ok]]
            pcovs.append(np.packbits(cv, axis=1)); psizes.append(cv.sum(1))
            notin = (B[e] & ~cv).any(1)
            defer_users.append(round(100 * float(np.mean(notin[out])), 3))
            if e == E - 1: last_cv = cv
            print(k, "persistent snapshot", e, f"[{time.time()-t0:.0f}s]", flush=True)
        # interactions to date (end) of outside users deferred: chunk not in the sent cover
        mm = have[U]; r = pos[U[mm]]; keep = out[r]; r = r[keep]; c = ch[mm][keep]
        defer_int = round(100 * float(np.mean(~last_cv[r, c])), 3)
        cls = [classes_of(pcovs[E - R:]) for R in range(1, E + 1)]
        rk["persistent"] = {"linkage_R_outside": [round(float(np.mean(1.0 / c_[out])), 5) for c_ in cls],
                            "smallest_class_outside_all_R": int(cls[-1][out].min()),
                            "share_outside_in_class_below_k_pct": round(100 * float(np.mean(cls[-1][out] < k)), 3),
                            "sends_all_C_outside_pct": round(100 * float(np.mean(g0[out] < 0)), 3),
                            "users_with_deferred_outside_pct": defer_users,
                            "interactions_deferred_outside_pct_at_end": defer_int,
                            "upload_vs_own_outside": [round(float(psizes[e][out].mean() / own[e][out].mean()), 4) for e in range(E)],
                            "traffic_vs_sparse_outside": [traffic_ratio(psizes[e], e) for e in range(E)],
                            "mean_cover_chunks_outside": [round(float(psizes[e][out].mean()), 4) for e in range(E)]}
        print(k, "persistent", rk["persistent"], f"[{time.time()-t0:.0f}s]", flush=True)
        rk["dense"] = {"linkage": round(1.0 / n, 9), "traffic_vs_sparse_outside": [traffic_ratio(np.full(n, C), e) for e in range(E)]}
        res["k"][str(k)] = rk
        json.dump(res, open(os.path.join(RESULTS, "cover_epochs_public_hk.json"), "w"), indent=2)
    print("SAVED", f"[{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
