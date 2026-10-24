"""Poly-NGCF traffic of sparse, cover (k=10) and dense requests (Table 7 MiB, §5.7, App. F).
Inputs: e2e_seal_ngcf.json, cover_population_cost.json, poly_ngcf_serve.json. Output: results/ngcf_cost.json
"""
from _paths import RESULTS
import os, json, statistics as st

J = lambda f: json.load(open(os.path.join(RESULTS, f)))
e = J("e2e_seal_ngcf.json")["datasets"]; pc = J("cover_population_cost.json")["datasets"]
rec = J("poly_ngcf_serve.json")["datasets"]
out = {"note": __doc__.strip().split("\n\n")[0], "datasets": {}}
for ds in ("Gowalla", "Amazon-Book", "AR2023-HK"):
    if ds not in e: continue
    d = e[ds]; regs = [r for r in ("sparse", "cover", "dense") if r in d]; rows = [r for reg in regs for r in d[reg]["rows"]]
    first = next(r for r in rows if "galois_keys_seeded_KiB" in r)
    up = st.median(r["upload_seeded_KiB_per_chunk"] for r in rows if "upload_seeded_KiB_per_chunk" in r)
    keys = (first["galois_keys_seeded_KiB"] + first["relin_key_seeded_KiB"]) / 1024
    sc = first["score_ct_KiB"]; Cs = d["C_score"]
    mib = lambda ch: ch * up / 1024 + keys + Cs * sc / 1024
    ch = pc[ds]["mean_chunks"]
    MiB = {r: round(mib(ch[r]), 2) for r in ("sparse", "cover", "dense")}
    srv = {r: round(st.median(x["server_total_s"] for x in d[r]["rows"]), 3) for r in regs}
    cli = max(x["client_keygen_s"] + x["client_encrypt_s"] + x["client_decrypt_rank_s"] for x in rows)
    out["datasets"][ds] = {
        "chunk_KiB": round(up, 1), "keys_MiB": round(keys, 2), "score_ct_KiB": round(sc, 1), "C_score": Cs,
        "MiB": MiB, "traffic_vs_sparse": {r: round(MiB[r] / MiB["sparse"], 3) for r in ("cover", "dense")},
        "keys_scores_share_sparse": round((keys + Cs * sc / 1024) / MiB["sparse"], 3),
        "server_s": srv, "server_vs_sparse": {r: round(srv[r] / srv["sparse"], 3) for r in regs if r != "sparse"},
        "client_max_s": round(cli, 3), "item_plaintexts_GiB": d["item_plaintexts_GiB"],
        "requests": len(rows), "top20_equal": sum(r["top20_overlap_with_plaintext"] == 1 for r in rows),
        "max_score_err": max(r["max_abs_score_err"] for r in rows),
        **({"recall@20": rec[ds]["poly_ngcf_recall@20"], "ngcf_checkpoint_recall@20": rec[ds]["ngcf_checkpoint_recall@20"]} if ds in rec else {})}
json.dump(out, open(os.path.join(RESULTS, "ngcf_cost.json"), "w"), indent=2)
print(json.dumps(out, indent=1))
