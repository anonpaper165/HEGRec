"""Server and client time from the paired SEAL runs (Table 7 time ratios, §5.7, App. F stages).
Reads results/e2e_seal{,_ngcf}_paired_s50{PAIRED_SUFFIX}/; the paper's times are the sequential runs, PAIRED_SUFFIX=_seq.
Output: results/paired_timing{PAIRED_SUFFIX}.json
"""
from _paths import RESULTS
import os, json, glob, statistics as st

SH = ("sparse", "cover", "dense")


def rows(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(RESULTS, d, "*_interleaved_*.json"))):
        x = json.load(open(f)); e = out.setdefault(x["dataset"], {"meta": {k: v for k, v in x.items() if k != "rows"}, "rows": []})
        e["rows"] += [r for r in x["rows"] if r.get("regime") in SH]
    return out


def summarise(data, srv, cli, stage_keys):
    res = {}
    for ds, e in data.items():
        by = {}
        for r in e["rows"]: by.setdefault(r["user"], {})[r["regime"]] = r
        us = [u for u in by.values() if all(s in u for s in SH)]
        med = {s: st.median(srv(u[s]) for u in us) for s in SH}
        m = lambda s, k: round(st.median(u[s][k] for u in us), 3)
        res[ds] = {"users": len(us),
                   "server_s": {s: round(v, 3) for s, v in med.items()},
                   "server_vs_sparse": {s: round(med[s] / med["sparse"], 3) for s in ("cover", "dense")},
                   "paired_ratio_median": {s: round(st.median(srv(u[s]) / srv(u["sparse"]) for u in us), 3) for s in ("cover", "dense")},
                   "cover_vs_dense_server_saving": round(1 - med["cover"] / med["dense"], 3),
                   "client_max_s": round(max(cli(u[s]) for u in us for s in SH), 3),
                   "stages": {k: {s: m(s, k) for s in SH} for k in stage_keys if k in us[0]["sparse"]},
                   "sparse_share": {k: round(st.median(u["sparse"][k] for u in us) / med["sparse"], 3) for k in stage_keys if k.startswith("server_") and k in us[0]["sparse"]},
                   "dense_vs_sparse_stage": {k: round(st.median(u["dense"][k] for u in us) / st.median(u["sparse"][k] for u in us), 3) for k in stage_keys if k.startswith("server_") and k in us[0]["sparse"]},
                   "top20_identical": sum(u[s]["top20_overlap_with_plaintext"] == 1.0 for u in us for s in SH),
                   "requests": 3 * len(us), "meta": e["meta"]}
    return res


SUF = os.environ.get("PAIRED_SUFFIX", "")
lg = rows("e2e_seal_paired_s50" + SUF); ng = rows("e2e_seal_ngcf_paired_s50" + SUF)
out = {"note": __doc__.strip().split("\n\n")[0],
       "LightGCN": summarise(lg, lambda r: r["server_aggregate_s"] + r["server_score_s"],
                             lambda r: r["client_keygen_s"] + r["client_encrypt_s"] + r["client_decrypt_rank_s"],
                             ("client_keygen_s", "client_encrypt_s", "server_aggregate_s", "server_score_s", "client_decrypt_rank_s")),
       "NGCF": summarise(ng, lambda r: r["server_total_s"],
                         lambda r: r["client_keygen_s"] + r["client_encrypt_s"] + r["client_decrypt_rank_s"],
                         ("client_keygen_s", "client_encrypt_s", "server_aggregate_s", "server_layers_s", "server_score_s", "client_decrypt_rank_s"))}
json.dump(out, open(os.path.join(RESULTS, f"paired_timing{SUF}.json"), "w"), indent=2)
print(json.dumps({m: {ds: {k: v for k, v in r.items() if k in ("server_s", "server_vs_sparse", "paired_ratio_median", "cover_vs_dense_server_saving", "client_max_s")} for ds, r in x.items()} for m, x in out.items() if m != "note"}, indent=1))
