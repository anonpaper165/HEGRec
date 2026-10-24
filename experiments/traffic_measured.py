"""Traffic relative to sparse from the SEAL-measured sizes (Table 4 and Table 5 traffic, Fig. 3a; §5.7).
Inputs: e2e_seal_request.json, unified_comparison_hk_v2.json, cover_epochs_hk.json. Output: results/traffic_measured.json
"""
from _paths import RESULTS
import os, json, statistics
import numpy as np

OLD_UP, OLD_KEYS, OLD_SCORE = 326.5, 8.46, 128.1
MEAN_ACTIVE = {"Gowalla": (5.886, 11), "Amazon-Book": (11.83, 23), "AR2023-HK": (8.361, 187)}


def main():
    e = json.load(open(os.path.join(RESULTS, "e2e_seal_request.json")))["datasets"]
    rows = [r for d in e.values() for reg, x in d.items() if reg != "C" for r in x["rows"]]
    up = statistics.median([r["upload_seeded_KiB"] / r["chunks_sent"] for r in rows])
    keys = statistics.median([r["galois_keys_seeded_KiB"] for r in rows]) / 1024
    score = statistics.median([r["download_KiB"] / e[d]["C"] for d in e for reg, x in e[d].items() if reg != "C" for r in x["rows"]])
    tr = lambda chunks, C: chunks * up / 1024 + keys + C * score / 1024
    out = {"note": __doc__.strip().split("\n\n")[0],
           "constants": {"upload_KiB_per_chunk": round(up, 2), "keys_MiB": round(keys, 3), "score_KiB_per_chunk": round(score, 2)},
           "per_dataset": {}}
    for ds, (a, C) in MEAN_ACTIVE.items():
        out["per_dataset"][ds] = {"sparse_MiB": round(tr(a, C), 2), "dense_MiB": round(tr(C, C), 2),
                                  "dense_over_sparse": round(tr(C, C) / tr(a, C), 3),
                                  "upload_dense_over_sparse": round(C / a, 2),
                                  "share_keys_scores_sparse": round((keys + C * score / 1024) / tr(a, C), 3)}
    u = json.load(open(os.path.join(RESULTS, "unified_comparison_hk_v2.json")))
    C = u["C"]; base = None; t5 = {}
    for m, rec in u["methods"].items():
        old = rec["traffic_MiB_per_request"][-1]
        ch = (old - OLD_KEYS - C * OLD_SCORE / 1024) / (OLD_UP / 1024)
        t5[m] = {"mean_chunks_last_request": round(ch, 3), "MiB": round(tr(ch, C), 2)}
    for m in t5: t5[m]["vs_sparse"] = round(t5[m]["MiB"] / t5["sparse"]["MiB"], 3)
    out["table_unified"] = t5
    ce = json.load(open(os.path.join(RESULTS, "cover_epochs_hk.json"))); C = ce["C"]; t6 = {}
    for k, res in ce["k"].items():
        own = [c / r for c, r in zip(res["persistent"]["mean_cover_chunks"], res["persistent"]["upload_vs_own"])]
        sp = [tr(o, C) for o in own]
        t6[k] = {"regroup": [round(tr(r * o, C) / s, 3) for r, o, s in zip(res["regroup"]["upload_vs_own"], own, sp)],
                 "persistent": [round(tr(c, C) / s, 3) for c, s in zip(res["persistent"]["mean_cover_chunks"], sp)],
                 "static": [round(tr(r * o, C) / s, 3) for r, o, s in zip(res["static"]["upload_vs_own"], own, sp)],
                 "dense": [round(tr(C, C) / s, 3) for s in sp], "own_mean_chunks": [round(o, 3) for o in own]}
    out["table_epochs"] = t6
    try:
        dv = json.load(open(os.path.join(RESULTS, "cover_epochs_deviceonly_hk.json"))); own = t6["10"]["own_mean_chunks"]
        out["deviceonly"] = {f: [round(tr(c, C) / tr(o, C), 3) for c, o in zip(r["mean_cover_chunks"], own)] for f, r in dv["f"].items()}
    except FileNotFoundError:
        pass
    json.dump(out, open(os.path.join(RESULTS, "traffic_measured.json"), "w"), indent=2)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
