"""LightGCN traffic of sparse, cover (k=10) and dense requests over all users (Table 7 MiB, §5.7).
Inputs: traffic_measured.json, location_padding2.json. Output: results/cover_population_cost.json
"""
from _paths import RESULTS
import os, sys, json, math
from collections import Counter
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from location_padding2 import to_bits, Codebook, cover_bits
import stale_profile as SP

T = json.load(open(os.path.join(RESULTS, "traffic_measured.json")))
UP, KEYS, SC = T["constants"]["upload_KiB_per_chunk"], T["constants"]["keys_MiB"], T["constants"]["score_KiB_per_chunk"]


def mib(chunks, C):
    return chunks * UP / 1024 + KEYS + C * SC / 1024


out = {"note": __doc__.strip().split("\n\n")[0], "constants": T["constants"], "datasets": {}}
for name, key in (("Gowalla", "gowalla"), ("Amazon-Book", "amazon-book")):
    tr, te, M = SP.load(key); users = sorted([u for u in tr if tr[u]]); C = math.ceil(M / 4096)
    B = to_bits([sorted({i // 4096 for i in tr[u]}) for u in users], C)
    cb = Codebook(B, 10); g = cb.send(B); cov = cover_bits(cb, g)
    own, cm = float(B.sum(1).mean()), float(cov.sum(1).mean())
    out["datasets"][name] = {"C": C, "users": len(users), "mean_chunks": {"sparse": round(own, 3), "cover": round(cm, 3), "dense": C},
                             "MiB": {"sparse": round(mib(own, C), 2), "cover": round(mib(cm, C), 2), "dense": round(mib(C, C), 2)},
                             "upload_vs_sparse": {"cover": round(cm / own, 3), "dense": round(C / own, 3)},
                             "traffic_vs_sparse": {"cover": round(mib(cm, C) / mib(own, C), 3), "dense": round(mib(C, C) / mib(own, C), 3)},
                             "keys_scores_share_sparse": round((KEYS + C * SC / 1024) / mib(own, C), 3)}
lp = json.load(open(os.path.join(RESULTS, "location_padding2.json")))["datasets"]["AR2023-HK"]["upper_bound_run"]; C = 187
own = lp["mean_chunks"]; cm = own * lp["k"]["10"]["upload_vs_sparse"]
out["datasets"]["AR2023-HK"] = {"C": C, "mean_chunks": {"sparse": own, "cover": cm, "dense": C},
                                "MiB": {"sparse": round(mib(own, C), 2), "cover": round(mib(cm, C), 2), "dense": round(mib(C, C), 2)},
                                "upload_vs_sparse": {"cover": round(cm / own, 3), "dense": round(C / own, 3)},
                                "traffic_vs_sparse": {"cover": round(mib(cm, C) / mib(own, C), 3), "dense": round(mib(C, C) / mib(own, C), 3)},
                                "keys_scores_share_sparse": round((KEYS + C * SC / 1024) / mib(own, C), 3)}
json.dump(out, open(os.path.join(RESULTS, "cover_population_cost.json"), "w"), indent=2)
print(json.dumps(out["datasets"], indent=1))
