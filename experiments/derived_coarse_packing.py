"""
Coarser packing on AR2023-HK: linkage of linked sparse requests (hk_coarse_linked.json) against the per-request
traffic of a sparse request at each ring, computed for every user as in Table 6: seeded Galois keys + mean chunk
count at the end of the data x seeded chunk size + returned score ciphertexts, all measured in SEAL
(e2e_seal_request.py with E2E_NPOLY = 8192, 16384, 32768; 20 users each). Ratios are against the ring-8192 sparse
request. Server times of those runs overlapped other jobs on the machine and are not used.
Output: results/coarse_packing.json
"""
from _paths import RESULTS
import os, json, glob, statistics as st
lk = {r["n_poly"]: r for r in json.load(open(os.path.join(RESULTS, "hk_coarse_linked.json")))["rows"]}
rows = {}
for n in (8192, 16384, 32768):
    d = json.load(open(glob.glob(os.path.join(RESULTS, f"e2e_seal_coarse_{n}", "*.json"))[0])); R = d["rows"]
    keys = st.median(r["galois_keys_seeded_KiB"] for r in R) / 1024
    chunk = st.median(r["upload_seeded_KiB"] / r["chunks_sent"] for r in R if r["chunks_sent"]) / 1024
    down = st.median(r["download_KiB"] for r in R) / 1024
    mib = keys + lk[n]["mean_chunks_end"] * chunk + down
    rows[str(n)] = {"slots": n // 2, "C": d["C"], "keys_MiB": round(keys, 2), "chunk_KiB": round(chunk * 1024, 1),
                    "scores_MiB": round(down, 2), "traffic_MiB": round(mib, 2), "R1": lk[n]["R1"], "R2": lk[n]["R2"], "R3": lk[n]["R3"]}
base = rows["8192"]["traffic_MiB"]
for r in rows.values(): r["traffic_vs_sparse"] = round(r["traffic_MiB"] / base, 3)
rows["65536"] = {"slots": 32768, "C": lk[65536]["C"], "R1": lk[65536]["R1"], "R2": lk[65536]["R2"], "R3": lk[65536]["R3"],
                 "note": "ring beyond the largest Microsoft SEAL supports at the 128-bit level; linkage only"}
out = {"note": __doc__.strip().split("\n\n")[0], "rings": rows}
json.dump(out, open(os.path.join(RESULTS, "coarse_packing.json"), "w"), indent=2)
print(json.dumps(rows, indent=1))
