# HEGRec: Controlling Account Linkage in Graph-Based Recommendation under Homomorphic Encryption

Code for the anonymous submission. It measures how the chunk set of a sparse encrypted request links
the request to a stored account, tests simple defenses, and implements HEGRec's cover codebook. Each
script writes its output as JSON to `results/`.

## Layout

```
src/              LightGCN and NGCF models, data loader, evaluation, degree bucketing (padding baseline)
train_lightgcn.py LightGCN training for ML-100K, ML-1M, Gowalla and Amazon-Book
experiments/      one script per experiment (table below); _paths.py resolves all paths
results/          JSON output of the scripts
data/             datasets (not redistributed; see below)
models/           trained checkpoints (created by training)
```

Paths can be overridden with `HEGREC_HOME`, `HEGREC_DATA` and `HEGREC_RESULTS`. Long-running
scripts accept `HEGREC_NPROC` for parallel workers.

## Setup

```
pip install -r requirements.txt     # Python 3.10; tenseal only for the SEAL runs
```

Datasets (place under `data/`):

| Dataset | Source | Path |
|---|---|---|
| Gowalla, Amazon-Book | released LightGCN splits (`train.txt`, `test.txt`) | `data/gowalla/`, `data/amazon-book/` |
| ML-100K, ML-1M | GroupLens, binarized at rating >= 4, leave-one-out | `data/ml-100k/`, `data/ml-1m/` |
| AR2023-HK | Amazon Reviews 2023, Home_and_Kitchen, `benchmark/5core/rating_only` (McAuley Lab), saved as one CSV with columns `user_id,parent_asin,rating,timestamp` | `data/ar2023_hk/hk.csv` (or set `HEGREC_HK_CSV`) |

## Training

```
python train_lightgcn.py --dataset gowalla        # also amazon-book, ml-1m, ml-100k; writes models/{ds}_best.pt
python experiments/poly_ngcf_serve.py             # poly-NGCF, starts from models/{ds}_ngcf_3l.pt
```

`poly_ngcf_serve.py` fine-tunes a standard 3-layer NGCF checkpoint (`models/{ds}_ngcf_3l.pt`,
LeakyReLU, trained with BPR) into the polynomial-activation model served under CKKS.

## Experiments by table and figure

Run from the repository root, e.g. `python experiments/hk_chunkonly_stale.py`. Leakage scripts need
only the data; scripts marked † need trained models, and ‡ need TenSEAL.

| Paper element | Script(s) | Output in `results/` |
|---|---|---|
| Table 1 (AR2023-HK) and current-profile linkage | `dataset_amazon2023_hk.py` | `dataset_amazon2023_hk.json` |
| Table 2, top: profiles stored at each date | `hk_chunkonly_stale.py` | `hk_chunkonly_stale.json` |
| Table 2, bottom: requests carrying part of the history | `hk_overlap_curve.py` | `hk_overlap_curve.json` |
| Table 3: per-item vs chunked, degree- and chunk-keyed padding | `chunked_bucketed.py` | `chunked_bucketed.json` |
| Table 3: redrawn decoys | `decoy_multirequest.py` | `decoy_multirequest.json` |
| Table 3: item layout | `layout_table.py` | `layout_table.json` |
| Table 4 and Fig. 2a: decoys, covers, dense (current profiles) | `unified_comparison_hk_v2.py` | `unified_comparison_hk_v2.json` |
| Table 4: year-old profiles (rescaled by the 98.8% with a profile) | `unified_comparison_stale_hk.py` | `unified_comparison_stale_hk.json` |
| Fig. 2a: bundles | `hk_bundle_linked.py` | `hk_bundle_linked.json` |
| Fig. 2a: larger ciphertexts (ring x4) | `hk_coarse_linked.py`, then `derived_coarse_packing.py` | `coarse_packing.json` |
| Fig. 2b: partial histories, linkage and recall † | `removal_fixed_linked_ab.py`, `removal_recall_ab.py` | `removal_fixed_linked_ab.json`, `removal_recall_ab.json` |
| Fig. 2b and Table 5: cover recall † | `cover_recall_seeds_ab.py` | `cover_recall_seeds_ab.json` |
| Table 5: covers over five codebooks, built by the server | `cover_epochs_hk.py` | `cover_epochs_hk.json` |
| Table 5: covers from a public sample | `cover_epochs_public_hk.py` | `cover_epochs_public_hk.json` |
| Table 6: whole requests in SEAL †‡ | `e2e_seal_request.py`, `e2e_seal_ngcf.py`, then `derived_paired_timing.py` | `e2e_seal_request.json`, `e2e_seal_ngcf.json`, `paired_timing.json` |

SEAL runs for Table 6 and Fig. 2a (each writes per-shard files to `results/<E2E_OUTDIR>/`;
`DATASET` is `Gowalla`, `Amazon-Book` or `AR2023-HK`):

```
# score error, top-20 agreement and traffic: 200 users per dataset and shape
E2E_N=200 python experiments/e2e_seal_request.py DATASET sparse 0 1     # also cover, dense; then --merge
E2E_N=200 python experiments/e2e_seal_ngcf.py    DATASET sparse 0 1     # also cover, dense; then --merge
# paired server time: 50 users, the three shapes back to back per user
E2E_N=50 E2E_OUTDIR=e2e_seal_paired_s50      python experiments/e2e_seal_request.py DATASET interleaved 0 1
E2E_N=50 E2E_OUTDIR=e2e_seal_ngcf_paired_s50 python experiments/e2e_seal_ngcf.py    DATASET interleaved 0 1
python experiments/derived_paired_timing.py
# Fig. 2a ring x4: sparse AR2023-HK requests at each ring size
E2E_NHK=20 E2E_NPOLY=16384 E2E_OUTDIR=e2e_seal_coarse_16384 python experiments/e2e_seal_request.py AR2023-HK sparse 0 1
python experiments/derived_coarse_packing.py   # after the 8192, 16384 and 32768 runs and hk_coarse_linked.py
```

Shared modules: `location_padding2.py` (the cover codebook of Algorithm 1), `dataset_amazon2023_hk.py`
(AR2023-HK loading and linkage), `stale_profile.py`, `decoy_frontier_hk.py`, `_par.py`, `_paths.py`.

Linkage is computed from the trace each request would produce (Sec. 2.3). The SEAL runs check that
real CKKS execution produces exactly these traces. AR2023-HK scripts load the 28.2M-row CSV (about
30 s) and run in minutes to hours on one machine; `HEGREC_NPROC` parallelizes the sampled ones.
