# HEGRec: Controlling Account Linkage in Graph-Based Recommendation under Homomorphic Encryption

Code for the anonymous submission. Each script reproduces one or more tables, figures or numbers of the
paper and writes its output as JSON to `results/`.

## Layout

```
src/              LightGCN and NGCF models, data loader, evaluation, degree bucketing
train_lightgcn.py LightGCN training for ML-100K, ML-1M, Gowalla and Amazon-Book
train_ngcf.py     standard NGCF training for the same datasets
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
| AR2023-HK | `Home_and_Kitchen.csv` from https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023/tree/main/benchmark/5core/rating_only, used unchanged (28,202,600 rows plus a header; SHA-256 `455f85aac9d8e3eb2892fca2fe370696b89ffba5f118e4d83bb21a3e0e0f448d`). Items are numbered in the file's row order. | `data/ar2023_hk/hk.csv` (or set `HEGREC_HK_CSV`) |

Recall@20 is the number of a user's test items in the top 20 over the whole catalog (training items
excluded) divided by the user's number of test items (hits/|test|), averaged over users with a test item.

## Training

```
python train_lightgcn.py --dataset gowalla        # also amazon-book, ml-1m, ml-100k -> models/{ds}_best.pt
python train_ngcf.py --dataset gowalla            # also amazon-book, ml-1m, ml-100k -> models/{ds}_ngcf_3l.pt
POLY_DEG=2 POLY_DS=gowalla POLY_EPOCHS=15 python experiments/poly_ngcf_serve.py        # poly-NGCF
POLY_DEG=2 POLY_DS=amazon-book POLY_EPOCHS=25 python experiments/poly_ngcf_serve.py    # -> results/poly_ngcf/{ds}.pt
HK_TAG=_lr5e3 HK_LR=5e-3 HK_BATCH=65536 python experiments/hk_lightgcn_train.py        # AR2023-HK LightGCN
bash experiments/hk_ngcf_final_run.sh             # AR2023-HK NGCF and poly-NGCF; four GPUs by default, HK_GPUS=1 for one
```

Reported Recall@20 is computed by `accuracy_table_eval.py` and `accuracy_ml.py`.
ML-100K and ML-1M poly-NGCF are trained inside `accuracy_ml.py` (80 and 120 passes). AR2023-HK
scripts read `results/hk_lightgcn_lr5e3/`.

## Experiments by table and figure

Run from the repository root, e.g. `python experiments/hk_chunkonly_stale.py`. Leakage scripts need
only the data; scripts marked † need trained models, and ‡ need TenSEAL. Sampled AR2023-HK runs use
`HEGREC_NQ=10000` and `NN_QUERIES=10000`, as in the paper.

| Paper element | Script(s) | Output in `results/` |
|---|---|---|
| Table 1 (AR2023-HK row), §5.2 chunks per history | `dataset_amazon2023_hk.py` | `dataset_amazon2023_hk.json` |
| Table 2 top | `hk_chunkonly_stale.py` | `hk_chunkonly_stale.json` |
| Table 2 bottom | `hk_overlap_curve.py` | `hk_overlap_curve.json` |
| Table 3 per-item, chunked, dense | `chunked_dense_endpoint.py` | `chunked_dense_endpoint.json` |
| Table 3 degree- and chunk-keyed padding | `chunked_bucketed.py` | `chunked_bucketed.json` |
| Table 3 redrawn decoys | `decoy_map_bound.py` | `decoy_map_bound.json` |
| Table 3 item layout | `layout_table.py` | `layout_table.json` |
| §5.3 padding rules | `target_input_sweep.py` | `target_input_sweep.json` |
| Table 4 and Fig. 3a, current profiles | `unified_comparison_hk_v2.py` | `unified_comparison_hk_v2.json` |
| Table 4, year-old profiles | `unified_comparison_stale_hk.py` | `unified_comparison_stale_hk.json` |
| Table 4, Table 5 and Fig. 3a traffic ‡ | `traffic_measured.py` (after `e2e_seal_request.py`, `cover_epochs_hk.py`) | `traffic_measured.json` |
| §5.4 and Fig. 3a bundles | `hk_bundle_linked.py` | `hk_bundle_linked.json` |
| §5.3-5.4 partial histories; Fig. 3b | `removal_fixed_linked_ab.py` | `removal_fixed_linked_ab.json`, `removal_fixed_linked_sparse_ab.json` |
| §5.4 partial-history recall; Fig. 3b † | `removal_recall_ab.py`, `ngcf_defence_recall.py` | `removal_recall_ab.json`, `ngcf_defence_recall.json` |
| Table 5; §5.5 | `cover_epochs_hk.py` | `cover_epochs_hk.json` |
| §5.5 deferral within a period | `cover_epochs_within_hk.py` | `cover_epochs_within_hk.json` |
| §5.5 unrecorded interactions | `cover_epochs_deviceonly_hk.py` | `cover_epochs_deviceonly_hk.json` |
| §5.5 unequal request rates | `dormant_prior_robust.py`, `rate_cut_epochs_hk.py` | `dormant_prior_robust.json`, `rate_cut_epochs_hk.json` |
| §5.5 sorted vs random grouping | `codebook_order_ablation.py` | `codebook_order_ablation.json` |
| Table 6 † | `accuracy_table_eval.py`, `accuracy_ml.py`, `hk_ngcf_final_run.sh` | `accuracy_table_eval.json`, `accuracy_ml.json`, `hk_ngcf_r_final.json` |
| §5.6, Fig. 3b (Amazon-Book, Gowalla) † | `cover_recall_seeds_ab.py`, `ngcf_defence_recall.py` | `cover_recall_seeds_ab.json`, `ngcf_defence_recall.json` |
| §5.6 (AR2023-HK) † | `hk_cover_recall.py` (`HK_SPLIT=temporal`) | `hk_cover_recall*.json` |
| Fig. 3b cover linkage † | `cover_substitution_recall_ab.py` | `cover_substitution_recall_ab.json` |
| §5.2 open world | `HEGREC_NQ=10000 SKIP_PARTIAL=1 python experiments/hk_openworld_partial.py` | `hk_openworld_partial.json` |
| Table 7, §5.7 †‡ | SEAL runs below, `derived_paired_timing.py`, `location_padding2.py`, `cover_population_cost.py`, `ngcf_cost.py` | `paired_timing_seq.json`, `cover_population_cost.json`, `ngcf_cost.json` |
| Algorithm 1 | `location_padding2.py` (`Codebook`) | `location_padding2.json` |

SEAL runs for Table 7, §5.7 and App. F (each writes per-shard files to `results/<E2E_OUTDIR>/`;
`DATASET` is `Gowalla`, `Amazon-Book` or `AR2023-HK`; the timing and size runs on AR2023-HK use random item vectors of the same shape):

```
# sizes for traffic_measured.py: 40 users (16 on AR2023-HK)
python experiments/e2e_seal_request.py DATASET sparse 0 1        # also cover, dense; then --merge
# score error and top-20 agreement: 200 users per dataset, model and request type
E2E_N=200 E2E_OUTDIR=e2e_seal_seq200 E2E_MERGED=e2e_seal_request_seq200.json python experiments/e2e_seal_request.py DATASET sparse 0 1
E2E_N=200 python experiments/e2e_seal_ngcf.py DATASET sparse 0 1  # Gowalla, Amazon-Book; also cover, dense; then --merge
# server and client time: 50 users, the three request types in turn per user, one job at a time
E2E_N=50 E2E_OUTDIR=e2e_seal_paired_s50_seq      python experiments/e2e_seal_request.py DATASET interleaved 0 1
E2E_N=50 E2E_OUTDIR=e2e_seal_ngcf_paired_s50_seq python experiments/e2e_seal_ngcf.py    DATASET interleaved 0 1
PAIRED_SUFFIX=_seq python experiments/derived_paired_timing.py
# traffic: Table 7 MiB
python experiments/traffic_measured.py && python experiments/location_padding2.py && \
  python experiments/cover_population_cost.py && python experiments/ngcf_cost.py
# score error and top-20 agreement on AR2023-HK with the trained models
bash experiments/hk_lgcn_seal_trained.sh
bash experiments/hk_ngcf_seal_trained.sh
```

Shared modules: `location_padding2.py` (the cover codebook of Algorithm 1), `dataset_amazon2023_hk.py`
(AR2023-HK loading and linkage), `stale_profile.py`, `decoy_frontier_hk.py`, `hk_chunkonly_stale.py`,
`cover_epochs_hk.py`, `serving_formulation.py`, `poly_ngcf_serve.py`, `hk_lightgcn_train.py`,
`e2e_seal_request.py`, `_par.py`, `_paths.py`.

Linkage is computed from the trace each request would produce (§3.3). AR2023-HK scripts load the 28.2M-row CSV
(about 30 s) and run in minutes to hours on one machine; `HEGREC_NPROC` parallelizes the sampled ones.
