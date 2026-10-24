#!/bin/bash
# AR2023-HK LightGCN SEAL requests with the trained model (hk_lightgcn_train.py output), 200 users (§5.7, App. F).
# Output: results/e2e_seal_hk_lgcn_trained.json.
set -u
cd "$(dirname "$0")"
RES=${HEGREC_RESULTS:-../results}; L=$RES/logs/hk_lgcn_seal_trained; mkdir -p $L
export E2E_HK_LGCN=$RES/hk_lightgcn_lr5e3 E2E_NHK=200 E2E_OUTDIR=e2e_seal_hk_lgcn_trained E2E_MERGED=e2e_seal_hk_lgcn_trained.json OMP_NUM_THREADS=1
for r in sparse cover dense; do for s in 0 1 2 3; do
  python3 e2e_seal_request.py AR2023-HK $r $s 4 > $L/${r}_$s.log 2>&1 &
done; done
wait
python3 e2e_seal_request.py --merge > $L/merge.log 2>&1
