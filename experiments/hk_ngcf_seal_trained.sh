#!/bin/bash
# AR2023-HK poly-NGCF SEAL requests with the trained model (after hk_ngcf_final_run.sh), 200 users (§5.7, App. F).
# Each process needs about 32 GB; three run at once. Output: results/e2e_seal_ngcf_hk_trained.json.
set -u
cd "$(dirname "$0")"
RES=${HEGREC_RESULTS:-../results}; CK=${HK_CKPT_DIR:-$RES}
POLY=${HK_POLY:-$(python3 -c "import json; print(json.load(open('$RES/hk_ngcf_r_final.json'))['poly'])")}   # model chosen by hk_ngcf_final_run.sh
L=$RES/logs/hk_ngcf_seal_trained; mkdir -p $L
python3 e2e_seal_ngcf.py --prep-hk $CK/hk_ngcf_r_s1/ngcf.pt $POLY $RES/hk_ngcf_served.npz > $L/prep.log 2>&1 || exit 1
export E2E_HK_SERVED=$RES/hk_ngcf_served.npz E2E_OUTDIR=e2e_seal_ngcf_hk_trained E2E_MERGED=e2e_seal_ngcf_hk_trained.json OMP_NUM_THREADS=1
for s in 0 1 2; do
  python3 e2e_seal_ngcf.py AR2023-HK interleaved $s 3 > $L/interleaved_$s.log 2>&1 &
done
wait
python3 e2e_seal_ngcf.py --merge > $L/merge.log 2>&1
