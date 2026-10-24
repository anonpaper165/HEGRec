#!/bin/bash
# AR2023-HK NGCF and poly-NGCF for Table 6 (App. D.2): stage 1, poly-NGCF over batch x learning rate chosen on
# held-out items, then the test split. Output: results/hk_ngcf_r_*.json, results/hk_ngcf_r_final.json.
set -u
cd "$(dirname "$0")"
RES=${HEGREC_RESULTS:-../results}
export HK_CKPT_DIR=${HK_CKPT_DIR:-$RES}
L=$RES/logs/hk_ngcf; mkdir -p $L
HK_DEVICE=cuda:0 HK_TAG=_r_s1 HK_STAGE=1 HK_TEST=0 HK_LR=2e-4 HK_REG=1e-5 HK_EPOCHS=200 python3 hk_ngcf_train.py > $L/r_s1.log 2>&1 || exit 1
INIT=$HK_CKPT_DIR/hk_ngcf_r_s1/ngcf.pt
run() { HK_DEVICE=cuda:$1 HK_TAG=_r_s2_$2 HK_STAGE=2 HK_TEST=0 HK_BATCH=$3 HK_POLY_LR=$4 HK_POLY_EPOCHS=$5 HK_REG=1e-5 HK_INIT=$INIT \
        python3 hk_ngcf_train.py > $L/r_s2_$2.log 2>&1; }
G=${HK_GPUS:-4}   # stage-2 runs share G GPUs (HK_GPUS=1 runs them one after another on cuda:0)
i=0
for cfg in "b65536_plr1e-3 65536 1e-3 30" "b16384_plr1e-3 16384 1e-3 20" "b16384_plr5e-4 16384 5e-4 20" "b4096_plr1e-3 4096 1e-3 8"; do
  set -- $cfg; run $((i % G)) $1 $2 $3 $4 &
  i=$((i + 1)); [ $((i % G)) -eq 0 ] && wait
done
wait
best=$(RES=$RES python3 -c "
import json,os;r={}
for t in ['b65536_plr1e-3','b16384_plr1e-3','b16384_plr5e-4','b4096_plr1e-3']:
    f=os.path.join(os.environ['RES'],'hk_ngcf_r_s2_'+t+'.json')
    if os.path.exists(f):
        v=json.load(open(f)).get('poly_best_val_recall@20')
        if v is not None: r[t]=v
print(max(r,key=r.get))")
echo "poly choice $best"
HK_DEVICE=cuda:0 HK_TAG=_r_final HK_STAGE=eval HK_INIT=$INIT HK_POLY=$HK_CKPT_DIR/hk_ngcf_r_s2_$best/poly.pt python3 hk_ngcf_train.py > $L/r_final.log 2>&1
tail -n 1 $L/r_final.log
