#!/usr/bin/env bash
set -euo pipefail

GPU=0
EPOCHS=300
BS=512
WORKERS=8
OUT=./metric/lin

# (src, tgt, arch)
declare -a PAIRS=(
  "mnist usps lenet"
  "mnist svhn lenet"
)

declare -a SEEDS=(42 43 44 45 46)
declare -a PRIORS=("0.5 0.5" "0.3 0.7" "0.7 0.3")
declare -a LOSSES=("pu")

mkdir -p "$OUT"

for s in "${SEEDS[@]}"; do
  for p in "${PAIRS[@]}"; do
    set -- $p
    SRC=$1; TGT=$2; ARCH=$3
    for pr in "${PRIORS[@]}"; do
      set -- $pr
      SRC_P=$1; TGT_P=$2

      # TODO: 네가 저장한 encoder 경로 규칙으로 아래 변수를 채우세요.
      # 예) ENCODER_PTH="./checkpoint/${SRC}_${TGT}/seed${s}/syn_src${SRC_P}_tgt${TGT_P}.pth"
      ENCODER_PTH="./checkpoint/${SRC}_${TGT}/seed${s}/${ARCH}_src${SRC_P}_tgt${TGT_P}.pth"
      if [[ ! -f "$ENCODER_PTH" ]]; then
        echo "[skip encoder missing] $ENCODER_PTH"
        continue
      fi

      for L in "${LOSSES[@]}"; do
        echo "[eval] seed=$s pair=${SRC}->${TGT} priors=(${SRC_P},${TGT_P}) arch=$ARCH loss=$L"
        python evaluate.py \
          --src_dataset "$SRC" \
          --tgt_dataset "$TGT" \
          --src_prior "$SRC_P" \
          --tgt_prior "$TGT_P" \
          --arch "$ARCH" \
          --encoder "$ENCODER_PTH" \
          --gpu_id "$GPU" \
          --epochs "$EPOCHS" \
          --batch_size "$BS" \
          --val_split 0.1 \
          --patience 5 \
          --loss_type "$L" \
          --seed "$s" \
          --out_dir "$OUT" 
      done
    done
  done
done
