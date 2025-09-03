#!/usr/bin/env bash
set -euo pipefail

GPU=0
EPOCHS=300
BS=512
OUT=./metric/st

# (src, tgt, arch)
declare -a PAIRS=(
  "cifar cifarv2 cnn_cifar"
  "cifar cifar10c cnn_cifar"
  "cifar cinic cnn_cifar"
)

declare -a SEEDS=(42 43 44 45 46)
declare -a PRIORS=("0.5 0.5" "0.3 0.7" "0.7 0.3")

# run_robust의 --loss 선택지
declare -a LOSSES=("bce" "nnpu")

mkdir -p "$OUT"

for s in "${SEEDS[@]}"; do
  for p in "${PAIRS[@]}"; do
    set -- $p
    SRC=$1; TGT=$2; ARCH=$3
    for pr in "${PRIORS[@]}"; do
      set -- $pr
      SRC_P=$1; TGT_P=$2

      # encoder 경로 규칙은 eval_all.sh와 동일하게 맞춤
      ENCODER_PTH="./checkpoint_500ep/${SRC}_${TGT}/seed${s}/${ARCH}_src${SRC_P}_tgt${TGT_P}.pth"
      if [[ ! -f "$ENCODER_PTH" ]]; then
        echo "[skip encoder missing] $ENCODER_PTH"
        continue
      fi

      for L in "${LOSSES[@]}"; do
        echo "[robust] seed=$s pair=${SRC}->${TGT} priors=(${SRC_P},${TGT_P}) arch=$ARCH"
        python run_robust.py \
          --train_dataset "$SRC" \
          --test_dataset "$TGT" \
          --src_prior "$SRC_P" \
          --tgt_prior "$TGT_P" \
          --arch "$ARCH" \
          --encoder "$ENCODER_PTH" \
          --gpu_id "$GPU" \
          --epochs "$EPOCHS" \
          --batch_size "$BS" \
          --batch_size_val "$BS" \
          --val_split 0.1 \
          --patience 999 \
          --seed "$s" \
          --output_dir "$OUT"
      done
    done
  done
done
