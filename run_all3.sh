#!/usr/bin/env bash
set -euo pipefail

DATA=./data
CKPT=./checkpoint_500ep
EPOCHS=500
BS=2048
GPU=1

# (src, tgt, arch)
#   "mnist usps lenet"
#   "mnist svhn lenet"
declare -a PAIRS=(
  "cifar cifarv2 cnn_cifar"
  "cifar cifar10c cnn_cifar"
  "cifar cinic cnn_cifar"
)

declare -a SEEDS=(44 45 46)
declare -a PRIORS=("0.5 0.5" "0.3 0.7" "0.7 0.3")

for s in "${SEEDS[@]}"; do
  for p in "${PAIRS[@]}"; do
    set -- $p
    SRC=$1; TGT=$2; ARCH=$3
    for pr in "${PRIORS[@]}"; do
      set -- $pr
      SRC_P=$1; TGT_P=$2
      echo "[run] seed=$s pair=${SRC}_${TGT} src_prior=$SRC_P tgt_prior=$TGT_P arch=$ARCH"
      python run_pretrain.py "$DATA" \
        --workers 8 \
        --epochs $EPOCHS \
        --batch-size $BS \
        --gpu_id $GPU \
        --src_dataset "$SRC" \
        --tgt_dataset "$TGT" \
        --src_prior $SRC_P \
        --tgt_prior $TGT_P \
        --arch "$ARCH" \
        --seed $s \
        --checkpoint-dir "$CKPT"
    done
  done
done
