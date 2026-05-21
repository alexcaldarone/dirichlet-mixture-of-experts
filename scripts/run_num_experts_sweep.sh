#!/usr/bin/env bash
set -euo pipefail

lambda_sparsity="${1:-0.1}"

for experts in 4 8 16; do
  uv run python3 train.py \
    --data-path data/dirmoe_corpus_10gb.jsonl \
    --experiment-name experts_${experts}_lambda_${lambda_sparsity} \
    --steps 3000 \
    --seq-len 512 \
    --batch-size 32 \
    --d-model 256 \
    --num-layers 6 \
    --num-heads 8 \
    --num-kv-heads 4 \
    --d-ffn 768 \
    --router-hidden-dim 128 \
    --num-experts "${experts}" \
    --k 1 \
    --lambda-sparsity "${lambda_sparsity}" \
    --log-every 10 \
    --save-every 1000
done
