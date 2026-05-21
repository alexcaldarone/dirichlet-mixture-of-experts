#!/usr/bin/env bash
set -euo pipefail

for lambda_sparsity in 0.01 0.05 0.1 0.5 1.0; do
  uv run python3 train.py \
    --data-path data/dirmoe_corpus_10gb.jsonl \
    --experiment-name sparsity_${lambda_sparsity} \
    --steps 3000 \
    --seq-len 512 \
    --batch-size 32 \
    --d-model 256 \
    --num-layers 6 \
    --num-heads 8 \
    --num-kv-heads 4 \
    --d-ffn 768 \
    --router-hidden-dim 128 \
    --num-experts 8 \
    --k 1 \
    --lambda-sparsity "${lambda_sparsity}" \
    --log-every 10 \
    --save-every 1000
done
