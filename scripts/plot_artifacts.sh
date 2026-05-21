#!/usr/bin/env bash
set -euo pipefail

output_dir="${1:-artifacts/plots}"

uv run python3 plot_results.py \
  --artifacts-dir artifacts \
  --output-dir "${output_dir}"
