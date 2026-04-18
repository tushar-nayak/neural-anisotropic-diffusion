#!/usr/bin/env bash
set -euo pipefail

python -u main.py \
  --run-ablation-suite \
  --ablation-epochs 20 \
  --results-dir results_ablation \
  --checkpoint-dir checkpoints_ablation
