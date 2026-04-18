#!/usr/bin/env bash
set -euo pipefail

python -u main.py \
  --eval-only \
  --checkpoint checkpoints_extended/unified_model.pth \
  --train-unet-baseline-epochs 50 \
  --results-dir results_unet_baseline \
  --checkpoint-dir checkpoints_unet_baseline
