#!/usr/bin/env bash
set -euo pipefail

python -u main.py \
  --eval-only \
  --checkpoint checkpoints_extended/unified_model.pth \
  --results-dir results_extended_eval \
  --checkpoint-dir checkpoints_extended
