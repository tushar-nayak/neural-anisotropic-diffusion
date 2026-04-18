#!/usr/bin/env bash
set -euo pipefail

python -u main.py \
  --eval-only \
  --checkpoint checkpoints_extended/unified_model.pth \
  --noise-sweep \
  --noise-sweep-types gaussian,rician,speckle,mixed \
  --noise-sweep-sigmas 0.05,0.10,0.15,0.20 \
  --results-dir results_noise_sweep \
  --checkpoint-dir checkpoints_extended
