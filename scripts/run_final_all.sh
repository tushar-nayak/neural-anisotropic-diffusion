#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
LOG_DIR="${LOG_DIR:-logs/final_all_${TIMESTAMP}}"
CHECKPOINT="${CHECKPOINT:-checkpoints_extended/unified_model.pth}"

FINAL_RESULTS_DIR="${FINAL_RESULTS_DIR:-results_final}"
NOISE_SWEEP_DIR="${NOISE_SWEEP_DIR:-results_noise_sweep}"
UNET_RESULTS_DIR="${UNET_RESULTS_DIR:-results_unet_baseline}"
UNET_CHECKPOINT_DIR="${UNET_CHECKPOINT_DIR:-checkpoints_unet_baseline}"
ABLATION_RESULTS_DIR="${ABLATION_RESULTS_DIR:-results_ablation}"
ABLATION_CHECKPOINT_DIR="${ABLATION_CHECKPOINT_DIR:-checkpoints_ablation}"

UNET_EPOCHS="${UNET_EPOCHS:-50}"
ABLATION_EPOCHS="${ABLATION_EPOCHS:-20}"
NOISE_SWEEP_TYPES="${NOISE_SWEEP_TYPES:-gaussian,rician,speckle,mixed}"
NOISE_SWEEP_SIGMAS="${NOISE_SWEEP_SIGMAS:-0.05,0.10,0.15,0.20}"

mkdir -p "$LOG_DIR"

run_step() {
  local name="$1"
  shift
  echo
  echo "============================================================"
  echo "Running: $name"
  echo "Log: $LOG_DIR/${name}.log"
  echo "============================================================"
  "$@" 2>&1 | tee "$LOG_DIR/${name}.log"
}

echo "Repository: $ROOT_DIR"
echo "Logs: $LOG_DIR"
echo "Checkpoint: $CHECKPOINT"
echo
echo "Tip: to force GPU 1 only, run:"
echo "  CUDA_VISIBLE_DEVICES=1 scripts/run_final_all.sh"
echo

run_step "01_final_eval" \
  python -u main.py \
    --eval-only \
    --checkpoint "$CHECKPOINT" \
    --results-dir "$FINAL_RESULTS_DIR" \
    --checkpoint-dir checkpoints_extended

run_step "02_noise_sweep" \
  python -u main.py \
    --eval-only \
    --checkpoint "$CHECKPOINT" \
    --noise-sweep \
    --noise-sweep-types "$NOISE_SWEEP_TYPES" \
    --noise-sweep-sigmas "$NOISE_SWEEP_SIGMAS" \
    --results-dir "$NOISE_SWEEP_DIR" \
    --checkpoint-dir checkpoints_extended

run_step "03_unet_baseline" \
  python -u main.py \
    --eval-only \
    --checkpoint "$CHECKPOINT" \
    --train-unet-baseline-epochs "$UNET_EPOCHS" \
    --results-dir "$UNET_RESULTS_DIR" \
    --checkpoint-dir "$UNET_CHECKPOINT_DIR"

run_step "04_ablation_suite" \
  python -u main.py \
    --run-ablation-suite \
    --ablation-epochs "$ABLATION_EPOCHS" \
    --results-dir "$ABLATION_RESULTS_DIR" \
    --checkpoint-dir "$ABLATION_CHECKPOINT_DIR"

echo
echo "Final all-in-one run complete."
echo
echo "Primary outputs:"
echo "  $FINAL_RESULTS_DIR/"
echo "  $NOISE_SWEEP_DIR/"
echo "  $UNET_RESULTS_DIR/"
echo "  $ABLATION_RESULTS_DIR/"
echo "  $LOG_DIR/"

