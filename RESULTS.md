# Results

This file summarizes the committed experiments for the learned neural anisotropic diffusion denoiser. The project is evaluated as an MRI denoising benchmark, not as a tumor classifier.

## Dataset

- Dataset: Br35H brain tumor MRI slices
- Images used by the loader: `3000`
- Split: `2100` train / `450` validation / `450` test
- Training corruption: synthetic Rician-style noise in the extended run
- Metrics: PSNR and SSIM

## Final Evaluation

The final no-retrain checkpoint evaluation is stored in [`results_final/unified_comparison_table.csv`](https://github.com/tushar-nayak/neural-anisotropic-diffusion/blob/extended/results_final/unified_comparison_table.csv).

| Method | PSNR (dB) | PSNR Std | SSIM | SSIM Std | Edge MSE |
| --- | ---: | ---: | ---: | ---: | ---: |
| Noisy Input | 17.711 | 3.370 | 0.425 | 0.141 | 0.194 |
| Gaussian Smoothing | 19.011 | 2.855 | 0.552 | 0.136 | 0.185 |
| Median Filter | 19.538 | 3.106 | 0.521 | 0.138 | 0.136 |
| Bilateral Filter | 18.955 | 3.751 | 0.463 | 0.157 | 0.136 |
| Non-Local Means | 20.393 | 3.714 | 0.585 | 0.152 | 0.087 |
| Wavelet Denoising | 19.519 | 3.394 | 0.530 | 0.140 | 0.106 |
| Skimage TV | 19.482 | 3.047 | 0.570 | 0.140 | 0.134 |
| Curvature Flow (16 iter) | 19.985 | 3.325 | 0.560 | 0.140 | 0.113 |
| Classical PM (16 iter) | 19.635 | 3.784 | 0.539 | 0.166 | 0.106 |
| Unified Neural PDE (Ours) | 24.853 | 2.120 | 0.719 | 0.111 | 0.056 |

The strongest classical baseline in this table is Non-Local Means. The learned diffusion model improves over it by:

| Comparison | PSNR Gain | SSIM Gain |
| --- | ---: | ---: |
| Ours vs Non-Local Means | +4.460 dB | +0.134 |

The PSNR/SSIM bar chart is stored in [`results_final/unified_metric_bars.png`](https://github.com/tushar-nayak/neural-anisotropic-diffusion/blob/extended/results_final/unified_metric_bars.png).

![PSNR and SSIM bar charts](https://raw.githubusercontent.com/tushar-nayak/neural-anisotropic-diffusion/extended/results_final/unified_metric_bars.png)

## Neural Baseline

The final U-Net baseline comparison is stored in [`results_unet_baseline/unified_comparison_table.csv`](https://github.com/tushar-nayak/neural-anisotropic-diffusion/blob/extended/results_unet_baseline/unified_comparison_table.csv).

| Method | PSNR (dB) | SSIM | Edge MSE |
| --- | ---: | ---: | ---: |
| Plain U-Net Baseline | 25.875 | 0.759 | 0.053 |
| Unified Neural PDE (Ours) | 24.853 | 0.719 | 0.056 |

The plain U-Net is the strongest overall denoiser in the final run. This gives the project a more honest conclusion: the learned PDE model is very strong against classical denoisers and more interpretable as an iterative diffusion process, but a generic supervised U-Net achieves better raw denoising metrics on this split.

## Noise Robustness Sweep

The noise sweep is stored in [`results_noise_sweep/unified_noise_sweep.csv`](https://github.com/tushar-nayak/neural-anisotropic-diffusion/blob/extended/results_noise_sweep/unified_noise_sweep.csv).

| Noise Type | Sigma | PSNR (dB) | SSIM | Edge MSE |
| --- | ---: | ---: | ---: | ---: |
| Gaussian | 0.05 | 26.376 | 0.769 | 0.028 |
| Gaussian | 0.10 | 24.567 | 0.728 | 0.043 |
| Gaussian | 0.15 | 22.655 | 0.679 | 0.064 |
| Gaussian | 0.20 | 21.063 | 0.633 | 0.090 |
| Rician | 0.05 | 26.669 | 0.784 | 0.028 |
| Rician | 0.10 | 25.536 | 0.752 | 0.041 |
| Rician | 0.15 | 24.513 | 0.713 | 0.063 |
| Rician | 0.20 | 22.094 | 0.569 | 0.102 |
| Speckle | 0.05 | 27.026 | 0.784 | 0.024 |
| Speckle | 0.10 | 26.627 | 0.773 | 0.028 |
| Speckle | 0.15 | 26.002 | 0.758 | 0.034 |
| Speckle | 0.20 | 25.190 | 0.741 | 0.041 |
| Mixed | 0.05 | 26.699 | 0.780 | 0.027 |
| Mixed | 0.10 | 25.600 | 0.754 | 0.037 |
| Mixed | 0.15 | 24.407 | 0.719 | 0.054 |
| Mixed | 0.20 | 22.765 | 0.648 | 0.078 |

The model is most robust to speckle noise in this sweep and degrades most sharply under high Gaussian or high Rician corruption.

## Ablation Suite

The ablation suite is stored in [`results_ablation/unified_ablation_suite.csv`](https://github.com/tushar-nayak/neural-anisotropic-diffusion/blob/extended/results_ablation/unified_ablation_suite.csv).

| Variant | PSNR (dB) | SSIM | Edge MSE | Best Val Loss |
| --- | ---: | ---: | ---: | ---: |
| Full Model | 24.118 | 0.691 | 0.061 | 0.354 |
| No MiniUNet Guidance | 24.249 | 0.699 | 0.061 | 0.344 |
| No Residual Refinement | 22.798 | 0.601 | 0.064 | 0.443 |
| 4-Neighbor Diffusion | 24.008 | 0.686 | 0.062 | 0.356 |

The ablation results suggest that residual refinement is the most important optional component. MiniUNet guidance did not improve this short ablation run, and 8-neighbor diffusion was slightly stronger than the 4-neighbor variant.

## Existing Archived Runs

### First-Pass Br35H Run

The first-pass run is archived in `results_30epochs/`. The folder name is historical; the notes record that it came from a 10-epoch command.

- Command: `python -u main.py --epochs 10`
- Best validation loss: `0.3547`
- Unified Neural PDE: `24.132 dB` PSNR / `0.690` SSIM

### Interrupted Long Run

The longer run is archived in `results_300epochs/`. It was started for 300 epochs and stopped after the validation metrics had flattened around epoch 110.

- Command: `python -u main.py --epochs 300 --results-dir results_300epochs --checkpoint-dir checkpoints_300epochs`
- Last captured epoch: `110 / 300`
- Checkpoint test result: `24.586 dB` PSNR / `0.713` SSIM

## Recommended Follow-Up Experiments

The final all-in-one run can be reproduced with:

```bash
scripts/run_final_all.sh
```

Individual scripts are also available:

```bash
scripts/run_extended_eval.sh
scripts/run_noise_sweep.sh
scripts/run_ablation_suite.sh
scripts/run_unet_baseline.sh
```

These generate updated tables with PSNR mean/std, SSIM mean/std, Sobel edge MSE, full comparison grids, metric bar plots, and run metadata JSON files. They are useful when GPU memory is available.

For a fast smoke check without a full test-set pass:

```bash
python -u main.py \
  --eval-only \
  --checkpoint checkpoints_extended/unified_model.pth \
  --eval-limit 50 \
  --results-dir results_quick_eval \
  --checkpoint-dir checkpoints_extended
```

A two-example smoke evaluation is committed in `results_quick_eval/`. It is only a runtime sanity check for the richer evaluator outputs, not the main reported benchmark.

## Interpretation

The main result supports the core claim: learned anisotropic diffusion can preserve the useful structure of classical PDE-based denoising while learning better spatially adaptive smoothing behavior from data. The method substantially outperforms the hand-designed denoisers included in the current benchmark.
