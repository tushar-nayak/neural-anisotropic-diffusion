# Results

This file summarizes the committed experiments for the learned neural anisotropic diffusion denoiser. The project is evaluated as an MRI denoising benchmark, not as a tumor classifier.

## Dataset

- Dataset: Br35H brain tumor MRI slices
- Images used by the loader: `3000`
- Split: `2100` train / `450` validation / `450` test
- Training corruption: synthetic Rician-style noise in the extended run
- Metrics: PSNR and SSIM

## Main Extended Result

The latest committed extended result is stored in [`results_extended/unified_comparison_table.csv`](https://github.com/tushar-nayak/neural-anisotropic-diffusion/blob/extended/results_extended/unified_comparison_table.csv).

| Method | PSNR (dB) | SSIM |
| --- | ---: | ---: |
| Noisy Input | 17.727 | 0.426 |
| Gaussian Smoothing | 18.968 | 0.549 |
| Median Filter | 19.502 | 0.519 |
| Bilateral Filter | 18.952 | 0.461 |
| Non-Local Means | 20.376 | 0.583 |
| Wavelet Denoising | 19.523 | 0.529 |
| Skimage TV | 19.457 | 0.568 |
| Curvature Flow (16 iter) | 19.957 | 0.557 |
| Classical PM (16 iter) | 19.610 | 0.535 |
| Unified Neural PDE (Ours) | 24.932 | 0.722 |

The strongest classical baseline in this table is Non-Local Means. The learned diffusion model improves over it by:

| Comparison | PSNR Gain | SSIM Gain |
| --- | ---: | ---: |
| Ours vs Non-Local Means | +4.556 dB | +0.139 |

The PSNR/SSIM bar chart is stored in [`results_extended/unified_metric_bars.png`](https://github.com/tushar-nayak/neural-anisotropic-diffusion/blob/extended/results_extended/unified_metric_bars.png).

![PSNR and SSIM bar charts](https://raw.githubusercontent.com/tushar-nayak/neural-anisotropic-diffusion/extended/results_extended/unified_metric_bars.png)

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

The current code supports richer result generation through scripts:

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
