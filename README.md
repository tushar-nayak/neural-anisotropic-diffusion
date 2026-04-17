# Neural Anisotropic Diffusion

This repository contains a unified, working version of a learned Perona-Malik style image denoiser for brain MRI slices. The model unrolls a PDE-style diffusion process and predicts spatially varying conduction weights with a neural network so it can smooth noise while trying to preserve edges.

## Project Summary

Medical image denoising is not just a smoothing problem. In brain MRI, useful denoising should suppress noise while preserving anatomical boundaries, lesion margins, and tissue structure. Classical filters such as Gaussian smoothing can improve visual smoothness, but they often blur the same edges that matter downstream.

This project explores a learned anisotropic diffusion model for MRI denoising. The model keeps the structure of a Perona-Malik style diffusion process, but replaces the fixed hand-designed conduction function with learned spatially varying conduction weights. The result is a hybrid model: it has the inductive bias of a PDE-based denoiser while still learning from data.

The current extended branch includes:

- a unified PyTorch training and evaluation script
- 4-neighbor and 8-neighbor diffusion modes
- optional MiniUNet guidance features
- optional residual refinement
- classical and neural comparison baselines
- edge-preservation metrics
- eval-only, noise-sweep, ablation, and config-driven experiment modes

## Method

The model starts from a noisy MRI slice and repeatedly applies a learned diffusion update. At each unrolled step, local image gradients are computed across either 4 or 8 neighboring directions. A small neural conduction network predicts how much diffusion should happen along each direction, using both local gradients and optional guidance features.

The training objective combines:

- SSIM loss for structural similarity
- L1 loss for pixel-level fidelity
- gradient loss for edge preservation

The default loss is:

```text
SSIM + L1 + 0.1 * gradient_loss
```

This is intended to reward denoising without fully washing out anatomical detail.

## Dataset And Evaluation

The experiments use the Br35H brain tumor MRI dataset downloaded into the local `brain_tumor_dataset/` directory. The labels in the dataset are used only for stratified splitting and qualitative grouping. This project treats the images as a denoising benchmark, not as a tumor classifier.

The committed extended result uses:

- `3000` total images
- `2100 / 450 / 450` train, validation, and test split
- Rician-style synthetic corruption during training
- held-out test evaluation against classical denoisers
- PSNR and SSIM as image-quality metrics

The latest extended comparison is saved in [`results_extended/unified_comparison_table.csv`](https://github.com/tushar-nayak/neural-anisotropic-diffusion/blob/extended/results_extended/unified_comparison_table.csv).

## Results

The extended run produced the following held-out test results:

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

The learned diffusion model is the strongest method in this comparison. Relative to the best classical baseline in this table, Non-Local Means, the unified neural PDE improves:

| Comparison | PSNR Gain | SSIM Gain |
| --- | ---: | ---: |
| Ours vs Non-Local Means | +4.556 dB | +0.139 |

This result supports the main project claim: a learned diffusion process can outperform standard hand-designed denoisers on this MRI denoising setup while retaining the interpretability of iterative diffusion.

## Outputs And Figures

The extended result artifacts are committed in [`results_extended/`](https://github.com/tushar-nayak/neural-anisotropic-diffusion/tree/extended/results_extended):

- [`unified_comparison_table.csv`](https://github.com/tushar-nayak/neural-anisotropic-diffusion/blob/extended/results_extended/unified_comparison_table.csv): quantitative comparison table
- [`unified_loss_curves.png`](https://github.com/tushar-nayak/neural-anisotropic-diffusion/blob/extended/results_extended/unified_loss_curves.png): training and validation curves
- [`unified_qualitative_results.png`](https://github.com/tushar-nayak/neural-anisotropic-diffusion/blob/extended/results_extended/unified_qualitative_results.png): qualitative denoising examples

### Training Curves

![Training and validation loss and PSNR curves](https://raw.githubusercontent.com/tushar-nayak/neural-anisotropic-diffusion/extended/results_extended/unified_loss_curves.png)

### Qualitative Denoising Examples

![Qualitative MRI denoising examples](https://raw.githubusercontent.com/tushar-nayak/neural-anisotropic-diffusion/extended/results_extended/unified_qualitative_results.png)

## What the unified script does

- Loads grayscale MRI slices from the local Br35H Kaggle download in `brain_tumor_dataset/`.
- Splits the data into stratified train, validation, and test sets.
- Synthesizes noise on the fly using Gaussian, Rician, speckle, or mixed corruption.
- Trains a learnable anisotropic diffusion model with SSIM + L1 + gradient loss.
- Optionally uses:
  - a 4-neighbor or 8-neighbor PDE update
  - a MiniUNet guidance encoder
  - a residual refinement stage
- Writes local outputs for:
  - training curves
  - qualitative examples
  - a comparison table against noisy input, Gaussian smoothing, median filtering, bilateral filtering, non-local means, wavelet denoising, classical Perona-Malik, curvature flow, and TV denoising

The main entry point is [`main.py`](https://github.com/tushar-nayak/neural-anisotropic-diffusion/blob/extended/main.py).

## Repository Layout

- [`main.py`](https://github.com/tushar-nayak/neural-anisotropic-diffusion/blob/extended/main.py): unified runnable version
- [`Makefile`](https://github.com/tushar-nayak/neural-anisotropic-diffusion/blob/extended/Makefile): `run` and `smoke` targets
- [`requirements.txt`](https://github.com/tushar-nayak/neural-anisotropic-diffusion/blob/extended/requirements.txt): dependency list
- [`download_br35h_dataset.py`](https://github.com/tushar-nayak/neural-anisotropic-diffusion/blob/extended/download_br35h_dataset.py): downloads the Br35H Kaggle dataset into the repo-local `brain_tumor_dataset/` path
- `brain_tumor_dataset/`: local Br35H MRI dataset, ignored by Git

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- pandas
- matplotlib
- scikit-learn
- Pillow
- scipy
- scikit-image

Optional:

- `pytorch_msssim` for the SSIM loss implementation

If `pytorch_msssim` is not installed, `main.py` falls back to a local Torch SSIM implementation.

## Usage

Run the default unified model:

```bash
python main.py
```

Or use the make target:

```bash
make run
```

Useful flags:

```bash
python main.py --neighbor-mode 4
python main.py --neighbor-mode 8
python main.py --noise-type gaussian
python main.py --noise-type rician
python main.py --iterations 16 --lambda-param 0.05
python main.py --epochs 300 --batch-size 8
python main.py --epochs 300 --results-dir results_300epochs --checkpoint-dir checkpoints_300epochs
python main.py --eval-only --checkpoint checkpoints_extended/unified_model.pth --results-dir results_eval
python main.py --noise-sweep --noise-sweep-types gaussian,rician,speckle --noise-sweep-sigmas 0.05,0.10,0.15,0.20
python main.py --train-unet-baseline-epochs 50
python main.py --run-ablation-suite --ablation-epochs 20
python main.py --config configs/example.json
python main.py --no-refinement
python main.py --no-unet-guidance
```

For a fast sanity check:

```bash
make smoke
```

Download or refresh the Br35H dataset locally:

```bash
python download_br35h_dataset.py
```

Default behavior:

- `neighbor-mode=8`
- `noise-type=rician`
- `iterations=16` for 8-neighbor mode
- `lambda-param=0.05` for 8-neighbor mode
- `iterations=10` for 4-neighbor mode
- `lambda-param=0.1` for 4-neighbor mode

## Outputs

The unified script writes these files locally when you run it:

- `results/unified_loss_curves.png`
- `results/unified_qualitative_results.png`
- `results/unified_comparison_table.csv`
- `results/unified_full_comparison_grid.png`
- `results/unified_noise_sweep.csv` when `--noise-sweep` is enabled
- `results/unified_ablation_suite.csv` when `--run-ablation-suite` is enabled
- `checkpoints/unified_model.pth`

The comparison CSV reports mean and standard deviation for PSNR and SSIM, plus a Sobel-edge MSE metric for edge preservation. Lower edge MSE indicates closer agreement with the clean image's edge map.

Optional experiment modes:

- `--eval-only` evaluates an existing checkpoint without retraining.
- `--noise-sweep` evaluates robustness across fixed corruption types and noise levels.
- `--train-unet-baseline-epochs N` trains a plain U-Net denoising baseline and adds it to the comparison table.
- `--run-ablation-suite` trains common variants: full model, no MiniUNet guidance, no residual refinement, and 4-neighbor diffusion.
- `--config path/to/config.json` loads arguments from a JSON file using the same names as the command-line flags.

## Notes

- The dataset is treated as a denoising benchmark, not a classifier.
- The `no` and `yes` folder labels are used for stratified splitting and for labeling qualitative examples.
- The script is headless-safe and uses the Agg matplotlib backend, so it runs over SSH or in a non-GUI environment.
- The repository is intentionally cleaned to keep only the unified source, docs, and reproducibility files.
