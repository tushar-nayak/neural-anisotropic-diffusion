# Neural Anisotropic Diffusion

This repository contains a unified, working version of a learned Perona-Malik style image denoiser for brain MRI slices. The model unrolls a PDE-style diffusion process and predicts spatially varying conduction weights with a neural network so it can smooth noise while trying to preserve edges.

## What the unified script does

- Loads grayscale MRI slices from the local Br35H Kaggle download in [`brain_tumor_dataset/`](/home/sofa/host_dir/nad/neural-anisotropic-diffusion/brain_tumor_dataset).
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

The main entry point is [`main.py`](/home/sofa/host_dir/nad/neural-anisotropic-diffusion/main.py).

## Repository Layout

- [`main.py`](/home/sofa/host_dir/nad/neural-anisotropic-diffusion/main.py): unified runnable version
- [`Makefile`](/home/sofa/host_dir/nad/neural-anisotropic-diffusion/Makefile): `run` and `smoke` targets
- [`requirements.txt`](/home/sofa/host_dir/nad/neural-anisotropic-diffusion/requirements.txt): dependency list
- [`download_br35h_dataset.py`](/home/sofa/host_dir/nad/neural-anisotropic-diffusion/download_br35h_dataset.py): downloads the Br35H Kaggle dataset into the repo-local `brain_tumor_dataset/` path
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
