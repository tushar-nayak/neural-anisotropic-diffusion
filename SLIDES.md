# Neural Anisotropic Diffusion

## Slide Deck for Presentation

---

## Slide 1: Title

**Neural Anisotropic Diffusion for MRI Denoising**

- Unrolled PDE-based restoration for brain MRI slices
- Learnable conduction coefficients instead of fixed diffusion rules
- Unified final implementation in `main.py`

**Speaker notes**

This project asks whether a classical anisotropic diffusion process can be turned into a learnable model that denoises MRI scans without destroying anatomy. The final branch contains a single working implementation that unrolls the diffusion process and learns the local diffusion weights.

---

## Slide 2: Why this matters

- Medical denoising should remove noise without erasing anatomy
- Standard filters can blur tumor margins and tissue boundaries
- MRI often needs edge-aware restoration rather than aggressive smoothing

**Speaker notes**

The motivation is practical: in brain MRI, preserving structure is as important as removing noise. A denoiser that makes the image look smoother but hides the boundary of a lesion is not useful. That is why anisotropic diffusion is a natural starting point.

---

## Slide 3: Core idea

- Start from Perona-Malik anisotropic diffusion
- Unroll the iterative update as a neural network
- Predict conduction coefficients from the local image context
- Update pixels by learned, spatially varying diffusion weights

**Speaker notes**

Instead of using a hand-designed conduction function, the model learns where to diffuse strongly and where to preserve edges. This keeps the classical PDE prior, but lets the network adapt it to actual image content.

---

## Slide 4: Dataset and corruption

- Local brain MRI dataset in `brain_tumor_dataset/`
- 253 grayscale slices total
- Stratified split:
  - 177 train
  - 38 validation
  - 38 test
- Synthetic corruption added on the fly:
  - Gaussian
  - Rician
  - Speckle
  - Mixed

**Speaker notes**

The dataset is small, so the model is trained as a denoiser rather than a classifier. Labels are only used for stratified splitting and to organize examples in the qualitative figures. The corruption is generated dynamically during loading so the model sees varied noisy versions of the same clean images.

---

## Slide 5: Model architecture

- Guidance encoder:
  - MiniUNet or lightweight convolutional encoder
- PDE loop:
  - 4-neighbor or 8-neighbor gradients
  - sigmoid-constrained conduction coefficients
- Optional residual refinement stage

**Speaker notes**

The architecture has three parts: guidance extraction, iterative diffusion, and optional refinement. The 8-neighbor mode is more expressive and was used for the final unified run. The refinement block helps recover some detail after diffusion.

---

## Slide 6: Training objective

- SSIM loss
- L1 loss
- Gradient loss
- Final weighting:
  - `SSIM + L1 + 0.1 * gradient_loss`

**Speaker notes**

The loss is designed to care about both perceptual structure and pixel fidelity. The gradient term is important because edge preservation is the main medical-imaging concern here. Without it, the model is more likely to blur boundaries.

---

## Slide 7: Training setup

- Headless-safe PyTorch script
- Cosine annealing learning rate schedule
- Gradient clipping
- Best validation checkpoint saved
- Baselines evaluated on the same test set

**Speaker notes**

The final script is designed to run locally or over SSH. It writes plots and a CSV table automatically. The evaluation compares the learned model against Gaussian smoothing, classical Perona-Malik diffusion, and TV denoising.

---

## Slide 8: Quantitative results

| Method | PSNR (dB) | SSIM |
| --- | ---: | ---: |
| Gaussian Smoothing | 19.996 | 0.549 |
| Skimage TV | 20.068 | 0.558 |
| Classical PM (16 iter) | 20.005 | 0.498 |
| Unified Neural PDE (Ours) | 23.612 | 0.657 |

**Speaker notes**

On the held-out test split, the unified model clearly outperforms the classical baselines in both PSNR and SSIM. The improvement is substantial enough to be meaningful, not just a rounding artifact.

---

## Slide 9: Training behavior

- Best validation loss: `0.2971`
- Loss decreased steadily over training
- PSNR improved quickly early on and then plateaued
- Validation oscillation remained moderate

**Speaker notes**

The training curve shows a stable learning process rather than collapse or divergence. The validation signal is noisy because the dataset is small and the corruption is dynamic, but the best checkpoint is clearly better than the start of training.

---

## Slide 10: Qualitative observations

- Noise is reduced strongly
- Brain structure remains recognizable
- Large-scale contrast is preserved
- Fine boundaries are still somewhat over-smoothed

**Speaker notes**

The figure shows that the model behaves as intended, but it is not perfect. It preserves useful anatomy while reducing noise, yet it can still smooth small structures that may matter clinically. That tradeoff is the main thing to improve next.

---

## Slide 11: Limitations

- Small dataset
- Synthetic noise only
- One dataset and one split
- Some detail loss remains
- No external clinical validation yet

**Speaker notes**

These limitations matter. The model is promising, but not a deployed medical tool. The evaluation is best understood as a controlled proof of concept on a compact benchmark.

---

## Slide 12: Takeaway

- Learned anisotropic diffusion is viable
- The unified branch is runnable and documented
- The method beats the classical baselines on this setup
- Next steps: more data, more realism, better detail preservation

**Speaker notes**

The main takeaway is that a learned PDE prior is useful. This branch now contains a coherent implementation, a clean README, a report, and reproducible run instructions. The result is a solid foundation for further research, not just a one-off experiment.

---

## Appendix: How to run

```bash
make run
```

```bash
make smoke
```

**Speaker notes**

The `make run` target executes the full experiment. The `make smoke` target is a short sanity check that is useful for quick verification or demos.
