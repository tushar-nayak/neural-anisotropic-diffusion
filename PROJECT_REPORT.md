# Neural Anisotropic Diffusion

## A manuscript-style report on the unified MRI denoising project

### Abstract
This project explores a learned variant of Perona-Malik anisotropic diffusion for brain MRI denoising. Instead of using a fixed conduction function, the model unrolls a diffusion process and predicts spatially varying diffusion weights with a neural network. The goal is to reduce synthetic noise while preserving anatomical boundaries such as lesion margins, cortical edges, and tissue interfaces. The final unified implementation supports both 4-neighbor and 8-neighbor diffusion, learnable guidance features, optional residual refinement, and multiple synthetic corruption modes. In a full training run on the local brain MRI dataset, the unified model achieved `23.612 dB` PSNR and `0.657` SSIM on the held-out test set, outperforming Gaussian smoothing, TV denoising, and classical Perona-Malik diffusion under the same evaluation split.

### 1. Introduction
Image denoising in medical imaging is not only about removing noise. In MRI, especially in tumor-related scans, a useful denoiser must preserve structural detail that matters for downstream interpretation. Standard smoothing filters often blur the exact boundaries clinicians and researchers care about. This project was built around that problem.

The central idea is to treat denoising as a learned partial differential equation. Classical anisotropic diffusion updates an image iteratively by comparing local gradients and applying a conduction coefficient that suppresses diffusion near edges. Here, that coefficient is not fixed. It is predicted by a neural network from local gradient information and a guidance branch, allowing the model to adapt diffusion spatially and per image.

The repository went through several experimental stages. The final branch now contains a single clean entry point in [`main.py`](/home/sofa/host_dir/nad/neural-anisotropic-diffusion/main.py), with the earlier experimental scripts removed. The cleaned branch is intended to be reproducible, readable, and suitable for presentation.

### 2. Project Objective
The objective is to learn a denoiser that:

1. Removes synthetic MRI corruption.
2. Preserves fine edges and large-scale anatomy.
3. Beats classical denoising baselines on the same data split.
4. Remains simple enough to run locally and inspect visually.

The project is intentionally framed as a denoising benchmark rather than a classifier. The `no` and `yes` folder labels in the dataset are used for stratified splitting and for labeling qualitative examples, not for predicting disease.

### 3. Dataset
The data lives in [`brain_tumor_dataset/`](/home/sofa/host_dir/nad/neural-anisotropic-diffusion/brain_tumor_dataset). It contains grayscale MRI slices stored in two folders:

- `no`
- `yes`

The final dataset size used by the scripts is 253 images:

- 98 images in `no`
- 155 images in `yes`

Each image is resized to `128 x 128` and converted to a single grayscale channel. The data is split with stratification into:

- 177 training images
- 38 validation images
- 38 test images

This split preserves the class balance across all three subsets.

### 4. Corruption Model
Noise is synthesized on the fly in the dataset loader, so the training pairs are created dynamically. The final unified script supports:

- Gaussian corruption
- Rician corruption
- Speckle corruption
- Mixed corruption

The default training configuration uses Rician noise. For the latest full run reported here, the model was trained in the default unified configuration and evaluated on the held-out test set after the best validation checkpoint was selected.

Using dynamic corruption has two consequences:

1. The model sees slightly different noisy inputs across epochs, which acts as a form of augmentation.
2. Evaluation must be handled carefully so the same test corruption is reused consistently when comparing the model to classical baselines.

The final branch handles this by caching the test batches before computing the comparison table.

### 5. Method
The unified model is an unrolled diffusion network. It starts from a noisy input and iteratively updates the image using predicted diffusion weights.

#### 5.1 Guidance encoder
The model first extracts guidance features from a lightly smoothed version of the noisy input. The branch can use either:

- a MiniUNet guidance encoder, or
- a lighter legacy-style convolutional guidance encoder

The default unified configuration uses the MiniUNet branch for richer context.

#### 5.2 Directional gradients
At each iteration, the model computes local gradients using replicated padding and finite differences. It supports:

- 4-neighbor diffusion: north, south, east, west
- 8-neighbor diffusion: the four cardinal directions plus four diagonals

The 8-neighbor version is more expressive but also more constrained by stability. The code therefore uses a smaller step size for the 8-neighbor configuration.

#### 5.3 Conduction network
The stacked gradients and guidance features are passed through a small CNN that predicts per-pixel conduction coefficients. These coefficients are constrained to `[0, 1]` with a sigmoid.

The update rule is:

`x <- x + lambda * sum(c_i * grad_i)`

where `c_i` are the learned conduction coefficients and `grad_i` are directional gradients.

#### 5.4 Optional refinement
After the diffusion loop, the unified model can optionally add a residual refinement step. This is a light convolutional correction stage intended to recover detail that may be partially smoothed by the PDE iterations.

### 6. Training Objective
The unified script trains with a composite loss:

- SSIM loss
- L1 loss
- gradient loss

The gradient term discourages edge blurring by comparing image derivatives between prediction and target. This is more aligned with medical-image denoising than a pixel-only loss.

The loss used in the final unified run is:

`SSIM + L1 + 0.1 * gradient_loss`

This balance was chosen to encourage structural fidelity while still keeping the optimization stable on a small dataset.

### 7. Baselines
The final evaluation compares the learned model to classical denoisers:

- Gaussian smoothing
- Classical Perona-Malik diffusion
- TV denoising via `skimage`

These baselines are important because they show whether the learned PDE actually offers an advantage over standard hand-designed methods on the same images.

### 8. Implementation Notes
The cleaned branch now contains:

- [`main.py`](/home/sofa/host_dir/nad/neural-anisotropic-diffusion/main.py)
- [`README.md`](/home/sofa/host_dir/nad/neural-anisotropic-diffusion/README.md)
- [`requirements.txt`](/home/sofa/host_dir/nad/neural-anisotropic-diffusion/requirements.txt)
- [`Makefile`](/home/sofa/host_dir/nad/neural-anisotropic-diffusion/Makefile)

The final script also includes:

- a local SSIM fallback if `pytorch_msssim` is not installed
- headless plotting via the Agg backend
- a reproducible train/val/test split
- configurable noise type and architecture settings

The `Makefile` exposes:

- `make run`
- `make smoke`

### 9. Experimental Results
The full run produced the following test comparison table:

| Method | PSNR (dB) | SSIM |
| --- | ---: | ---: |
| Gaussian Smoothing | 19.996 | 0.549 |
| Skimage TV | 20.068 | 0.558 |
| Classical PM (16 iter) | 20.005 | 0.498 |
| Unified Neural PDE (Ours) | 23.612 | 0.657 |

These results indicate that the unified learned diffusion model is materially better than the classical baselines on this test split.

The training curves in [`results/unified_loss_curves.png`](/home/sofa/host_dir/nad/neural-anisotropic-diffusion/results/unified_loss_curves.png) show:

- a sharp improvement during early epochs
- a slower convergence phase afterward
- moderate validation oscillation, which is expected on a small medical-image dataset with dynamic synthetic corruption

The best validation loss reached `0.2971`.

### 10. Qualitative Analysis
The visual results in [`results/unified_qualitative_results.png`](/home/sofa/host_dir/nad/neural-anisotropic-diffusion/results/unified_qualitative_results.png) show that the model:

- removes a large amount of synthetic noise
- keeps the overall brain anatomy recognizable
- preserves broad contrast structure better than simple smoothing
- still tends to over-smooth some fine boundaries and small lesions

This is the key tradeoff in the current system: the model is clearly better than classical baselines overall, but the outputs still show some loss of fine detail. That is not a failure of the project; it is the central area for future improvement.

### 11. Interpretation
The project supports a practical conclusion: learnable anisotropic diffusion is worth exploring for medical-image denoising. The classical PDE is already a good inductive bias, and learning the conduction weights lets the network adapt to local anatomy and noise patterns. The unified model is not perfect, but it meaningfully improves quantitative metrics over the hand-crafted methods included in the repository.

The results also suggest that the data regime matters. The dataset is small, the corruption is synthetic, and the model is sensitive to hyperparameters. The architecture works, but it is not yet tuned for clinical deployment.

### 12. Limitations
The main limitations are:

1. The dataset is relatively small.
2. Noise is synthetic rather than acquired from scanner artifacts.
3. The evaluation is limited to a single dataset and split.
4. The model can smooth away detail that may matter diagnostically.
5. There is no external validation set from a separate institution.

### 13. Future Work
The natural next steps are:

- train on larger and more diverse MRI datasets
- use real noise models or scanner-level corruption
- add uncertainty estimation
- test additional losses such as perceptual or boundary-aware terms
- compare against more modern restoration baselines
- measure performance on task-specific downstream metrics

### 14. Reproducibility
To reproduce the main run:

```bash
make run
```

For a quick sanity check:

```bash
make smoke
```

Dependencies are listed in [`requirements.txt`](/home/sofa/host_dir/nad/neural-anisotropic-diffusion/requirements.txt).

### 15. Conclusion
This project demonstrates a unified learned anisotropic diffusion system for MRI denoising. The final implementation is clean, runnable, and documented. On the current dataset and evaluation setup, it outperforms the classical baselines it was compared against, while still revealing the expected tension between denoising strength and detail preservation. The repo is now organized around one coherent branch, one coherent entry point, and one reproducible experimental story.
