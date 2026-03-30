# Neural Anisotropic Diffusion (Unrolled Partial Differential Equation)

Standard image smoothing filters often degrade critical structural boundaries in medical scans, such as tumor margins or bone structures. This project introduces a modern, deep-learning-based approach to the classical image relaxation methods (specifically, Perona-Malik anisotropic diffusion). Instead of relying on a static mathematical decay function for the conduction coefficient, this architecture unrolls the iterative steps of a Partial Differential Equation (PDE) solver. It replaces the mathematical coefficient with a spatially-aware, learnable Convolutional Neural Network (CNN) that predicts the optimal diffusion weights locally across the image. The result is a model that dynamically filters severe speckle and Gaussian noise while acting as a barrier to preserve critical anatomical edges.

## Architecture
The core model is a PyTorch module that unrolls $T$ discrete steps of the diffusion equation. 
* **Input:** A noisy medical image.
* **Gradients:** Calculates spatial gradients (North, South, East, West) using finite differences.
* **Neural Conduction Predictor:** A lightweight, 3-layer CNN processes the stacked directional gradients and outputs four localized conduction coefficients constrained between [0, 1] via a Sigmoid activation.
* **Update Step:** Applies the discrete PDE update using the predicted, spatially-aware weights.
* **Loss:** The network is trained in a supervised manner using Mean Squared Error (MSE) against clean, uncorrupted ground-truth images.

## Dataset
This project is designed to work with high-contrast medical imaging datasets. The current implementation supports Synthetic Corruption (The data loader dynamically injects severe Gaussian/speckle noise into clean images to generate supervised training pair) and this entire pipeline has been train-validated-tested on a Brain MRI Dataset from [Kaggle](https://www.kaggle.com/datasets/hasimdev/brain-mri-dataset)

## Requirements
* Python 3.8+
* PyTorch (MPS/CUDA supported for hardware acceleration)
* Torchvision
* NumPy
* Matplotlib (for visualizing diffusion results)

## Usage
1. Clone the repository:
   ```bash
   git clone [https://github.com/tushar-nayak/neural-anisotropic-diffusion.git](https://github.com/tushar-nayak/neural-anisotropic-diffusion.git)
