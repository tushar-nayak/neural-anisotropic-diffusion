# ============================================================
# Neural Anisotropic Diffusion - Unified Final Working Version
# 4/8-Neighbor PDE + Learnable Guidance + Optional Refinement
# ============================================================
import argparse
import datetime
import os
import glob
import json
import platform
import subprocess
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, Subset

from scipy.ndimage import gaussian_filter, median_filter, sobel
from skimage.restoration import (
    denoise_bilateral,
    denoise_nl_means,
    denoise_tv_chambolle,
    denoise_wavelet,
)
from skimage.metrics import structural_similarity as skimage_ssim
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr


warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

try:
    from pytorch_msssim import ssim
except ImportError:
    def _gaussian_window(window_size, sigma, device, dtype):
        coords = torch.arange(window_size, device=device, dtype=dtype) - window_size // 2
        g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        g = g / g.sum()
        window = (g[:, None] * g[None, :]).unsqueeze(0).unsqueeze(0)
        return window

    def ssim(x, y, data_range=1.0, window_size=11, sigma=1.5, size_average=True):
        channel = x.size(1)
        window = _gaussian_window(window_size, sigma, x.device, x.dtype).repeat(channel, 1, 1, 1)
        padding = window_size // 2

        mu_x = F.conv2d(x, window, padding=padding, groups=channel)
        mu_y = F.conv2d(y, window, padding=padding, groups=channel)

        mu_x2 = mu_x * mu_x
        mu_y2 = mu_y * mu_y
        mu_xy = mu_x * mu_y

        sigma_x2 = F.conv2d(x * x, window, padding=padding, groups=channel) - mu_x2
        sigma_y2 = F.conv2d(y * y, window, padding=padding, groups=channel) - mu_y2
        sigma_xy = F.conv2d(x * y, window, padding=padding, groups=channel) - mu_xy

        c1 = (0.01 * data_range) ** 2
        c2 = (0.03 * data_range) ** 2

        ssim_map = ((2 * mu_xy + c1) * (2 * sigma_xy + c2)) / (
            (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)
        )

        return ssim_map.mean() if size_average else ssim_map.flatten(1).mean(dim=1)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "brain_tumor_dataset")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")


def resolve_repo_path(path):
    return path if os.path.isabs(path) else os.path.join(BASE_DIR, path)


class MiniUNet(nn.Module):
    """Compact global context encoder for guidance features."""

    def __init__(self, in_ch=1, out_ch=8):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_ch, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec = nn.Sequential(
            nn.Conv2d(32 + 16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, out_ch, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        return self.dec(torch.cat([self.up(e2), e1], dim=1))


class SimpleGuidanceEncoder(nn.Module):
    """Lightweight guidance encoder for the legacy 4-neighbor setup."""

    def __init__(self, out_ch=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, out_ch, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return self.net(x)


class UnifiedNeuralPeronaMalik(nn.Module):
    def __init__(
        self,
        iterations=16,
        lambda_param=0.05,
        guidance_channels=8,
        neighbor_mode=8,
        use_refinement=True,
        use_unet_guidance=True,
    ):
        super().__init__()

        if neighbor_mode not in (4, 8):
            raise ValueError("neighbor_mode must be 4 or 8")

        self.iterations = iterations
        self.lambda_param = lambda_param
        self.neighbor_mode = neighbor_mode
        self.use_refinement = use_refinement

        if neighbor_mode == 4:
            assert lambda_param <= 0.25, "lambda_param > 0.25 violates 4-neighbor stability"
        else:
            assert lambda_param <= 0.125, "lambda_param > 0.125 violates 8-neighbor stability"

        self.guidance_encoder = MiniUNet(1, guidance_channels) if use_unet_guidance else SimpleGuidanceEncoder(guidance_channels)

        in_ch = neighbor_mode + guidance_channels
        self.conduction_net = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, neighbor_mode, kernel_size=3, padding=1),
            nn.Sigmoid(),
        )

        self.refinement_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1),
        )

    def _neighbor_gradients(self, x):
        x_pad = F.pad(x, (1, 1, 1, 1), mode="replicate")
        grad_n = x_pad[:, :, :-2, 1:-1] - x
        grad_s = x_pad[:, :, 2:, 1:-1] - x
        grad_e = x_pad[:, :, 1:-1, 2:] - x
        grad_w = x_pad[:, :, 1:-1, :-2] - x
        grads = [grad_n, grad_s, grad_e, grad_w]

        if self.neighbor_mode == 8:
            scale = 0.707
            grad_nw = (x_pad[:, :, :-2, :-2] - x) * scale
            grad_ne = (x_pad[:, :, :-2, 2:] - x) * scale
            grad_sw = (x_pad[:, :, 2:, :-2] - x) * scale
            grad_se = (x_pad[:, :, 2:, 2:] - x) * scale
            grads.extend([grad_nw, grad_ne, grad_sw, grad_se])

        return grads

    def forward(self, x):
        x_smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        guidance_features = self.guidance_encoder(x_smooth).detach()

        for _ in range(self.iterations):
            grads = self._neighbor_gradients(x)
            combined = torch.cat(grads + [guidance_features], dim=1)
            coeffs = self.conduction_net(combined)
            coeffs = torch.split(coeffs, 1, dim=1)

            update = 0.0
            for c, g in zip(coeffs, grads):
                update = update + c * g
            x = x + self.lambda_param * update

        if self.use_refinement:
            x = x + self.refinement_net(x)

        return x


class UNetDenoiser(nn.Module):
    """Plain neural denoiser baseline trained on the same noisy/clean pairs."""

    def __init__(self):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.out = nn.Conv2d(64, 1, kernel_size=1)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        b = self.bottleneck(self.pool(e2))
        d1 = self.dec1(torch.cat([self.up(b), e2], dim=1))
        return x + self.out(torch.cat([self.up(d1), e1], dim=1))


def gradient_loss(pred, target):
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_tgt = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
    dy_tgt = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.mse_loss(dx_pred, dx_tgt) + F.mse_loss(dy_pred, dy_tgt)


def psnr_metric(pred, target):
    mse = F.mse_loss(pred, target)
    return 10 * torch.log10(1.0 / (mse + 1e-8))


def edge_mse_np(clean, denoised):
    clean_edges = np.hypot(sobel(clean, axis=0), sobel(clean, axis=1))
    denoised_edges = np.hypot(sobel(denoised, axis=0), sobel(denoised, axis=1))
    return float(np.mean((clean_edges - denoised_edges) ** 2))


def edge_mse_torch(pred, target):
    scores = []
    for pred_img, target_img in zip(pred.detach().cpu(), target.detach().cpu()):
        scores.append(edge_mse_np(target_img.squeeze().numpy(), pred_img.squeeze().numpy()))
    return scores


def combined_loss(output, clean, l1_fn, grad_weight=0.1):
    ssim_l = 1.0 - ssim(output, clean, data_range=1.0)
    l1_l = l1_fn(output, clean)
    grad_l = gradient_loss(output, clean)
    return ssim_l + l1_l + grad_weight * grad_l


def classical_perona_malik(img, iterations=16, kappa=0.1, gamma=0.05):
    u = img.copy()
    for _ in range(iterations):
        n = np.roll(u, -1, axis=0) - u
        s = np.roll(u, 1, axis=0) - u
        e = np.roll(u, -1, axis=1) - u
        w = np.roll(u, 1, axis=1) - u

        c_n = 1 / (1 + (np.abs(n) / kappa) ** 2)
        c_s = 1 / (1 + (np.abs(s) / kappa) ** 2)
        c_e = 1 / (1 + (np.abs(e) / kappa) ** 2)
        c_w = 1 / (1 + (np.abs(w) / kappa) ** 2)

        u = u + gamma * (c_n * n + c_s * s + c_e * e + c_w * w)

    return np.clip(u, 0, 1)


def curvature_flow(img, iterations=16, gamma=0.05, eps=1e-6):
    u = img.astype(np.float32, copy=True)
    for _ in range(iterations):
        ux = 0.5 * (np.roll(u, -1, axis=1) - np.roll(u, 1, axis=1))
        uy = 0.5 * (np.roll(u, -1, axis=0) - np.roll(u, 1, axis=0))
        uxx = np.roll(u, -1, axis=1) - 2.0 * u + np.roll(u, 1, axis=1)
        uyy = np.roll(u, -1, axis=0) - 2.0 * u + np.roll(u, 1, axis=0)
        uxy = 0.25 * (
            np.roll(np.roll(u, -1, axis=0), -1, axis=1)
            - np.roll(np.roll(u, -1, axis=0), 1, axis=1)
            - np.roll(np.roll(u, 1, axis=0), -1, axis=1)
            + np.roll(np.roll(u, 1, axis=0), 1, axis=1)
        )

        grad_sq = ux * ux + uy * uy
        curvature_update = (
            uxx * uy * uy - 2.0 * ux * uy * uxy + uyy * ux * ux
        ) / (grad_sq + eps)
        u = np.clip(u + gamma * curvature_update, 0.0, 1.0)

    return u


def estimate_noise_sigma(img):
    high_freq = img - gaussian_filter(img, sigma=1.0)
    mad = np.median(np.abs(high_freq - np.median(high_freq)))
    return max(float(mad / 0.6745), 1e-3)


class MRIDenoisingDataset(Dataset):
    def __init__(self, folder_path, image_size=128, noise_type="rician", sigma_range=(0.05, 0.20)):
        self.image_paths = []
        self.labels = []
        self.noise_type = noise_type
        self.sigma_range = sigma_range

        for label, subfolder in enumerate(["no", "yes"]):
            paths = (
                glob.glob(os.path.join(folder_path, subfolder, "*.jpg"))
                + glob.glob(os.path.join(folder_path, subfolder, "*.jpeg"))
                + glob.glob(os.path.join(folder_path, subfolder, "*.png"))
            )
            self.image_paths.extend(paths)
            self.labels.extend([label] * len(paths))

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ]
        )

    def __len__(self):
        return len(self.image_paths)

    def _apply_noise(self, clean_tensor):
        sigma = torch.empty(1).uniform_(*self.sigma_range).item()

        if self.noise_type == "gaussian":
            noisy = clean_tensor + torch.randn_like(clean_tensor) * sigma
        elif self.noise_type == "speckle":
            noisy = clean_tensor + clean_tensor * torch.randn_like(clean_tensor) * sigma
        elif self.noise_type == "mixed":
            choice = np.random.choice(["gaussian", "rician", "speckle"])
            if choice == "gaussian":
                noisy = clean_tensor + torch.randn_like(clean_tensor) * sigma
            elif choice == "speckle":
                noisy = clean_tensor + clean_tensor * torch.randn_like(clean_tensor) * sigma
            else:
                noise_real = torch.randn_like(clean_tensor) * sigma
                noise_imag = torch.randn_like(clean_tensor) * sigma
                noisy = torch.sqrt((clean_tensor + noise_real) ** 2 + noise_imag ** 2)
        else:
            noise_real = torch.randn_like(clean_tensor) * sigma
            noise_imag = torch.randn_like(clean_tensor) * sigma
            noisy = torch.sqrt((clean_tensor + noise_real) ** 2 + noise_imag ** 2)

        return torch.clamp(noisy, 0.0, 1.0)

    def __getitem__(self, idx):
        clean_tensor = self.transform(Image.open(self.image_paths[idx]).convert("L"))
        noisy_tensor = self._apply_noise(clean_tensor)
        return noisy_tensor, clean_tensor, self.labels[idx]


def build_dataloaders(dataset, batch_size=8, seed=42, num_workers=0):
    indices = list(range(len(dataset)))
    labels = dataset.labels

    train_idx, temp_idx = train_test_split(
        indices, test_size=0.30, stratify=labels, random_state=seed
    )
    temp_labels = [labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(
        temp_idx, test_size=0.50, stratify=temp_labels, random_state=seed
    )

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)
    test_set = Subset(dataset, test_idx)

    pin_memory = torch.cuda.is_available()
    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

    return train_set, val_set, test_set, train_loader, val_loader, test_loader


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_model(model, loader, device):
    psnrs = []
    ssims = []
    edge_mses = []
    examples = {0: [], 1: []}

    model.eval()
    with torch.no_grad():
        for noisy, clean, label in loader:
            noisy = noisy.to(device)
            clean = clean.to(device)
            out = torch.clamp(model(noisy), 0.0, 1.0)

            p = psnr_metric(out, clean).item()
            s = ssim(out, clean, data_range=1.0).item()
            psnrs.append(p)
            ssims.append(s)
            edge_mses.extend(edge_mse_torch(out, clean))

            lv = label.item()
            if len(examples[lv]) < 2:
                examples[lv].append((clean.cpu(), noisy.cpu(), out.cpu(), s, p))

    return psnrs, ssims, edge_mses, examples


def save_loss_plot(train_losses, val_losses, train_psnrs, val_psnrs, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(train_losses, label="Train")
    ax1.plot(val_losses, label="Val")
    ax1.set_title("Combined Loss (SSIM + L1 + Gradient)")
    ax1.set_xlabel("Epoch")
    ax1.legend()
    ax1.grid(True)

    ax2.plot(train_psnrs, label="Train")
    ax2.plot(val_psnrs, label="Val")
    ax2.set_title("PSNR (dB)")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("dB")
    ax2.legend()
    ax2.grid(True)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_qualitative_plot(examples, path):
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    row = 0

    for label_id, data_list in examples.items():
        class_name = "Healthy" if label_id == 0 else "Tumor"
        for clean_img, noisy_img, out_img, s, p in data_list:
            axes[row, 0].imshow(clean_img.squeeze(), cmap="gray")
            axes[row, 0].set_title(f"{class_name} - Ground Truth")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(noisy_img.squeeze(), cmap="gray")
            axes[row, 1].set_title("Input (Noisy)")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(out_img.squeeze(), cmap="gray")
            axes[row, 2].set_title(f"Denoised | SSIM: {s:.3f} | PSNR: {p:.1f} dB")
            axes[row, 2].axis("off")
            row += 1

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def metric_summary(method, psnrs, ssims, edge_mses):
    return {
        "Method": method,
        "PSNR (dB)": np.mean(psnrs),
        "PSNR Std": np.std(psnrs),
        "SSIM": np.mean(ssims),
        "SSIM Std": np.std(ssims),
        "Edge MSE": np.mean(edge_mses),
        "Edge MSE Std": np.std(edge_mses),
    }


def record_numpy_metrics(res, key, clean, output):
    res[key]["psnr"].append(skimage_psnr(clean, output, data_range=1.0))
    res[key]["ssim"].append(skimage_ssim(clean, output, data_range=1.0))
    res[key]["edge_mse"].append(edge_mse_np(clean, output))


def run_classical_baselines(test_batches, neighbor_mode, capture_examples=2):
    res = {}
    methods = [
        "Noisy Input",
        "Gaussian Smoothing",
        "Median Filter",
        "Bilateral Filter",
        "Non-Local Means",
        "Wavelet Denoising",
        "Skimage TV",
        "Curvature Flow",
        "Classical PM",
    ]
    for method in methods:
        res[method] = {"psnr": [], "ssim": [], "edge_mse": []}

    pm_iterations = 16 if neighbor_mode == 8 else 8
    pm_gamma = 0.05 if neighbor_mode == 8 else 0.1
    curvature_iterations = pm_iterations
    curvature_gamma = pm_gamma
    examples = []

    for noisy, clean, _ in test_batches:
        n_img = noisy.squeeze().cpu().numpy()
        c_img = clean.squeeze().cpu().numpy()
        outputs = {"Clean": c_img, "Noisy Input": n_img}

        record_numpy_metrics(res, "Noisy Input", c_img, n_img)

        g_out = gaussian_filter(n_img, sigma=1.0)
        outputs["Gaussian Smoothing"] = g_out
        record_numpy_metrics(res, "Gaussian Smoothing", c_img, g_out)

        median_out = median_filter(n_img, size=3)
        outputs["Median Filter"] = median_out
        record_numpy_metrics(res, "Median Filter", c_img, median_out)

        bilateral_out = denoise_bilateral(
            n_img,
            sigma_color=0.08,
            sigma_spatial=3,
            channel_axis=None,
        )
        bilateral_out = np.clip(bilateral_out, 0.0, 1.0)
        outputs["Bilateral Filter"] = bilateral_out
        record_numpy_metrics(res, "Bilateral Filter", c_img, bilateral_out)

        sigma_est = estimate_noise_sigma(n_img)
        nlm_out = denoise_nl_means(
            n_img,
            h=0.8 * sigma_est,
            sigma=sigma_est,
            patch_size=5,
            patch_distance=6,
            fast_mode=True,
            channel_axis=None,
        )
        nlm_out = np.clip(nlm_out, 0.0, 1.0)
        outputs["Non-Local Means"] = nlm_out
        record_numpy_metrics(res, "Non-Local Means", c_img, nlm_out)

        wavelet_out = denoise_wavelet(
            n_img,
            sigma=sigma_est,
            method="BayesShrink",
            mode="soft",
            rescale_sigma=True,
            channel_axis=None,
        )
        wavelet_out = np.clip(wavelet_out, 0.0, 1.0)
        outputs["Wavelet Denoising"] = wavelet_out
        record_numpy_metrics(res, "Wavelet Denoising", c_img, wavelet_out)

        pm_out = classical_perona_malik(
            n_img, iterations=pm_iterations, kappa=0.1, gamma=pm_gamma
        )
        pm_name = f"Classical PM ({pm_iterations} iter)"
        outputs[pm_name] = pm_out
        res[pm_name] = res.pop("Classical PM") if "Classical PM" in res else res[pm_name]
        record_numpy_metrics(res, pm_name, c_img, pm_out)

        curvature_out = curvature_flow(
            n_img, iterations=curvature_iterations, gamma=curvature_gamma
        )
        curvature_name = f"Curvature Flow ({curvature_iterations} iter)"
        outputs[curvature_name] = curvature_out
        res[curvature_name] = res.pop("Curvature Flow") if "Curvature Flow" in res else res[curvature_name]
        record_numpy_metrics(res, curvature_name, c_img, curvature_out)

        tv_out = denoise_tv_chambolle(n_img, weight=0.1)
        outputs["Skimage TV"] = tv_out
        record_numpy_metrics(res, "Skimage TV", c_img, tv_out)

        if len(examples) < capture_examples:
            examples.append(outputs)

    order = [
        "Noisy Input",
        "Gaussian Smoothing",
        "Median Filter",
        "Bilateral Filter",
        "Non-Local Means",
        "Wavelet Denoising",
        "Skimage TV",
        curvature_name,
        pm_name,
    ]
    rows = [metric_summary(name, res[name]["psnr"], res[name]["ssim"], res[name]["edge_mse"]) for name in order]
    return rows, examples


def save_comparison_grid(examples, path, model_outputs=None, max_cols=6):
    if not examples:
        return

    rows = []
    for idx, outputs in enumerate(examples):
        row = dict(outputs)
        if model_outputs and idx < len(model_outputs):
            row.update(model_outputs[idx])
        rows.append(row)

    method_names = list(rows[0].keys())[:max_cols]
    fig, axes = plt.subplots(len(rows), len(method_names), figsize=(3 * len(method_names), 3 * len(rows)))
    axes = np.atleast_2d(axes)
    for row_idx, outputs in enumerate(rows):
        for col_idx, name in enumerate(method_names):
            axes[row_idx, col_idx].imshow(outputs[name], cmap="gray", vmin=0, vmax=1)
            axes[row_idx, col_idx].set_title(name)
            axes[row_idx, col_idx].axis("off")

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_metric_bar_plots(df, path):
    plot_df = df.copy()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    axes[0].barh(plot_df["Method"], plot_df["PSNR (dB)"], color="#4277b6")
    axes[0].set_title("PSNR by Method")
    axes[0].set_xlabel("PSNR (dB)")
    axes[0].invert_yaxis()
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(plot_df["Method"], plot_df["SSIM"], color="#4f9d69")
    axes[1].set_title("SSIM by Method")
    axes[1].set_xlabel("SSIM")
    axes[1].invert_yaxis()
    axes[1].grid(axis="x", alpha=0.25)

    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def save_comparison_table(test_batches, model_psnrs, model_ssims, model_edge_mses, path, neighbor_mode, extra_rows=None):
    results_data, baseline_examples = run_classical_baselines(test_batches, neighbor_mode)
    results_data.extend(extra_rows or [])
    results_data.append(
        metric_summary(
            "Unified Neural PDE (Ours)",
            model_psnrs,
            model_ssims,
            model_edge_mses,
        )
    )

    df = pd.DataFrame(results_data).round(3)
    print("\n=======================================================")
    print("                 FINAL COMPARISON TABLE                ")
    print("=======================================================")
    print(df.to_markdown(index=False))
    df.to_csv(path, index=False)
    print(f"\nSaved: {path}")
    return df, baseline_examples


def train_model(model, train_loader, val_loader, device, epochs, grad_weight, checkpoint_path, lr=1e-3):
    if isinstance(model, UnifiedNeuralPeronaMalik):
        optimizer = torch.optim.Adam(
            [
                {"params": model.guidance_encoder.parameters(), "lr": 1e-4},
                {"params": model.conduction_net.parameters(), "lr": 1e-3},
                {"params": model.refinement_net.parameters(), "lr": 1e-3},
            ]
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    l1_criterion = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(epochs, 1), eta_min=1e-5
    )

    train_losses, val_losses = [], []
    train_psnrs, val_psnrs = [], []
    best_val_loss = float("inf")

    print("Starting training...")
    for epoch in range(epochs):
        model.train()
        epoch_train_loss = 0.0
        epoch_train_psnr = 0.0

        for noisy, clean, _ in train_loader:
            noisy = noisy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            output = torch.clamp(model(noisy), 0.0, 1.0)
            loss = combined_loss(output, clean, l1_criterion, grad_weight=grad_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_train_loss += loss.item()
            epoch_train_psnr += psnr_metric(output.detach(), clean).item()

        model.eval()
        epoch_val_loss = 0.0
        epoch_val_psnr = 0.0
        with torch.no_grad():
            for noisy, clean, _ in val_loader:
                noisy = noisy.to(device)
                clean = clean.to(device)
                output = torch.clamp(model(noisy), 0.0, 1.0)
                epoch_val_loss += combined_loss(output, clean, l1_criterion, grad_weight=grad_weight).item()
                epoch_val_psnr += psnr_metric(output, clean).item()

        scheduler.step()

        avg_train = epoch_train_loss / len(train_loader)
        avg_val = epoch_val_loss / len(val_loader)
        avg_train_psnr = epoch_train_psnr / len(train_loader)
        avg_val_psnr = epoch_val_psnr / len(val_loader)

        train_losses.append(avg_train)
        val_losses.append(avg_val)
        train_psnrs.append(avg_train_psnr)
        val_psnrs.append(avg_val_psnr)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), checkpoint_path)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"Epoch [{epoch + 1:>3}/{epochs}] "
                f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | "
                f"Train PSNR: {avg_train_psnr:.2f} dB | Val PSNR: {avg_val_psnr:.2f} dB | "
                f"LR: {scheduler.get_last_lr()[0]:.2e}"
            )

    return best_val_loss, train_losses, val_losses, train_psnrs, val_psnrs


def collect_model_examples(model, test_batches, device, max_examples=2, name="Unified Neural PDE (Ours)"):
    examples = []
    model.eval()
    with torch.no_grad():
        for noisy, _, _ in test_batches[:max_examples]:
            out = torch.clamp(model(noisy.to(device)), 0.0, 1.0)
            examples.append({name: out.squeeze().cpu().numpy()})
    return examples


def evaluate_neural_baseline(train_loader, val_loader, test_batches, device, epochs, grad_weight, checkpoint_path):
    baseline = UNetDenoiser().to(device)
    best_val, _, _, _, _ = train_model(
        baseline,
        train_loader,
        val_loader,
        device,
        epochs,
        grad_weight,
        checkpoint_path,
        lr=1e-3,
    )
    baseline.load_state_dict(torch.load(checkpoint_path, map_location=device))
    psnrs, ssims, edge_mses, _ = evaluate_model(baseline, test_batches, device)
    print(f"U-Net baseline best val loss: {best_val:.4f}")
    return metric_summary("Plain U-Net Baseline", psnrs, ssims, edge_mses), baseline


def run_noise_sweep(model, dataset_path, args, device, path):
    rows = []
    sweep_types = [x.strip() for x in args.noise_sweep_types.split(",") if x.strip()]
    sweep_sigmas = [float(x.strip()) for x in args.noise_sweep_sigmas.split(",") if x.strip()]

    for noise_type in sweep_types:
        for sigma in sweep_sigmas:
            set_seed(args.seed)
            dataset = MRIDenoisingDataset(
                dataset_path,
                image_size=args.image_size,
                noise_type=noise_type,
                sigma_range=(sigma, sigma),
            )
            _, _, _, _, _, test_loader = build_dataloaders(
                dataset, batch_size=args.batch_size, seed=args.seed, num_workers=0
            )
            test_batches = list(test_loader)
            if args.eval_limit is not None:
                test_batches = test_batches[: args.eval_limit]
            psnrs, ssims, edge_mses, _ = evaluate_model(model, test_batches, device)
            row = metric_summary("Unified Neural PDE (Ours)", psnrs, ssims, edge_mses)
            row["Noise Type"] = noise_type
            row["Sigma"] = sigma
            rows.append(row)

    df = pd.DataFrame(rows).round(3)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return df


def make_unified_model(args, neighbor_mode=None, use_refinement=None, use_unet_guidance=None):
    mode = args.neighbor_mode if neighbor_mode is None else neighbor_mode
    iterations = args.iterations if args.iterations is not None else (16 if mode == 8 else 10)
    lambda_param = args.lambda_param if args.lambda_param is not None else (0.05 if mode == 8 else 0.1)
    return UnifiedNeuralPeronaMalik(
        iterations=iterations,
        lambda_param=lambda_param,
        neighbor_mode=mode,
        use_refinement=not args.no_refinement if use_refinement is None else use_refinement,
        use_unet_guidance=not args.no_unet_guidance if use_unet_guidance is None else use_unet_guidance,
    )


def run_ablation_suite(train_loader, val_loader, test_batches, args, device, checkpoint_dir, path):
    configs = [
        ("Full Model", args.neighbor_mode, True, True),
        ("No MiniUNet Guidance", args.neighbor_mode, True, False),
        ("No Residual Refinement", args.neighbor_mode, False, True),
        ("4-Neighbor Diffusion", 4, True, True),
    ]
    rows = []

    for name, neighbor_mode, use_refinement, use_unet_guidance in configs:
        print(f"\nRunning ablation: {name}")
        model = make_unified_model(
            args,
            neighbor_mode=neighbor_mode,
            use_refinement=use_refinement,
            use_unet_guidance=use_unet_guidance,
        ).to(device)
        checkpoint_path = os.path.join(
            checkpoint_dir,
            f"ablation_{name.lower().replace(' ', '_').replace('-', '_')}.pth",
        )
        best_val, _, _, _, _ = train_model(
            model,
            train_loader,
            val_loader,
            device,
            args.ablation_epochs,
            args.grad_weight,
            checkpoint_path,
        )
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        psnrs, ssims, edge_mses, _ = evaluate_model(model, test_batches, device)
        row = metric_summary(name, psnrs, ssims, edge_mses)
        row["Best Val Loss"] = best_val
        row["Neighbor Mode"] = neighbor_mode
        row["Refinement"] = use_refinement
        row["MiniUNet Guidance"] = use_unet_guidance
        rows.append(row)

    df = pd.DataFrame(rows).round(3)
    df.to_csv(path, index=False)
    print(f"Saved: {path}")
    return df


def git_commit_hash():
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=BASE_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def gpu_names():
    if not torch.cuda.is_available():
        return []
    return [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]


def save_run_metadata(args, path, train_count, val_count, test_count, checkpoint_path, comparison_path):
    metadata = {
        "timestamp_utc": datetime.datetime.now(datetime.UTC).replace(microsecond=0).isoformat(),
        "git_commit": git_commit_hash(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpus": gpu_names(),
        "dataset_path": DATASET_PATH,
        "dataset_split": {
            "train": train_count,
            "validation": val_count,
            "test": test_count,
            "eval_limit": args.eval_limit,
        },
        "checkpoint_path": checkpoint_path,
        "comparison_table": comparison_path,
        "args": vars(args),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, sort_keys=True)
    print(f"Saved: {path}")
    return metadata


def apply_json_config(args):
    if not args.config:
        return args
    config_path = resolve_repo_path(args.config)
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    for key, value in config.items():
        attr = key.replace("-", "_")
        if hasattr(args, attr):
            setattr(args, attr, value)
    return args


def parse_args():
    parser = argparse.ArgumentParser(description="Unified Neural Anisotropic Diffusion")
    parser.add_argument("--config", type=str, default=None, help="Optional JSON config file with argument names as keys.")
    parser.add_argument("--neighbor-mode", type=int, default=8, choices=[4, 8])
    parser.add_argument("--noise-type", type=str, default="rician", choices=["gaussian", "rician", "speckle", "mixed"])
    parser.add_argument("--sigma-min", type=float, default=0.05)
    parser.add_argument("--sigma-max", type=float, default=0.20)
    parser.add_argument("--iterations", type=int, default=None)
    parser.add_argument("--lambda-param", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--grad-weight", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default=RESULTS_DIR)
    parser.add_argument("--checkpoint-dir", type=str, default=CHECKPOINT_DIR)
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint path for eval-only or explicit loading.")
    parser.add_argument("--eval-only", action="store_true", help="Skip training and evaluate a saved checkpoint.")
    parser.add_argument("--eval-limit", type=int, default=None, help="Evaluate only the first N held-out test examples.")
    parser.add_argument("--train-unet-baseline-epochs", type=int, default=0, help="Train and include a plain U-Net denoiser baseline.")
    parser.add_argument("--noise-sweep", action="store_true", help="Evaluate the saved/trained model across fixed noise levels.")
    parser.add_argument("--noise-sweep-types", type=str, default="gaussian,rician,speckle,mixed")
    parser.add_argument("--noise-sweep-sigmas", type=str, default="0.05,0.10,0.15,0.20")
    parser.add_argument("--run-ablation-suite", action="store_true", help="Train and evaluate common architecture ablations.")
    parser.add_argument("--ablation-epochs", type=int, default=20)
    parser.add_argument("--no-refinement", action="store_true")
    parser.add_argument("--no-unet-guidance", action="store_true")
    return apply_json_config(parser.parse_args())


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print(f"Running on: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    dataset = MRIDenoisingDataset(
        DATASET_PATH,
        image_size=args.image_size,
        noise_type=args.noise_type,
        sigma_range=(args.sigma_min, args.sigma_max),
    )

    if len(dataset) == 0:
        raise RuntimeError(f"No images found at: {DATASET_PATH}")

    train_set, val_set, test_set, train_loader, val_loader, test_loader = build_dataloaders(
        dataset, batch_size=args.batch_size, seed=args.seed, num_workers=0
    )
    print(
        f"Dataset split - Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}"
    )

    checkpoint_dir = resolve_repo_path(args.checkpoint_dir)
    results_dir = resolve_repo_path(args.results_dir)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    model = make_unified_model(args).to(device)

    best_model_path = os.path.join(checkpoint_dir, "unified_model.pth")
    if args.checkpoint:
        best_model_path = resolve_repo_path(args.checkpoint)
    loss_plot_path = os.path.join(results_dir, "unified_loss_curves.png")
    qual_plot_path = os.path.join(results_dir, "unified_qualitative_results.png")
    comparison_grid_path = os.path.join(results_dir, "unified_full_comparison_grid.png")
    metric_plot_path = os.path.join(results_dir, "unified_metric_bars.png")
    comparison_path = os.path.join(results_dir, "unified_comparison_table.csv")
    metadata_path = os.path.join(results_dir, "run_metadata.json")

    if args.eval_only:
        if not os.path.exists(best_model_path):
            raise FileNotFoundError(f"Checkpoint not found for --eval-only: {best_model_path}")
        print(f"Skipping training. Loading checkpoint: {best_model_path}")
    else:
        best_val_loss, train_losses, val_losses, train_psnrs, val_psnrs = train_model(
            model,
            train_loader,
            val_loader,
            device,
            args.epochs,
            args.grad_weight,
            best_model_path,
        )
        print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")
        save_loss_plot(train_losses, val_losses, train_psnrs, val_psnrs, loss_plot_path)
        print(f"Saved: {loss_plot_path}")

    model.load_state_dict(torch.load(best_model_path, map_location=device))
    model.eval()

    test_batches = list(test_loader)
    if args.eval_limit is not None:
        test_batches = test_batches[: args.eval_limit]
        print(f"Evaluation limited to first {len(test_batches)} test examples.")
    model_psnrs, model_ssims, model_edge_mses, examples = evaluate_model(model, test_batches, device)
    save_qualitative_plot(examples, qual_plot_path)
    print(f"Saved: {qual_plot_path}")

    extra_rows = []
    extra_models = []
    if args.train_unet_baseline_epochs > 0:
        unet_checkpoint_path = os.path.join(checkpoint_dir, "unet_baseline.pth")
        print(f"\nTraining plain U-Net baseline for {args.train_unet_baseline_epochs} epochs...")
        unet_row, unet_model = evaluate_neural_baseline(
            train_loader,
            val_loader,
            test_batches,
            device,
            args.train_unet_baseline_epochs,
            args.grad_weight,
            unet_checkpoint_path,
        )
        extra_rows.append(unet_row)
        extra_models = collect_model_examples(unet_model, test_batches, device, name="Plain U-Net Baseline")

    comparison_df, baseline_examples = save_comparison_table(
        test_batches,
        model_psnrs,
        model_ssims,
        model_edge_mses,
        comparison_path,
        args.neighbor_mode,
        extra_rows=extra_rows,
    )
    save_metric_bar_plots(comparison_df, metric_plot_path)
    print(f"Saved: {metric_plot_path}")

    model_examples = collect_model_examples(model, test_batches, device)
    merged_model_examples = []
    for idx, model_example in enumerate(model_examples):
        merged = dict(model_example)
        if idx < len(extra_models):
            merged.update(extra_models[idx])
        merged_model_examples.append(merged)
    save_comparison_grid(baseline_examples, comparison_grid_path, merged_model_examples, max_cols=12)
    print(f"Saved: {comparison_grid_path}")
    save_run_metadata(
        args,
        metadata_path,
        len(train_set),
        len(val_set),
        len(test_set),
        best_model_path,
        comparison_path,
    )

    if args.noise_sweep:
        sweep_path = os.path.join(results_dir, "unified_noise_sweep.csv")
        run_noise_sweep(model, DATASET_PATH, args, device, sweep_path)

    if args.run_ablation_suite:
        ablation_path = os.path.join(results_dir, "unified_ablation_suite.csv")
        run_ablation_suite(
            train_loader,
            val_loader,
            test_batches,
            args,
            device,
            checkpoint_dir,
            ablation_path,
        )


if __name__ == "__main__":
    main()
