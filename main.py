# ============================================================
# Neural Anisotropic Diffusion — Final Champion Model
# Optimized for pure L1 + SSIM PDE unrolling
# ============================================================
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt

import os
import glob
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from PIL import Image
from pytorch_msssim import ssim

<<<<<<< HEAD
# Suppress harmless torchvision.io libjpeg warning (PIL is used instead)
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")
=======
# Classical Baseline Imports
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import structural_similarity as skimage_ssim
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr
>>>>>>> 2a07c1c (.)


# ==========================================
# 1. OPTIMIZED MODEL ARCHITECTURE
# ==========================================
class NeuralPeronaMalik(nn.Module):
    def __init__(self, iterations=8, lambda_param=0.1, guidance_channels=8):
        super().__init__()
<<<<<<< HEAD
        # FIX: Assert stability bound for forward Euler diffusion
        assert lambda_param <= 0.25, "lambda_param > 0.25 violates 4-neighbor diffusion stability"
        self.iterations = iterations
        self.lambda_param = lambda_param

        # FIX: Removed BatchNorm before Sigmoid (compresses output to ~0.5, kills dynamic range)
        # FIX: Removed inplace=True from all ReLUs (autograd risk in iterative loop)
        # FIX: Added dilated conv for wider receptive field (~9x9) in guidance encoder
=======
        assert lambda_param <= 0.25, "lambda_param > 0.25 violates stability"

        self.iterations = iterations
        self.lambda_param = lambda_param

        # The Guidance Encoder: Pre-smoothes and finds anatomical edges
>>>>>>> 2a07c1c (.)
        self.guidance_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, guidance_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

<<<<<<< HEAD
        # FIX: Channels parameterized via comment (not hardcoded)
        # Input: 4 directional gradients + guidance_channels = (4 + guidance_channels) total
=======
        # The Conduction Network: Learns the 4-neighbor diffusion coefficients
>>>>>>> 2a07c1c (.)
        self.conduction_net = nn.Sequential(
            nn.Conv2d(4 + guidance_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 4, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
<<<<<<< HEAD
        # FIX: Pre-smooth noisy input before guidance extraction
        # Prevents structural prior from being dominated by noise artifacts
        x_smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)

        # FIX: .detach() prevents guidance encoder from receiving
        # ~iterations× amplified gradients vs conduction_net
=======
        x_smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
>>>>>>> 2a07c1c (.)
        guidance_features = self.guidance_encoder(x_smooth).detach()

        for _ in range(self.iterations):
            x_pad  = F.pad(x, (1, 1, 1, 1), mode='replicate')
            grad_N = x_pad[:, :, :-2, 1:-1] - x
            grad_S = x_pad[:, :, 2:,  1:-1] - x
            grad_E = x_pad[:, :, 1:-1, 2:]  - x
            grad_W = x_pad[:, :, 1:-1, :-2] - x

            combined = torch.cat([grad_N, grad_S, grad_E, grad_W, guidance_features], dim=1)
            c = self.conduction_net(combined)
            c_N, c_S, c_E, c_W = torch.split(c, 1, dim=1)

            x = x + self.lambda_param * (
                c_N * grad_N + c_S * grad_S +
                c_E * grad_E + c_W * grad_W
            )
        return x


# ==========================================
# 2. PURE STRUCTURAL LOSS
# ==========================================
<<<<<<< HEAD
def gradient_loss(pred, target):
    """Edge-preserving loss: penalizes blurring of structural boundaries."""
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_tgt  = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :]  - pred[:, :, :-1, :]
    dy_tgt  = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.mse_loss(dx_pred, dx_tgt) + F.mse_loss(dy_pred, dy_tgt)


def psnr(pred, target):
    mse = F.mse_loss(pred, target)
    return 10 * torch.log10(1.0 / (mse + 1e-8))


def combined_loss(output, clean, l1_criterion):
    ssim_loss = 1 - ssim(output, clean, data_range=1.0)
    l1        = l1_criterion(output, clean)
    grad      = gradient_loss(output, clean)
    return ssim_loss + 1.0 * l1 + 0.25 * grad
=======
def psnr_metric(pred, target):
    mse = F.mse_loss(pred, target)
    return 10 * torch.log10(1.0 / (mse + 1e-8))

def combined_loss(output, clean, l1_fn):
    """
    Stripped of VGG and Gradient losses. 
    Relies purely on L1 (pixel fidelity) and SSIM (structural/perceptual fidelity).
    """
    ssim_l = 1.0 - ssim(output, clean, data_range=1.0)
    l1_l   = l1_fn(output, clean)
    return ssim_l + l1_l
>>>>>>> 2a07c1c (.)


# ==========================================
# 3. CLASSICAL BASELINES & DATASET
# ==========================================
def classical_perona_malik(img, iterations=8, kappa=0.1, gamma=0.1):
    u = img.copy()
    for _ in range(iterations):
        n = np.roll(u, -1, axis=0) - u
        s = np.roll(u, 1, axis=0) - u
        e = np.roll(u, -1, axis=1) - u
        w = np.roll(u, 1, axis=1) - u
        
        c_n = 1 / (1 + (np.abs(n)/kappa)**2)
        c_s = 1 / (1 + (np.abs(s)/kappa)**2)
        c_e = 1 / (1 + (np.abs(e)/kappa)**2)
        c_w = 1 / (1 + (np.abs(w)/kappa)**2)
        
        u = u + gamma * (c_n*n + c_s*s + c_e*e + c_w*w)
    return np.clip(u, 0, 1)

class MRIDenoisingDataset(Dataset):
    def __init__(self, folder_path, image_size=128):
        self.image_paths, self.labels = [], []
        for label, subfolder in enumerate(['no', 'yes']):
            paths = (glob.glob(os.path.join(folder_path, subfolder, '*.jpg'))  +
                     glob.glob(os.path.join(folder_path, subfolder, '*.jpeg')) +
                     glob.glob(os.path.join(folder_path, subfolder, '*.png')))
            self.image_paths.extend(paths)
            self.labels.extend([label] * len(paths))

        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        clean_tensor = self.transform(Image.open(self.image_paths[idx]).convert('L'))
<<<<<<< HEAD
        sigma = torch.empty(1).uniform_(0.10, 0.20).item()
        noise = torch.randn_like(clean_tensor) * sigma
        noisy_tensor = torch.clamp(clean_tensor + noise, 0., 1.)
=======
        sigma = torch.empty(1).uniform_(0.05, 0.25).item()
        noisy_tensor = torch.clamp(clean_tensor + torch.randn_like(clean_tensor) * sigma, 0., 1.)
>>>>>>> 2a07c1c (.)
        return noisy_tensor, clean_tensor, self.labels[idx]


# ==========================================
# 4. SETUP & TRAINING RUNNER
# ==========================================
if __name__ == "__main__":
    device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on: {device}")

    dataset_path = '/home/sofa/host_dir/nad/neural-anisotropic-diffusion/brain_tumor_dataset'
    full_dataset = MRIDenoisingDataset(dataset_path)

    train_idx, temp_idx = train_test_split(list(range(len(full_dataset))), test_size=0.30, stratify=full_dataset.labels, random_state=42)
    temp_labels = [full_dataset.labels[i] for i in temp_idx]
    val_idx, test_idx = train_test_split(temp_idx, test_size=0.50, stratify=temp_labels, random_state=42)

<<<<<<< HEAD
# FIX: Stratified split to preserve class balance across train/val/test
indices = list(range(len(full_dataset)))
labels  = full_dataset.labels

train_idx, temp_idx = train_test_split(
    indices, test_size=0.30, stratify=labels, random_state=42
)
temp_labels = [labels[i] for i in temp_idx]
val_idx, test_idx = train_test_split(
    temp_idx, test_size=0.50, stratify=temp_labels, random_state=42
)

train_set = Subset(full_dataset, train_idx)
val_set   = Subset(full_dataset, val_idx)
test_set  = Subset(full_dataset, test_idx)

print(f"Train: {len(train_set)} | Val: {len(val_set)} | Test: {len(test_set)}")

# FIX: num_workers=0 — macOS 'spawn' multiprocessing causes worker crash with num_workers > 0.
# pin_memory removed — only benefits CUDA, has no effect on MPS.
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_set,   batch_size=8, shuffle=False)
test_loader  = DataLoader(test_set,  batch_size=1, shuffle=False)

model = NeuralPeronaMalik(iterations=10, lambda_param=0.1).to(device)

optimizer = torch.optim.Adam([
    {'params': model.guidance_encoder.parameters(), 'lr': 1e-4},
    {'params': model.conduction_net.parameters(),   'lr': 1e-3}
])

l1_criterion = nn.L1Loss()
epochs      = 150

# FIX: Cosine annealing scheduler to avoid plateau from fixed LR
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)


# ==========================================
# 5. TRAINING LOOP
# ==========================================
train_losses, val_losses = [], []
train_psnrs,  val_psnrs  = [], []
best_val_loss = float('inf')

print("Starting training...")
for epoch in range(epochs):
    model.train()
    epoch_train_loss, epoch_train_psnr = 0.0, 0.0
    for noisy, clean, _ in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)

        # FIX: zero_grad at top of loop (before forward pass)
        optimizer.zero_grad()
        output = model(noisy)
        output = torch.clamp(output, 0., 1.)          # stabilize SSIM computation
        loss   = combined_loss(output, clean, l1_criterion)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # gradient clipping
        optimizer.step()
        epoch_train_loss += loss.item()
        epoch_train_psnr += psnr(output.detach(), clean).item()

    model.eval()
    epoch_val_loss, epoch_val_psnr = 0.0, 0.0
    with torch.no_grad():
        for noisy, clean, _ in val_loader:
=======
    train_set = Subset(full_dataset, train_idx)
    val_set   = Subset(full_dataset, val_idx)
    test_set  = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=1, shuffle=False)

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # --- Initialize Champion Model ---
    epochs = 300
    model = NeuralPeronaMalik(iterations=8).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    l1_criterion = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float('inf')
    best_model_path = 'checkpoints/champion_model.pth'

    print("\n--- Training Champion Model ---")
    for epoch in range(epochs):
        model.train()
        for noisy, clean, _ in train_loader:
>>>>>>> 2a07c1c (.)
            noisy, clean = noisy.to(device), clean.to(device)
            optimizer.zero_grad()
            output = torch.clamp(model(noisy), 0., 1.)
            loss = combined_loss(output, clean, l1_criterion)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.eval()
        epoch_val_loss = 0.0
        with torch.no_grad():
            for noisy, clean, _ in val_loader:
                noisy, clean = noisy.to(device), clean.to(device)
                output = torch.clamp(model(noisy), 0., 1.)
                epoch_val_loss += combined_loss(output, clean, l1_criterion).item()
        
        avg_val = epoch_val_loss / len(val_loader)
        scheduler.step()

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), best_model_path)
            
        if (epoch + 1) % 50 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Val Loss: {avg_val:.4f}")

    print(f"Training Complete. Best Val Loss: {best_val_loss:.4f}")

    # ==========================================
    # 5. FINAL EVALUATION & VISUALIZATION
    # ==========================================
    print("\n--- Running Final Evaluations ---")
    model.load_state_dict(torch.load(best_model_path, map_location=device, weights_only=True))
    model.eval()
    
    psnrs, ssims = [], []
    examples = {0: [], 1: []}
    
    with torch.no_grad():
        for noisy, clean, label in test_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            out = torch.clamp(model(noisy), 0., 1.)
            
            p = psnr_metric(out, clean).item()
            s = ssim(out, clean, data_range=1.0).item()
            psnrs.append(p)
            ssims.append(s)

<<<<<<< HEAD
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f} | "
              f"Train PSNR: {avg_train_psnr:.2f}dB | Val PSNR: {avg_val_psnr:.2f}dB | "
              f"LR: {scheduler.get_last_lr()[0]:.6f}")
=======
            # Save examples for visualization
            lv = label.item()
            if len(examples[lv]) < 2:
                examples[lv].append((clean.cpu(), noisy.cpu(), out.cpu(), s, p))
>>>>>>> 2a07c1c (.)

    # --- Generate Qualitative Results Image ---
    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    row = 0
    for label_id, data_list in examples.items():
        class_name = "Healthy" if label_id == 0 else "Tumor"
        for clean_img, noisy_img, out_img, s, p in data_list:
            axes[row, 0].imshow(clean_img.squeeze(), cmap='gray')
            axes[row, 0].set_title(f"{class_name} — Ground Truth"); axes[row, 0].axis('off')

            axes[row, 1].imshow(noisy_img.squeeze(), cmap='gray')
            axes[row, 1].set_title("Input (Noisy)"); axes[row, 1].axis('off')

<<<<<<< HEAD
# ==========================================
# 6. LOSS + PSNR CURVES
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(train_losses, label='Train'); ax1.plot(val_losses, label='Val')
ax1.set_title("Loss (SSIM + L1 + Gradient)")
ax1.set_xlabel("Epoch"); ax1.legend(); ax1.grid(True)
ax2.plot(train_psnrs, label='Train'); ax2.plot(val_psnrs, label='Val')
ax2.set_title("PSNR (dB)")
ax2.set_xlabel("Epoch"); ax2.set_ylabel("dB"); ax2.legend(); ax2.grid(True)
plt.tight_layout()
plt.savefig('loss_curves.png', dpi=150)
plt.show()

model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
model.eval()

examples = {0: [], 1: []}
with torch.no_grad():
    for noisy, clean, label in test_set:
        lv = label if isinstance(label, int) else label.item()
        if len(examples[lv]) < 2:
            noisy_in = noisy.unsqueeze(0).to(device)
            out      = torch.clamp(model(noisy_in), 0., 1.).cpu()
            s = ssim(out, clean.unsqueeze(0), data_range=1.0).item()
            p = psnr(out, clean.unsqueeze(0)).item()
            examples[lv].append((clean, noisy, out.squeeze(0), s, p))
        if len(examples[0]) == 2 and len(examples[1]) == 2:
            break

fig, axes = plt.subplots(4, 3, figsize=(12, 16))
row = 0
for label_id, data_list in examples.items():
    class_name = "Healthy" if label_id == 0 else "Tumor"
    for clean, noisy, out, s, p in data_list:
        axes[row, 0].imshow(clean.squeeze(), cmap='gray')
        axes[row, 0].set_title(f"{class_name} — Ground Truth")
        axes[row, 0].axis('off')

        axes[row, 1].imshow(noisy.squeeze(), cmap='gray')
        axes[row, 1].set_title("Input (Noisy)")
        axes[row, 1].axis('off')

        axes[row, 2].imshow(out.squeeze(), cmap='gray')
        axes[row, 2].set_title(f"Denoised | SSIM: {s:.3f} | PSNR: {p:.1f} dB")
        axes[row, 2].axis('off')

        row += 1

plt.tight_layout()
plt.savefig('qualitative_results.png', dpi=150)
plt.show()
=======
            axes[row, 2].imshow(out_img.squeeze(), cmap='gray')
            axes[row, 2].set_title(f"Denoised | SSIM: {s:.3f} | PSNR: {p:.1f} dB")
            axes[row, 2].axis('off')
            row += 1

    plt.tight_layout()
    plt.savefig('results/champion_qualitative_results.png', dpi=150)
    plt.close()
    print("Saved: results/champion_qualitative_results.png")

    # --- Evaluate Baselines & Generate Final CSV ---
    res = {"Gaussian": {"psnr": [], "ssim": []},
           "Classic PM": {"psnr": [], "ssim": []},
           "Skimage TV": {"psnr": [], "ssim": []}}
    
    for noisy, clean, _ in test_loader:
        n_img = noisy.squeeze().numpy()
        c_img = clean.squeeze().numpy()
        
        g_out = gaussian_filter(n_img, sigma=1.0)
        res["Gaussian"]["psnr"].append(skimage_psnr(c_img, g_out, data_range=1.0))
        res["Gaussian"]["ssim"].append(skimage_ssim(c_img, g_out, data_range=1.0))
        
        pm_out = classical_perona_malik(n_img, iterations=8, kappa=0.1, gamma=0.1)
        res["Classic PM"]["psnr"].append(skimage_psnr(c_img, pm_out, data_range=1.0))
        res["Classic PM"]["ssim"].append(skimage_ssim(c_img, pm_out, data_range=1.0))

        tv_out = denoise_tv_chambolle(n_img, weight=0.1)
        res["Skimage TV"]["psnr"].append(skimage_psnr(c_img, tv_out, data_range=1.0))
        res["Skimage TV"]["ssim"].append(skimage_ssim(c_img, tv_out, data_range=1.0))

    baseline_results = {k: (np.mean(v["psnr"]), np.mean(v["ssim"])) for k, v in res.items()}

    results_data = [
        {"Method": "Gaussian Smoothing", "PSNR (dB)": baseline_results["Gaussian"][0], "SSIM": baseline_results["Gaussian"][1]},
        {"Method": "Skimage TV", "PSNR (dB)": baseline_results["Skimage TV"][0], "SSIM": baseline_results["Skimage TV"][1]},
        {"Method": "Classical PM (8 iter)", "PSNR (dB)": baseline_results["Classic PM"][0], "SSIM": baseline_results["Classic PM"][1]},
        {"Method": "Neural Perona-Malik (Ours)", "PSNR (dB)": np.mean(psnrs), "SSIM": np.mean(ssims)},
    ]

    df = pd.DataFrame(results_data).round(3)
    
    print("\n=======================================================")
    print("                FINAL COMPARISON TABLE                 ")
    print("=======================================================")
    print(df.to_markdown(index=False))
    
    df.to_csv("results/final_comparison_table.csv", index=False)
    print("\nSaved: results/final_comparison_table.csv")
>>>>>>> 2a07c1c (.)
