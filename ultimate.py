# ============================================================
# Neural Anisotropic Diffusion — SOTA Hybrid Model
# 8-Neighbor PDE + U-Net Guidance + Rician Noise + Refinement
# Ultra-Stable Edition (Lambda = 0.05, Iterations = 16)
# ============================================================
import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt

import os
import glob
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

# Classical Baseline Imports
from scipy.ndimage import gaussian_filter
from skimage.restoration import denoise_tv_chambolle
from skimage.metrics import structural_similarity as skimage_ssim
from skimage.metrics import peak_signal_noise_ratio as skimage_psnr


# ==========================================
# 1. THE SOTA ARCHITECTURE
# ==========================================
class MiniUNet(nn.Module):
    """Global context extractor to guide the local PDE diffusion."""
    def __init__(self, in_ch=1, out_ch=8):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(in_ch, 16, 3, padding=1), nn.ReLU(inplace=True))
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(inplace=True))
        
        self.up   = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.dec  = nn.Sequential(
            nn.Conv2d(32 + 16, 16, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(16, out_ch, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        d  = self.dec(torch.cat([self.up(e2), e1], dim=1))
        return d


class NeuralPeronaMalikSOTA(nn.Module):
    # OPTION A APPLIED: Conservative lambda (0.05) and increased iterations (16)
    def __init__(self, iterations=16, lambda_param=0.05, guidance_channels=8):
        super().__init__()
        # 8-neighbor absolute max bound is ~0.146, we use 0.05 for ultra-stability
        assert lambda_param <= 0.125, "lambda_param > 0.125 violates 8-neighbor stability"

        self.iterations = iterations
        self.lambda_param = lambda_param

        # Global U-Net Guidance
        self.guidance_encoder = MiniUNet(in_ch=1, out_ch=guidance_channels)

        # 8-Neighbor Conduction Net (8 gradients + guidance channels)
        self.conduction_net = nn.Sequential(
            nn.Conv2d(8 + guidance_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Residual Refinement Net
        self.refinement_net = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x_smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        
        # Detach to prevent gradients blowing up across 16 iterations
        guidance_features = self.guidance_encoder(x_smooth).detach()

        # --- The 8-Neighbor PDE Loop ---
        for _ in range(self.iterations):
            x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
            
            # Cardinal Gradients
            grad_N = x_pad[:, :, :-2, 1:-1] - x
            grad_S = x_pad[:, :, 2:,  1:-1] - x
            grad_E = x_pad[:, :, 1:-1, 2:]  - x
            grad_W = x_pad[:, :, 1:-1, :-2] - x
            
            # Diagonal Gradients (Scaled by 1/sqrt(2))
            grad_NW = (x_pad[:, :, :-2, :-2] - x) * 0.707
            grad_NE = (x_pad[:, :, :-2, 2:]  - x) * 0.707
            grad_SW = (x_pad[:, :, 2:,  :-2] - x) * 0.707
            grad_SE = (x_pad[:, :, 2:,  2:]  - x) * 0.707

            combined = torch.cat([
                grad_N, grad_S, grad_E, grad_W, 
                grad_NW, grad_NE, grad_SW, grad_SE, 
                guidance_features
            ], dim=1)
            
            c = self.conduction_net(combined)
            c_N, c_S, c_E, c_W, c_NW, c_NE, c_SW, c_SE = torch.split(c, 1, dim=1)

            x = x + self.lambda_param * (
                c_N*grad_N + c_S*grad_S + c_E*grad_E + c_W*grad_W +
                c_NW*grad_NW + c_NE*grad_NE + c_SW*grad_SW + c_SE*grad_SE
            )

        # --- Residual Refinement Stage ---
        refined_output = x + self.refinement_net(x)
        return refined_output


# ==========================================
# 2. PURE STRUCTURAL LOSS
# ==========================================
def psnr_metric(pred, target):
    mse = F.mse_loss(pred, target)
    return 10 * torch.log10(1.0 / (mse + 1e-8))

def combined_loss(output, clean, l1_fn):
    ssim_l = 1.0 - ssim(output, clean, data_range=1.0)
    l1_l   = l1_fn(output, clean)
    return ssim_l + l1_l


# ==========================================
# 3. CLASSICAL BASELINES & DATASET (RICIAN NOISE)
# ==========================================
def classical_perona_malik(img, iterations=16, kappa=0.1, gamma=0.05):
    """Updated to match the 16 iteration / 0.05 gamma setup for fair comparison."""
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
        
        # True Rician Noise Simulation
        sigma = torch.empty(1).uniform_(0.05, 0.20).item()
        noise_real = torch.randn_like(clean_tensor) * sigma
        noise_imag = torch.randn_like(clean_tensor) * sigma
        
        noisy_tensor = torch.sqrt((clean_tensor + noise_real)**2 + noise_imag**2)
        noisy_tensor = torch.clamp(noisy_tensor, 0., 1.)
        
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

    train_set = Subset(full_dataset, train_idx)
    val_set   = Subset(full_dataset, val_idx)
    test_set  = Subset(full_dataset, test_idx)

    train_loader = DataLoader(train_set, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_set, batch_size=8, shuffle=False, num_workers=2, pin_memory=True)
    test_loader  = DataLoader(test_set, batch_size=1, shuffle=False)

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('results', exist_ok=True)

    # --- Initialize SOTA Model ---
    epochs = 300
    model = NeuralPeronaMalikSOTA(iterations=16, lambda_param=0.05).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    l1_criterion = nn.L1Loss()
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    best_val_loss = float('inf')
    best_model_path = 'checkpoints/sota_hybrid_model.pth'

    print("\n--- Training SOTA Hybrid Model ---")
    for epoch in range(epochs):
        model.train()
        for noisy, clean, _ in train_loader:
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

            if len(examples[label.item()]) < 2:
                examples[label.item()].append((clean.cpu(), noisy.cpu(), out.cpu(), s, p))

    fig, axes = plt.subplots(4, 3, figsize=(12, 16))
    row = 0
    for label_id, data_list in examples.items():
        class_name = "Healthy" if label_id == 0 else "Tumor"
        for clean_img, noisy_img, out_img, s, p in data_list:
            axes[row, 0].imshow(clean_img.squeeze(), cmap='gray')
            axes[row, 0].set_title(f"{class_name} — Ground Truth"); axes[row, 0].axis('off')

            axes[row, 1].imshow(noisy_img.squeeze(), cmap='gray')
            axes[row, 1].set_title("Input (Rician Noise)"); axes[row, 1].axis('off')

            axes[row, 2].imshow(out_img.squeeze(), cmap='gray')
            axes[row, 2].set_title(f"Denoised | SSIM: {s:.3f} | PSNR: {p:.1f} dB")
            axes[row, 2].axis('off')
            row += 1

    plt.tight_layout()
    plt.savefig('results/sota_qualitative_results.png', dpi=150)
    plt.close()
    print("Saved: results/sota_qualitative_results.png")

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
        
        pm_out = classical_perona_malik(n_img, iterations=16, kappa=0.1, gamma=0.05)
        res["Classic PM"]["psnr"].append(skimage_psnr(c_img, pm_out, data_range=1.0))
        res["Classic PM"]["ssim"].append(skimage_ssim(c_img, pm_out, data_range=1.0))

        tv_out = denoise_tv_chambolle(n_img, weight=0.1)
        res["Skimage TV"]["psnr"].append(skimage_psnr(c_img, tv_out, data_range=1.0))
        res["Skimage TV"]["ssim"].append(skimage_ssim(c_img, tv_out, data_range=1.0))

    baseline_results = {k: (np.mean(v["psnr"]), np.mean(v["ssim"])) for k, v in res.items()}

    results_data = [
        {"Method": "Gaussian Smoothing", "PSNR (dB)": baseline_results["Gaussian"][0], "SSIM": baseline_results["Gaussian"][1]},
        {"Method": "Skimage TV", "PSNR (dB)": baseline_results["Skimage TV"][0], "SSIM": baseline_results["Skimage TV"][1]},
        {"Method": "Classical PM (16 iter)", "PSNR (dB)": baseline_results["Classic PM"][0], "SSIM": baseline_results["Classic PM"][1]},
        {"Method": "SOTA Neural PDE Hybrid (Ours)", "PSNR (dB)": np.mean(psnrs), "SSIM": np.mean(ssims)},
    ]

    df = pd.DataFrame(results_data).round(3)
    
    print("\n=======================================================")
    print("                FINAL COMPARISON TABLE                 ")
    print("=======================================================")
    print(df.to_markdown(index=False))
    
    df.to_csv("results/sota_comparison_table.csv", index=False)
    print("\nSaved: results/sota_comparison_table.csv")