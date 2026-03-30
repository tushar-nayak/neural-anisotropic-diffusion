import os
import glob
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_msssim import ssim

# Suppress harmless torchvision.io libjpeg warning (PIL is used instead)
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


# ==========================================
# 1. THE ARCHITECTURE
# ==========================================
class NeuralPeronaMalik(nn.Module):
    def __init__(self, iterations=10, lambda_param=0.1, guidance_channels=8):
        super().__init__()
        assert lambda_param <= 0.25, "lambda_param > 0.25 violates 4-neighbor diffusion stability"
        self.iterations = iterations
        self.lambda_param = lambda_param

        self.guidance_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=2, dilation=2),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, guidance_channels, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

        # Input: 4 directional gradients + guidance_channels = (4 + guidance_channels) total
        self.conduction_net = nn.Sequential(
            nn.Conv2d(4 + guidance_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(16, 4, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x_smooth = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        guidance_features = self.guidance_encoder(x_smooth)

        for _ in range(self.iterations):
            x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
            grad_N = x_pad[:, :, :-2, 1:-1] - x
            grad_S = x_pad[:, :, 2:,  1:-1] - x
            grad_E = x_pad[:, :, 1:-1, 2:] - x
            grad_W = x_pad[:, :, 1:-1, :-2] - x

            combined = torch.cat([grad_N, grad_S, grad_E, grad_W, guidance_features], dim=1)
            c = self.conduction_net(combined)
            c_N, c_S, c_E, c_W = torch.split(c, 1, dim=1)

            x = x + self.lambda_param * (
                c_N * grad_N + c_S * grad_S + c_E * grad_E + c_W * grad_W
            )
        return x


# ==========================================
# 2. LOSSES
# ==========================================
def gradient_loss(pred, target):
    dx_pred = pred[:, :, :, 1:] - pred[:, :, :, :-1]
    dx_tgt  = target[:, :, :, 1:] - target[:, :, :, :-1]
    dy_pred = pred[:, :, 1:, :] - pred[:, :, :-1, :]
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


# ==========================================
# 3. DATASET
# ==========================================
class MRIDenoisingDataset(Dataset):
    def __init__(self, folder_path, image_size=128):
        self.image_paths = []
        self.labels = []

        for label, subfolder in enumerate(['no', 'yes']):
            paths = glob.glob(os.path.join(folder_path, subfolder, '*.jpg')) + \
                    glob.glob(os.path.join(folder_path, subfolder, '*.jpeg'))
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
        sigma = torch.empty(1).uniform_(0.10, 0.20).item()
        noise = torch.randn_like(clean_tensor) * sigma
        noisy_tensor = torch.clamp(clean_tensor + noise, 0., 1.)
        return noisy_tensor, clean_tensor, self.labels[idx]


# ==========================================
# 4. SETUP
# ==========================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Running on: {device}")

dataset_path = '/Users/tushar/Documents/Repositories/neural-anisotropic-diffusion/brain_tumor_dataset'
full_dataset = MRIDenoisingDataset(dataset_path)

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
epochs = 150

scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=50, T_mult=2, eta_min=1e-5
)

os.makedirs('checkpoints', exist_ok=True)


# ==========================================
# 5. TRAINING LOOP
# ==========================================
train_losses, val_losses = [], []
train_psnrs,  val_psnrs  = [], []
best_val_loss = float('inf')

for epoch in range(epochs):
    model.train()
    epoch_train_loss, epoch_train_psnr = 0.0, 0.0
    for noisy, clean, _ in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        optimizer.zero_grad()
        output = model(noisy)
        loss = combined_loss(output, clean, l1_criterion)
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
        epoch_train_psnr += psnr(output.detach(), clean).item()

    model.eval()
    epoch_val_loss, epoch_val_psnr = 0.0, 0.0
    with torch.no_grad():
        for noisy, clean, _ in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            epoch_val_loss += combined_loss(output, clean, l1_criterion).item()
            epoch_val_psnr += psnr(output, clean).item()

    scheduler.step()

    avg_train      = epoch_train_loss / len(train_loader)
    avg_val        = epoch_val_loss   / len(val_loader)
    avg_train_psnr = epoch_train_psnr / len(train_loader)
    avg_val_psnr   = epoch_val_psnr   / len(val_loader)

    train_losses.append(avg_train);     val_losses.append(avg_val)
    train_psnrs.append(avg_train_psnr); val_psnrs.append(avg_val_psnr)

    if avg_val < best_val_loss:
        best_val_loss = avg_val
        torch.save(model.state_dict(), 'checkpoints/best_model.pth')

    if (epoch + 1) % 10 == 0:
        lrs = [pg['lr'] for pg in optimizer.param_groups]
        print(f"Epoch [{epoch+1}/{epochs}] "
              f"Train: {avg_train:.4f} | Val: {avg_val:.4f} | "
              f"PSNR Train: {avg_train_psnr:.2f}dB | Val: {avg_val_psnr:.2f}dB | "
              f"LR: guide={lrs[0]:.2e} cond={lrs[1]:.2e}")

print(f"\nTraining complete. Best val loss: {best_val_loss:.4f}")


# ==========================================
# 6. VISUALIZATION
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
        lv = label
        if len(examples[lv]) < 2:
            out = model(noisy.unsqueeze(0).to(device)).cpu()
            s = ssim(out, clean.unsqueeze(0), data_range=1.0).item()
            p = psnr(out, clean.unsqueeze(0)).item()
            examples[lv].append((clean, noisy, out.squeeze(0), s, p))
        if len(examples[0]) == 2 and len(examples[1]) == 2:
            break

fig, axes = plt.subplots(4, 3, figsize=(12, 16))
row = 0
for label, data_list in examples.items():
    class_name = "Healthy" if label == 0 else "Tumor"
    for clean, noisy, out, s, p in data_list:
        axes[row, 0].imshow(clean.squeeze(), cmap='gray')
        axes[row, 0].set_title(f"{class_name} — Ground Truth"); axes[row, 0].axis('off')
        axes[row, 1].imshow(noisy.squeeze(), cmap='gray')
        axes[row, 1].set_title("Input (Noisy)"); axes[row, 1].axis('off')
        axes[row, 2].imshow(out.squeeze(), cmap='gray')
        axes[row, 2].set_title(f"Denoised | SSIM: {s:.3f} | PSNR: {p:.1f}dB")
        axes[row, 2].axis('off')
        row += 1

plt.tight_layout()
plt.savefig('qualitative_results.png', dpi=150)
plt.show()