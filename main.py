import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_msssim import ssim

# ==========================================
# 1. THE MODEL ARCHITECTURE
# ==========================================
class NeuralPeronaMalik(nn.Module):
    def __init__(self, iterations=10, lambda_param=0.1, guidance_channels=8):
        super().__init__()
        self.iterations = iterations
        self.lambda_param = lambda_param
        
        self.guidance_encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, guidance_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(guidance_channels),
            nn.Sigmoid() 
        )
        
        self.conduction_net = nn.Sequential(
            nn.Conv2d(4 + guidance_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        guidance_features = self.guidance_encoder(x)
        for t in range(self.iterations):
            x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
            grad_N = x_pad[:, :, :-2, 1:-1] - x
            grad_S = x_pad[:, :, 2:, 1:-1] - x
            grad_E = x_pad[:, :, 1:-1, 2:] - x
            grad_W = x_pad[:, :, 1:-1, :-2] - x
            
            combined = torch.cat([grad_N, grad_S, grad_E, grad_W, guidance_features], dim=1)
            c = self.conduction_net(combined)
            c_N, c_S, c_E, c_W = torch.split(c, 1, dim=1)
            
            x = x + self.lambda_param * (c_N * grad_N + c_S * grad_S + c_E * grad_E + c_W * grad_W)
        return x

# ==========================================
# 2. DATASET & UTILS
# ==========================================
class MRIDenoisingDataset(Dataset):
    def __init__(self, folder_path, image_size=128):
        self.image_paths, self.labels = [], []
        for label, subfolder in enumerate(['no', 'yes']):
            paths = glob.glob(os.path.join(folder_path, subfolder, '**', '*.jpg'), recursive=True) + \
                    glob.glob(os.path.join(folder_path, subfolder, '**', '*.jpeg'), recursive=True)
            self.image_paths.extend(paths)
            self.labels.extend([label] * len(paths))

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        clean = self.transform(Image.open(self.image_paths[idx]).convert('RGB'))
        noise = torch.randn_like(clean) * 0.15 
        noisy = torch.clamp(clean + noise, 0., 1.)
        return noisy, clean, self.labels[idx]

def get_metrics(img1, img2):
    mse = F.mse_loss(img1, img2).item()
    psnr = 100 if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse))
    s_score = ssim(img1, img2, data_range=1.0).item()
    return psnr, s_score

# ==========================================
# 3. SETUP
# ==========================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dataset_path = '/Users/tushar/Documents/Repositories/neural-anisotropic-diffusion/brain_tumor_dataset'

full_dataset = MRIDenoisingDataset(dataset_path)
train_size = int(0.7 * len(full_dataset))
val_size = int(0.15 * len(full_dataset))
test_size = len(full_dataset) - train_size - val_size
train_set, val_set, test_set = random_split(full_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

model = NeuralPeronaMalik(iterations=10, lambda_param=0.1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
l1_criterion = nn.L1Loss()

# ==========================================
# 4. TRAINING
# ==========================================
train_losses, val_losses = [], []
epochs = 200

print(f"Starting training on {device}...")
for epoch in range(epochs):
    model.train()
    batch_l = []
    for noisy, clean, _ in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        output = model(noisy)
        loss = (1 - ssim(output, clean, data_range=1.0)) + 1.0 * l1_criterion(output, clean)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        batch_l.append(loss.item())
    
    model.eval()
    v_batch_l = []
    with torch.no_grad():
        for noisy, clean, _ in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            out = model(noisy)
            v_loss = (1 - ssim(out, clean, data_range=1.0)) + 1.0 * l1_criterion(out, clean)
            v_batch_l.append(v_loss.item())

    train_losses.append(np.mean(batch_l))
    val_losses.append(np.mean(v_batch_l))
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: Train={train_losses[-1]:.4f}, Val={val_losses[-1]:.4f}")

# Save Loss Curve
plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title("Training Dynamics (SSIM + L1)")
plt.legend(); plt.grid(True)
plt.savefig('training_loss_plot.png')
print("Loss plot saved as 'training_loss_plot.png'")

# ==========================================
# 5. FINAL VISUALIZATION & EVALUATION
# ==========================================
model.eval()
examples = {0: [], 1: []}
with torch.no_grad():
    for noisy, clean, label in test_set:
        if len(examples[label]) < 2:
            noisy_in = noisy.unsqueeze(0).to(device)
            out = model(noisy_in).cpu()
            psnr, s_val = get_metrics(out, clean.unsqueeze(0))
            examples[label].append((clean, noisy, out, psnr, s_val))
        if all(len(v) == 2 for v in examples.values()): break

fig, axes = plt.subplots(4, 3, figsize=(15, 20))
row = 0
for label, data in examples.items():
    name = "Healthy (No)" if label == 0 else "Tumor (Yes)"
    for clean, noisy, out, psnr, s_val in data:
        axes[row, 0].imshow(clean.squeeze(), cmap='gray')
        axes[row, 0].set_title(f"{name} Ground Truth")
        
        axes[row, 1].imshow(noisy.squeeze(), cmap='gray')
        axes[row, 1].set_title("Input (Synthetic Noise)")
        
        axes[row, 2].imshow(out.squeeze(), cmap='gray')
        axes[row, 2].set_title(f"Neural PDE Output\nSSIM: {s_val:.4f} | PSNR: {psnr:.2f}dB")
        
        for ax in axes[row]: ax.axis('off')
        row += 1

plt.tight_layout()
plt.savefig('final_denoising_results.png')
print("Final visualization saved as 'final_denoising_results.png'")
plt.show()

torch.save(model.state_dict(), 'neural_pde_final_weights.pth')
print("Weights saved. Project Complete.")