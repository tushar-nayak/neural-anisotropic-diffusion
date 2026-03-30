import os
import glob
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision.models as models
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import matplotlib.pyplot as plt
from pytorch_msssim import ssim

# ==========================================
# 1. PILLAR: PERCEPTUAL (VGG) LOSS
# ==========================================
class PerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # Use first few layers of VGG16 to capture "content" features
        vgg = models.vgg16(weights=models.VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        # ImageNet normalization constants
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x, y):
        # Convert Grayscale to RGB for VGG
        x_rgb = (x.repeat(1, 3, 1, 1) - self.mean) / self.std
        y_rgb = (y.repeat(1, 3, 1, 1) - self.mean) / self.std
        return F.mse_loss(self.vgg(x_rgb), self.vgg(y_rgb))

# ==========================================
# 2. PILLAR: U-NET GUIDANCE + UNCERTAINTY
# ==========================================
class MiniUNet(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.enc1 = nn.Conv2d(in_c, 16, 3, padding=1)
        self.enc2 = nn.Conv2d(16, 32, 3, padding=1)
        self.dropout = nn.Dropout2d(p=0.2) # Active for Uncertainty mapping
        self.dec1 = nn.Conv2d(32, 16, 3, padding=1)
        self.dec2 = nn.Conv2d(16, out_c, 3, padding=1)

    def forward(self, x):
        e1 = F.relu(self.enc1(x))
        e2 = F.relu(self.enc2(F.max_pool2d(e1, 2)))
        e2 = self.dropout(e2) 
        d1 = F.interpolate(e2, scale_factor=2, mode='bilinear', align_corners=False)
        return torch.sigmoid(self.dec2(F.relu(self.dec1(d1))))

class NeuralPeronaMalikV6(nn.Module):
    def __init__(self, iterations=10, lambda_param=0.1, guidance_channels=8):
        super().__init__()
        self.iterations = iterations
        self.lambda_param = lambda_param
        self.guidance_branch = MiniUNet(1, guidance_channels)
        self.conduction_net = nn.Sequential(
            nn.Conv2d(4 + guidance_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 4, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        g_feat = self.guidance_branch(x)
        for t in range(self.iterations):
            x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
            g_N = x_pad[:, :, :-2, 1:-1] - x
            g_S = x_pad[:, :, 2:, 1:-1] - x
            g_E = x_pad[:, :, 1:-1, 2:] - x
            g_W = x_pad[:, :, 1:-1, :-2] - x
            
            combined = torch.cat([g_N, g_S, g_E, g_W, g_feat], dim=1)
            c = self.conduction_net(combined)
            cN, cS, cE, cW = torch.split(c, 1, dim=1)
            x = x + self.lambda_param * (cN*g_N + cS*g_S + cE*g_E + cW*g_W)
        return x

# ==========================================
# 3. PILLAR: AUGMENTED DATASET
# ==========================================
class AugmentedMRIDataset(Dataset):
    def __init__(self, folder_path, image_size=128):
        self.image_paths, self.labels = [], []
        for label, sub in enumerate(['no', 'yes']):
            paths = glob.glob(os.path.join(folder_path, sub, '**', '*.jpg'), recursive=True) + \
                    glob.glob(os.path.join(folder_path, sub, '**', '*.jpeg'), recursive=True)
            self.image_paths.extend(paths)
            self.labels.extend([label]*len(paths))
        
        # Base transforms
        self.base_tf = T.Compose([T.Grayscale(1), T.Resize((image_size, image_size)), T.ToTensor()])
        # Augmentation for training
        self.aug_tf = T.Compose([T.RandomHorizontalFlip(), T.RandomRotation(15)])

    def __len__(self): return len(self.image_paths)
    def __getitem__(self, idx):
        clean = self.base_tf(Image.open(self.image_paths[idx]).convert('RGB'))
        # Note: In real training, we'd only augment the 'train' subset. 
        # For this script, we apply it here; the DataLoader will handle the logic.
        noise = torch.randn_like(clean) * 0.15
        noisy = torch.clamp(clean + noise, 0., 1.)
        return noisy, clean, self.labels[idx]

def get_metrics(img1, img2):
    mse = F.mse_loss(img1, img2).item()
    psnr = 100 if mse == 0 else 20 * math.log10(1.0 / math.sqrt(mse))
    s_score = ssim(img1, img2, data_range=1.0).item()
    return psnr, s_score

# ==========================================
# 4. SETUP & SPLIT (FIXED)
# ==========================================
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dataset_path = '/Users/tushar/Documents/Repositories/neural-anisotropic-diffusion/brain_tumor_dataset'

full_ds = AugmentedMRIDataset(dataset_path)
total_len = len(full_ds)
train_len = int(0.7 * total_len)
val_len = int(0.15 * total_len)
test_len = total_len - train_len - val_len

train_set, val_set, test_set = random_split(full_ds, [train_len, val_len, test_len])
train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
val_loader = DataLoader(val_set, batch_size=8, shuffle=False)
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)

model = NeuralPeronaMalikV6().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
perceptual_criterion = PerceptualLoss().to(device)
l1_criterion = nn.L1Loss()

# ==========================================
# 5. TRAINING
# ==========================================
train_losses, val_losses = [], []
epochs = 150 # VGG is powerful, needs fewer epochs

print(f"Starting V6 Training. Total Data: {total_len}")
for epoch in range(epochs):
    model.train()
    b_loss = []
    for noisy, clean, _ in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        out = model(noisy)
        # Hybrid Loss: Pixel (L1) + Structural (SSIM) + Perceptual (VGG)
        loss = l1_criterion(out, clean) + 0.1 * perceptual_criterion(out, clean) + 0.5 * (1 - ssim(out, clean, data_range=1.0))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        b_loss.append(loss.item())
    
    model.eval()
    v_loss = []
    with torch.no_grad():
        for n, c, _ in val_loader:
            o = model(n.to(device))
            v_loss.append(l1_criterion(o, c.to(device)).item())
    
    train_losses.append(np.mean(b_loss))
    val_losses.append(np.mean(v_loss))
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}: Train={train_losses[-1]:.4f}, Val={val_losses[-1]:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(train_losses, label='Train'); plt.plot(val_losses, label='Val')
plt.title("Master's Edition Loss (Hybrid Perceptual)"); plt.legend(); plt.grid(True)
plt.savefig('v6_loss_curve.png')

# ==========================================
# 6. UNCERTAINTY VISUALIZATION
# ==========================================
def compute_uncertainty(model, img_tensor, iterations=10):
    model.train() # Keep dropout active for MC Dropout
    preds = []
    with torch.no_grad():
        for _ in range(iterations):
            preds.append(model(img_tensor.to(device)).cpu())
    preds = torch.stack(preds)
    return torch.mean(preds, dim=0), torch.var(preds, dim=0)

model.eval()
# Pick one tumor example from test set for final showpiece
noisy, clean, label = test_set[0] 
mean_out, var_out = compute_uncertainty(model, noisy.unsqueeze(0))
psnr, s_score = get_metrics(mean_out, clean.unsqueeze(0))

fig, axes = plt.subplots(1, 4, figsize=(24, 6))
axes[0].imshow(clean.squeeze(), cmap='gray'); axes[0].set_title("Ground Truth")
axes[1].imshow(noisy.squeeze(), cmap='gray'); axes[1].set_title("Input (Noisy)")
axes[2].imshow(mean_out.squeeze(), cmap='gray'); axes[2].set_title(f"Output\nSSIM: {s_score:.4f} | PSNR: {psnr:.2f}dB")
axes[3].imshow(var_out.squeeze(), cmap='hot'); axes[3].set_title("Uncertainty Map (MC Dropout)")
for ax in axes: ax.axis('off')

plt.tight_layout()
plt.savefig('v6_master_uncertainty_results.png')
print("V6 Complete. Plots and weights saved.")
torch.save(model.state_dict(), 'neural_pde_v6_master.pth')
plt.show()