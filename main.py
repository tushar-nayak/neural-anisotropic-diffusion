import os
import glob
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
# 1. THE ARCHITECTURE (Guidance + BatchNorm)
# ==========================================
class NeuralPeronaMalik(nn.Module):
    def __init__(self, iterations=10, lambda_param=0.1, guidance_channels=8):
        super().__init__()
        self.iterations = iterations
        self.lambda_param = lambda_param
        
        # Guidance Encoder with BatchNorm for stability
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
# 2. DATASET (Train/Val/Test Logic)
# ==========================================
class MRIDenoisingDataset(Dataset):
    def __init__(self, folder_path, image_size=128):
        # Find images and assign labels (0 for 'no', 1 for 'yes')
        self.image_paths = []
        self.labels = []
        
        for label, subfolder in enumerate(['no', 'yes']):
            paths = glob.glob(os.path.join(folder_path, subfolder, '*.jpg')) + \
                    glob.glob(os.path.join(folder_path, subfolder, '*.jpeg'))
            self.image_paths.extend(paths)
            self.labels.extend([label] * len(paths))

        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        clean_tensor = self.transform(Image.open(self.image_paths[idx]).convert('RGB'))
        noise = torch.randn_like(clean_tensor) * 0.15 
        noisy_tensor = torch.clamp(clean_tensor + noise, 0., 1.)
        return noisy_tensor, clean_tensor, self.labels[idx]

# ==========================================
# 3. TRAINING SETUP
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
# 4. TRAINING LOOP WITH METRICS
# ==========================================
train_losses, val_losses = [], []
epochs = 200

for epoch in range(epochs):
    model.train()
    epoch_train_loss = 0
    for noisy, clean, _ in train_loader:
        noisy, clean = noisy.to(device), clean.to(device)
        output = model(noisy)
        loss = (1 - ssim(output, clean, data_range=1.0)) + 1.0 * l1_criterion(output, clean)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_train_loss += loss.item()
    
    model.eval()
    epoch_val_loss = 0
    with torch.no_grad():
        for noisy, clean, _ in val_loader:
            noisy, clean = noisy.to(device), clean.to(device)
            output = model(noisy)
            loss = (1 - ssim(output, clean, data_range=1.0)) + 1.0 * l1_criterion(output, clean)
            epoch_val_loss += loss.item()

    train_losses.append(epoch_train_loss/len(train_loader))
    val_losses.append(epoch_val_loss/len(val_loader))
    
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}: Train={train_losses[-1]:.4f}, Val={val_losses[-1]:.4f}")

# ==========================================
# 5. ANALYSIS & VISUALIZATION
# ==========================================
# A. Loss Plot
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title("Training vs Validation Loss (L1 + SSIM)")
plt.legend()
plt.show()

# B. Class-Specific Visualization
model.eval()
examples = {0: [], 1: []} # 0: No Tumor, 1: Tumor
with torch.no_grad():
    for noisy, clean, label in test_set:
        label_val = label
        if len(examples[label_val]) < 2:
            out = model(noisy.unsqueeze(0).to(device))
            examples[label_val].append((clean, noisy, out.cpu()))
        if len(examples[0]) == 2 and len(examples[1]) == 2: break

fig, axes = plt.subplots(4, 3, figsize=(12, 16))
row = 0
for label, data_list in examples.items():
    class_name = "Healthy (No)" if label == 0 else "Tumor (Yes)"
    for clean, noisy, out in data_list:
        axes[row, 0].imshow(clean.squeeze(), cmap='gray'); axes[row, 0].set_title(f"{class_name} - Clean")
        axes[row, 1].imshow(noisy.squeeze(), cmap='gray'); axes[row, 1].set_title("Input (Noise)")
        axes[row, 2].imshow(out.squeeze(), cmap='gray'); axes[row, 2].set_title("Neural PDE Output")
        row += 1
plt.tight_layout()
plt.show()