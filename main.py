import os
import glob
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ==========================================
# 1. THE MODEL ARCHITECTURE
# ==========================================
class NeuralPeronaMalik(nn.Module):
    def __init__(self, iterations=10, lambda_param=0.1):
        super().__init__()
        self.iterations = iterations
        self.lambda_param = lambda_param
        
        self.conduction_net = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 4, kernel_size=3, padding=1),
            nn.Sigmoid() 
        )

    def forward(self, x):
        for t in range(self.iterations):
            x_pad = F.pad(x, (1, 1, 1, 1), mode='replicate')
            
            grad_N = x_pad[:, :, :-2, 1:-1] - x
            grad_S = x_pad[:, :, 2:, 1:-1] - x
            grad_E = x_pad[:, :, 1:-1, 2:] - x
            grad_W = x_pad[:, :, 1:-1, :-2] - x
            
            grads = torch.cat([grad_N, grad_S, grad_E, grad_W], dim=1)
            
            c = self.conduction_net(grads)
            c_N, c_S, c_E, c_W = torch.split(c, 1, dim=1)
            
            x = x + self.lambda_param * (c_N * grad_N + c_S * grad_S + c_E * grad_E + c_W * grad_W)
            
        return x

# ==========================================
# 2. THE DATA LOADER
# ==========================================
class MRIDenoisingDataset(Dataset):
    def __init__(self, folder_path, image_size=128):
        self.image_paths = glob.glob(os.path.join(folder_path, '*.jpg'), recursive=True) + \
                           glob.glob(os.path.join(folder_path, '*.jpeg'), recursive=True)
        
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        clean_image = Image.open(img_path).convert('RGB')
        clean_tensor = self.transform(clean_image)
        
        noise = torch.randn_like(clean_tensor) * 0.15 
        noisy_tensor = torch.clamp(clean_tensor + noise, 0., 1.)
        
        return noisy_tensor, clean_tensor

# ==========================================
# 3. SETUP & INITIALIZATION
# ==========================================
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Running on device: {device}")

dataset_path = '/Users/tushar/Documents/Repositories/neural-anisotropic-diffusion/brain_tumor_dataset' 

dataset = MRIDenoisingDataset(folder_path=dataset_path)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
print(f"Successfully loaded {len(dataset)} images.")

# TWEAK 1: Lowered lambda_param from 0.1 to 0.05 for tighter control
model = NeuralPeronaMalik(iterations=10, lambda_param=0.1).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# TWEAK 2: Swapped MSE for L1 Loss to punish blurriness 
criterion = nn.L1Loss()

# ==========================================
# 4. TRAINING LOOP
# ==========================================
# TWEAK 3: Increased epochs to 200 so it can fully map the edges
epochs = 300
print("Starting training...")

for epoch in range(epochs):
    epoch_loss = 0.0
    for noisy_imgs, clean_imgs in dataloader:
        noisy_imgs = noisy_imgs.to(device)
        clean_imgs = clean_imgs.to(device)
        
        denoised_imgs = model(noisy_imgs)
        loss = criterion(denoised_imgs, clean_imgs)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        epoch_loss += loss.item()
        
    avg_loss = epoch_loss / len(dataloader)
    
    # Updated to print every 10 epochs instead of 5 to keep the terminal clean
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f}")

print("Training Complete!")

# ==========================================
# 5. VISUALIZE RESULTS & SAVE
# ==========================================
print("Generating visual results...")
model.eval()

noisy_imgs, clean_imgs = next(iter(dataloader))
noisy_imgs = noisy_imgs.to(device)

with torch.no_grad():
    denoised_imgs = model(noisy_imgs)

clean_display = clean_imgs[0].cpu().squeeze().numpy()
noisy_display = noisy_imgs[0].cpu().squeeze().numpy()
denoised_display = denoised_imgs[0].cpu().squeeze().numpy()

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(clean_display, cmap='gray')
axes[0].set_title("Ground Truth (Clean)")
axes[0].axis('off')

axes[1].imshow(noisy_display, cmap='gray')
axes[1].set_title("Input (Synthetic Noise)")
axes[1].axis('off')

axes[2].imshow(denoised_display, cmap='gray')
axes[2].set_title("Neural PDE Output (Sharpened)")
axes[2].axis('off')

plt.tight_layout()
plt.show()

# Save the updated weights!
torch.save(model.state_dict(), 'neural_pde_weights_L1.pth')
print("Model weights saved successfully!")