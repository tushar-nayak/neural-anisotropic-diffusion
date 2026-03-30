import matplotlib.pyplot as plt

print("Generating visual results...")

# 1. Put the model in evaluation mode (freezes the layers)
model.eval()

# 2. Grab one batch of test data
noisy_imgs, clean_imgs = next(iter(dataloader))
noisy_imgs = noisy_imgs.to(device)

# 3. Run inference without calculating gradients to save memory
with torch.no_grad():
    denoised_imgs = model(noisy_imgs)

# 4. Move the first image of the batch back to the CPU to plot it
clean_display = clean_imgs[0].cpu().squeeze().numpy()
noisy_display = noisy_imgs[0].cpu().squeeze().numpy()
denoised_display = denoised_imgs[0].cpu().squeeze().numpy()

# 5. Plot the Before, During, and After
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(clean_display, cmap='gray')
axes[0].set_title("Ground Truth (Clean)")
axes[0].axis('off')

axes[1].imshow(noisy_display, cmap='gray')
axes[1].set_title("Input (Synthetic Noise)")
axes[1].axis('off')

axes[2].imshow(denoised_display, cmap='gray')
axes[2].set_title("Neural PDE Output")
axes[2].axis('off')

plt.tight_layout()
plt.show()