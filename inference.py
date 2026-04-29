import argparse
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Import the model architecture from main.py
from main import UnifiedNeuralPeronaMalik, MiniUNet, SimpleGuidanceEncoder, get_device


def _remap_legacy_state_dict(state_dict):
    if not any(key.startswith("conduction_net.") for key in state_dict):
        return state_dict

    remapped = {}
    for key, value in state_dict.items():
        new_key = key
        if key.startswith("conduction_net."):
            new_key = key.replace("conduction_net.0", "conduction_conv1")
            new_key = new_key.replace("conduction_net.2", "conduction_conv2")
            new_key = new_key.replace("conduction_net.4", "conduction_conv3")
        elif key.startswith("guidance_encoder."):
            new_key = key.replace("guidance_encoder.0", "guidance_encoder.net.0")
            new_key = new_key.replace("guidance_encoder.1", "guidance_encoder.net.1")
            new_key = new_key.replace("guidance_encoder.3", "guidance_encoder.net.3")
        remapped[new_key] = value
    return remapped

def load_model(checkpoint_path, device, neighbor_mode=8, use_refinement=True, use_unet_guidance=True):
    # Default iterations/lambda based on neighbor mode as seen in main.py
    iterations = 16 if neighbor_mode == 8 else 10
    lambda_param = 0.05 if neighbor_mode == 8 else 0.1
    
    model = UnifiedNeuralPeronaMalik(
        iterations=iterations,
        lambda_param=lambda_param,
        neighbor_mode=neighbor_mode,
        use_refinement=use_refinement,
        use_unet_guidance=use_unet_guidance
    ).to(device)
    
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found at: {checkpoint_path}")
        
    state_dict = torch.load(checkpoint_path, map_location=device)
    state_dict = _remap_legacy_state_dict(state_dict)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model

def run_inference(model, image_path, device, image_size=128):
    # Load and preprocess image
    img = Image.open(image_path).convert("L")
    original_size = img.size
    
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)
        output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
    
    # Convert back to numpy for visualization
    input_np = input_tensor.squeeze().cpu().numpy()
    output_np = output_tensor.squeeze().cpu().numpy()
    
    return input_np, output_np, original_size

def main():
    parser = argparse.ArgumentParser(description="Inference script for Neural Anisotropic Diffusion")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth", help="Path to the model checkpoint")
    parser.add_argument("--output", type=str, default="inference_result.png", help="Path to save the result visualization")
    parser.add_argument("--neighbor-mode", type=int, default=4, choices=[4, 8], help="Neighbor mode used during training")
    parser.add_argument("--no-refinement", action="store_true", help="Disable refinement stage")
    parser.add_argument("--no-unet-guidance", action="store_true", help="Disable MiniUNet guidance")
    parser.add_argument("--image-size", type=int, default=128, help="Internal resolution for the model")
    
    args = parser.parse_args()
    device = get_device()
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(
        args.checkpoint, 
        device, 
        neighbor_mode=args.neighbor_mode,
        use_refinement=not args.no_refinement,
        use_unet_guidance=not args.no_unet_guidance
    )
    
    print(f"Running inference on {args.image}...")
    noisy, denoised, orig_size = run_inference(model, args.image, device, image_size=args.image_size)
    
    # Save comparison plot
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(noisy, cmap="gray")
    axes[0].set_title("Input (Noisy)")
    axes[0].axis("off")
    
    axes[1].imshow(denoised, cmap="gray")
    axes[1].set_title("Neural PDE Denoised")
    axes[1].axis("off")
    
    plt.tight_layout()
    plt.savefig(args.output, dpi=150)
    print(f"Result saved to {args.output}")

if __name__ == "__main__":
    main()
