import argparse
import functools
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torchvision.transforms as transforms

try:
    import gradio as gr
except ImportError as exc:
    raise SystemExit(
        "Gradio is not installed. Add it to your environment with `pip install gradio`."
    ) from exc

from main import UnifiedNeuralPeronaMalik, get_device, resolve_repo_path


DEFAULT_CHECKPOINT = "checkpoints/best_model.pth"
try:
    RESAMPLING_BICUBIC = Image.Resampling.BICUBIC
except AttributeError:
    RESAMPLING_BICUBIC = Image.BICUBIC


def image_to_tensor(image, image_size):
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )
    return transform(image.convert("L")).unsqueeze(0)


def tensor_to_pil(tensor, original_size=None):
    array = tensor.squeeze().detach().cpu().clamp(0.0, 1.0).numpy()
    pil = Image.fromarray((array * 255).astype(np.uint8), mode="L")
    if original_size is not None:
        pil = pil.resize(original_size, RESAMPLING_BICUBIC)
    return pil


@functools.lru_cache(maxsize=8)
def load_model_cached(
    checkpoint_path,
    device_name,
    neighbor_mode,
    iterations,
    lambda_param,
    use_refinement,
    use_unet_guidance,
    use_multiscale,
    dropout_p,
):
    device = torch.device(device_name)
    model = UnifiedNeuralPeronaMalik(
        iterations=iterations,
        lambda_param=lambda_param,
        neighbor_mode=neighbor_mode,
        use_refinement=use_refinement,
        use_unet_guidance=use_unet_guidance,
        use_multiscale=use_multiscale,
        dropout_p=dropout_p,
    ).to(device)

    state_dict = torch.load(checkpoint_path, map_location=device)
    if any(key.startswith("conduction_net.") for key in state_dict):
        # Older checkpoints used the legacy module names from a previous model layout.
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
        state_dict = remapped

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    return model


def pick_frames(num_frames, max_frames):
    if num_frames <= 1:
        return [0]
    count = min(num_frames, max_frames)
    return list(np.unique(np.linspace(0, num_frames - 1, num=count, dtype=int)))


def make_trace_figure(trace, max_frames=6):
    captures = trace["captures"]
    frame_indices = pick_frames(len(captures), max_frames)

    fig, axes = plt.subplots(len(frame_indices), 2, figsize=(10, 3.0 * len(frame_indices)))
    axes = np.atleast_2d(axes)

    for row, capture_idx in enumerate(frame_indices):
        capture = captures[capture_idx]
        step_idx = capture["step"]
        axes[row, 0].imshow(capture["image"].squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
        title = "Input" if capture_idx == 0 else f"Step {step_idx}"
        if capture["stage"] == "refined":
            title = "Refined Output"
        axes[row, 0].set_title(title)
        axes[row, 0].axis("off")

        if capture["conduction_map"] is None:
            axes[row, 1].text(0.5, 0.5, "No conduction map for refinement", ha="center", va="center")
            axes[row, 1].axis("off")
        else:
            heatmap = capture["conduction_map"].squeeze().cpu().numpy()
            axes[row, 1].imshow(heatmap, cmap="magma", vmin=0, vmax=1)
            axes[row, 1].set_title(f"Mean conduction at step {step_idx}")
            axes[row, 1].axis("off")

    plt.tight_layout()
    return fig


def make_summary_figure(trace):
    iter_captures = [c for c in trace["captures"] if c["conduction_map"] is not None]
    mean_conduction = np.array([float(c["mean_conduction"].mean().item()) for c in iter_captures], dtype=np.float32)
    mean_update = np.array([float(c["mean_update"].mean().item()) for c in iter_captures], dtype=np.float32)
    final_map = iter_captures[-1]["conduction_map"].squeeze().cpu().numpy() if iter_captures else None

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(mean_conduction, color="#2f6db5", marker="o", linewidth=1.5)
    axes[0].set_title("Mean conduction per step")
    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Mean coefficient")
    axes[0].grid(alpha=0.25)

    axes[1].plot(mean_update, color="#b54d2f", marker="o", linewidth=1.5)
    axes[1].set_title("Mean update magnitude")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Abs update")
    axes[1].grid(alpha=0.25)

    if final_map is not None:
        im = axes[2].imshow(final_map, cmap="magma", vmin=0, vmax=1)
        axes[2].set_title("Final conduction heatmap")
        axes[2].axis("off")
        fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    else:
        axes[2].text(0.5, 0.5, "No conduction trace", ha="center", va="center")
        axes[2].axis("off")

    plt.tight_layout()
    return fig


def make_uncertainty_figure(mean_output, std_output, conduction_mean=None, conduction_std=None):
    fig, axes = plt.subplots(1, 3 if conduction_std is not None else 2, figsize=(15, 4))
    axes = np.atleast_1d(axes)

    axes[0].imshow(mean_output.squeeze().cpu().numpy(), cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Mean output")
    axes[0].axis("off")

    im = axes[1].imshow(std_output.squeeze().cpu().numpy(), cmap="magma")
    axes[1].set_title("Output uncertainty")
    axes[1].axis("off")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    if conduction_std is not None:
        im2 = axes[2].imshow(conduction_std.squeeze().cpu().numpy(), cmap="magma", vmin=0)
        axes[2].set_title("Conduction uncertainty")
        axes[2].axis("off")
        fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig


def run_demo(
    image,
    checkpoint_path,
    neighbor_mode,
    iterations,
    lambda_param,
    use_refinement,
    use_unet_guidance,
    use_multiscale,
    dropout_p,
    image_size,
    trace_stride,
    uncertainty_samples,
):
    if image is None:
        raise gr.Error("Upload an MRI slice first.")

    resolved_checkpoint = resolve_repo_path(checkpoint_path)
    if not os.path.exists(resolved_checkpoint):
        raise gr.Error(f"Checkpoint not found: {resolved_checkpoint}")

    device = get_device()
    try:
        model = load_model_cached(
            resolved_checkpoint,
            str(device),
            int(neighbor_mode),
            int(iterations),
            float(lambda_param),
            bool(use_refinement),
            bool(use_unet_guidance),
            bool(use_multiscale),
            float(dropout_p),
        )
    except RuntimeError as exc:
        raise gr.Error(
            "The checkpoint does not match the selected architecture. "
            "Make sure the neighbor mode and guidance/refinement toggles match the trained model."
        ) from exc

    input_image = image.convert("L")
    original_size = input_image.size
    input_tensor = image_to_tensor(input_image, int(image_size)).to(device)

    with torch.no_grad():
        output_tensor, trace = model.forward_with_trace(input_tensor, capture_every=int(trace_stride))
        output_tensor = torch.clamp(output_tensor, 0.0, 1.0)
        uncertainty = model.forward_uncertainty(input_tensor, samples=int(uncertainty_samples), capture_every=int(trace_stride))

    input_preview = tensor_to_pil(input_tensor, original_size)
    output_preview = tensor_to_pil(output_tensor, original_size)
    trace_fig = make_trace_figure(trace, max_frames=6)
    summary_fig = make_summary_figure(trace)
    uncertainty_fig = make_uncertainty_figure(
        uncertainty["mean_output"],
        uncertainty["std_output"],
        uncertainty["conduction_mean"],
        uncertainty["conduction_std"],
    )

    iter_captures = [c for c in trace["captures"] if c["conduction_map"] is not None]
    mean_conduction = np.mean([float(c["mean_conduction"].mean().item()) for c in iter_captures]) if iter_captures else 0.0
    mean_update = np.mean([float(c["mean_update"].mean().item()) for c in iter_captures]) if iter_captures else 0.0
    output_uncertainty = float(uncertainty["std_output"].mean().item())

    stats = (
        f"Device: `{device}`\n\n"
        f"Checkpoint: `{resolved_checkpoint}`\n\n"
        f"Neighbor mode: `{neighbor_mode}` | Iterations: `{iterations}` | Lambda: `{lambda_param}`\n\n"
        f"Multi-scale: `{use_multiscale}` | Dropout: `{dropout_p}` | Uncertainty samples: `{uncertainty_samples}`\n\n"
        f"Mean conduction: `{mean_conduction:.4f}` | Mean update magnitude: `{mean_update:.4f}` | Output uncertainty: `{output_uncertainty:.4f}`"
    )

    return input_preview, output_preview, uncertainty_fig, trace_fig, summary_fig, stats


def build_demo():
    with gr.Blocks(title="Neural Anisotropic Diffusion Demo") as demo:
        gr.Markdown(
            "# Neural Anisotropic Diffusion Demo\n"
            "Upload a brain MRI slice, run the learned PDE denoiser, and inspect the diffusion trace."
        )

        with gr.Row():
            with gr.Column(scale=1):
                image = gr.Image(type="pil", label="Input MRI Slice")
                checkpoint = gr.Textbox(
                    value=DEFAULT_CHECKPOINT,
                    label="Checkpoint Path",
                    placeholder="checkpoints/best_model.pth",
                )
                neighbor_mode = gr.Dropdown([4, 8], value=4, label="Neighbor Mode")
                iterations = gr.Slider(1, 32, value=10, step=1, label="Iterations")
                lambda_param = gr.Slider(0.005, 0.125, value=0.1, step=0.005, label="Lambda")
                use_multiscale = gr.Checkbox(value=False, label="Use Multi-Scale Diffusion")
                dropout_p = gr.Slider(0.0, 0.5, value=0.1, step=0.05, label="Dropout for Uncertainty")
                image_size = gr.Slider(64, 256, value=128, step=16, label="Model Input Size")
                trace_stride = gr.Slider(1, 4, value=1, step=1, label="Trace Capture Stride")
                uncertainty_samples = gr.Slider(2, 16, value=8, step=1, label="Uncertainty Samples")
                use_refinement = gr.Checkbox(value=False, label="Use Residual Refinement")
                use_unet_guidance = gr.Checkbox(value=False, label="Use MiniUNet Guidance")
                run = gr.Button("Run Diffusion")

            with gr.Column(scale=1):
                input_output = gr.Image(type="pil", label="Input Preview")
                output = gr.Image(type="pil", label="Denoised Output")
                stats = gr.Markdown()

        with gr.Row():
            uncertainty_plot = gr.Plot(label="Uncertainty Map")
            trace_plot = gr.Plot(label="Diffusion Trace")
            summary_plot = gr.Plot(label="Conduction Summary")

        run.click(
            fn=run_demo,
            inputs=[
                image,
                checkpoint,
                neighbor_mode,
                iterations,
                lambda_param,
                use_refinement,
                use_unet_guidance,
                use_multiscale,
                dropout_p,
                image_size,
                trace_stride,
                uncertainty_samples,
            ],
            outputs=[input_output, output, uncertainty_plot, trace_plot, summary_plot, stats],
        )

    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="Gradio demo for Neural Anisotropic Diffusion")
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    demo = build_demo()
    demo.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
