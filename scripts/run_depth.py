"""
run_depth.py

Generates a relative Digital Surface Model (rDSM) from a single-view
RGB image using Depth Anything V2, then removes the global illumination/
gradient bias that monocular depth models impose on nadir (top-down)
imagery, leaving mostly local structure (buildings, terrain relief).

Usage:
    python run_depth.py data/processed/rgb.png data/processed/site1

    This produces, inside data/processed/site1/:
        rdsm_raw.npy         - raw float32 relative depth (model output)
        rdsm_detrended.npy   - float32, gradient bias removed (USE THIS)
        depth_raw.png        - visualization of raw depth
        depth_detrended.png  - visualization of detrended depth
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from scipy.ndimage import gaussian_filter
from transformers import AutoImageProcessor, AutoModelForDepthEstimation


MODEL_NAME = "depth-anything/Depth-Anything-V2-Large-hf"

# How strong the low-pass blur is when estimating the "gradient bias" to
# remove. Larger = only removes very broad, slow trends. Smaller = also
# removes some real local structure. Tune this per image size.
DETREND_SIGMA_FRACTION = 0.15  # fraction of image width


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_model(model_name=MODEL_NAME, device=None):
    device = device or get_device()
    print("Using device:", device)
    print("Loading model:", model_name)

    processor = AutoImageProcessor.from_pretrained(model_name)
    model = AutoModelForDepthEstimation.from_pretrained(model_name)
    model.to(device)
    model.eval()

    print("Model loaded.")
    return processor, model, device


def estimate_relative_depth(image_path, processor, model, device):
    """Run monocular depth estimation. Returns float32 (H, W) array."""

    image = Image.open(image_path).convert("RGB")
    print("Image size:", image.size)

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        predicted_depth = outputs.predicted_depth

    prediction = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=image.size[::-1],  # (H, W)
        mode="bicubic",
        align_corners=False,
    ).squeeze()

    depth = prediction.cpu().numpy().astype(np.float32)

    print("Depth shape:", depth.shape)
    print("Depth min/max:", depth.min(), depth.max())

    return depth


def detrend_depth(depth, sigma_fraction=DETREND_SIGMA_FRACTION):
    """
    Remove the broad, slow gradient bias that monocular depth models impose
    on nadir imagery, leaving mostly local relief/structure.

    Works by estimating the low-frequency trend (heavy Gaussian blur) and
    subtracting it from the original depth map.
    """

    sigma = depth.shape[1] * sigma_fraction
    trend = gaussian_filter(depth, sigma=sigma)
    detrended = depth - trend

    print(f"Detrend sigma: {sigma:.1f}px")
    print("Detrended min/max:", detrended.min(), detrended.max())

    return detrended


def normalize_for_display(arr):
    arr_min, arr_max = arr.min(), arr.max()
    if arr_max - arr_min < 1e-8:
        return np.zeros_like(arr, dtype=np.uint8)
    norm = (arr - arr_min) / (arr_max - arr_min)
    return (norm * 255).astype(np.uint8)


def save_visualization(arr, title, out_path):
    plt.figure(figsize=(10, 8))
    plt.imshow(arr, cmap="inferno")
    plt.colorbar(label="Relative depth")
    plt.title(title)
    plt.axis("off")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", out_path)


def process_image(input_path, output_dir, processor, model, device):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    depth_raw = estimate_relative_depth(input_path, processor, model, device)
    depth_detrended = detrend_depth(depth_raw)

    # Save raw arrays -- these ARE your rDSM outputs
    np.save(output_dir / "rdsm_raw.npy", depth_raw)
    np.save(output_dir / "rdsm_detrended.npy", depth_detrended)

    # Save 16-bit PNGs (more precision than 8-bit, easy to inspect)
    Image.fromarray(
        (normalize_for_display(depth_raw).astype(np.uint16)) * 257
    ).save(output_dir / "rdsm_raw_16bit.png")

    Image.fromarray(
        (normalize_for_display(depth_detrended).astype(np.uint16)) * 257
    ).save(output_dir / "rdsm_detrended_16bit.png")

    # Save colorized visualizations for eyeballing
    save_visualization(
        depth_raw, "Relative Depth (raw)", output_dir / "depth_raw.png"
    )
    save_visualization(
        depth_detrended,
        "Relative Depth (gradient bias removed)",
        output_dir / "depth_detrended.png",
    )

    print(f"\nDone. Outputs in: {output_dir}\n")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python run_depth.py <input_image> <output_dir>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_dir = sys.argv[2]

    processor, model, device = load_model()
    process_image(input_path, output_dir, processor, model, device)