#!/usr/bin/env python3
"""
gen_depth.py — Generate depth maps using Depth Anything V2 Small.

Uses the HuggingFace Transformers model:
    depth-anything/Depth-Anything-V2-Small-hf

Reads images from collection sessions, runs depth estimation,
and saves normalized [0,1] depth maps as .npy files.

Higher depth values = closer objects (inverted from raw output).

Usage:
    python gen_depth.py                    # process all sessions in data/
    python gen_depth.py --data_dir path/   # custom data root
    python gen_depth.py --force            # overwrite existing depth maps
"""

import os
import glob
import argparse
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

# ═══════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════
DEFAULT_DATA_DIR = "data"
DEPTH_MAP_SIZE   = 48           # Output depth map resolution (NxN)
MODEL_ID         = "depth-anything/Depth-Anything-V2-Small-hf"
# ═══════════════════════════════════════════════


def load_depth_anything(device):
    """Load Depth Anything V2 Small from HuggingFace."""
    print(f"Loading {MODEL_ID}...")
    t0 = time.time()

    processor = AutoImageProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForDepthEstimation.from_pretrained(
        MODEL_ID, torch_dtype=torch.float32
    )
    model.to(device).eval()

    dt = time.time() - t0
    n_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"  Loaded in {dt:.1f}s — {n_params:.1f}M parameters")
    print(f"  Device: {device}")

    return model, processor


@torch.no_grad()
def predict_depth(model, processor, image, device, output_size=DEPTH_MAP_SIZE):
    """
    Run Depth Anything V2 on a single PIL image.

    Returns:
        depth_map: (output_size, output_size) float32 numpy array, normalized [0, 1].
                   Higher values = closer objects.
    """
    # Prepare input
    inputs = processor(images=image, return_tensors="pt").to(device)

    # Run model
    outputs = model(**inputs)
    predicted_depth = outputs.predicted_depth  # (1, H, W)

    # Interpolate to target size
    depth = torch.nn.functional.interpolate(
        predicted_depth.unsqueeze(1),
        size=(output_size, output_size),
        mode="bicubic",
        align_corners=False,
    ).squeeze().cpu().numpy()

    # Normalize to [0, 1]
    d_min, d_max = depth.min(), depth.max()
    if d_max - d_min > 1e-6:
        depth = (depth - d_min) / (d_max - d_min)
    else:
        depth = np.zeros_like(depth)

    # Depth Anything outputs disparity (high = close), which is what we want.
    # High values = close objects → obstacle.
    return depth.astype(np.float32)


def process_directory(img_dir, model, processor, device, force=False):
    """Generate depth maps for all images in a directory."""
    depth_dir = os.path.join(os.path.dirname(img_dir), "depth")
    os.makedirs(depth_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not image_files:
        print(f"  No images found in {img_dir}")
        return 0

    processed = 0
    skipped = 0
    t0 = time.time()

    for i, img_path in enumerate(image_files):
        basename = os.path.splitext(os.path.basename(img_path))[0]
        depth_path = os.path.join(depth_dir, f"{basename}.npy")

        # Skip if already processed (unless force)
        if os.path.exists(depth_path) and not force:
            skipped += 1
            continue

        # Load image
        img = Image.open(img_path).convert("RGB")

        # Run depth estimation
        depth = predict_depth(model, processor, img, device)

        # Save
        np.save(depth_path, depth)
        processed += 1

        # Progress
        total_done = processed + skipped
        if total_done % 50 == 0 or total_done == len(image_files):
            elapsed = time.time() - t0
            fps = processed / max(elapsed, 1e-3)
            print(f"  [{total_done}/{len(image_files)}] "
                  f"processed={processed} skipped={skipped} "
                  f"({fps:.1f} img/s)")

    elapsed = time.time() - t0
    print(f"  Done: {processed} new + {skipped} existing = "
          f"{processed + skipped} total ({elapsed:.1f}s)")
    return processed + skipped


def main():
    parser = argparse.ArgumentParser(
        description="Generate depth maps with Depth Anything V2"
    )
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR,
                        help="Root data directory with session subdirs")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing depth maps")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("═" * 55)
    print("  Depth Anything V2 — Depth Map Generator")
    print("═" * 55)

    model, processor = load_depth_anything(device)

    # Check if data_dir has images/ directly (single session)
    img_dir = os.path.join(args.data_dir, "images")
    if os.path.isdir(img_dir):
        print(f"\nProcessing: {args.data_dir}")
        process_directory(img_dir, model, processor, device, args.force)
    else:
        # Multi-session: scan subdirectories
        total = 0
        sessions = sorted(os.listdir(args.data_dir))
        for subdir in sessions:
            sub_img_dir = os.path.join(args.data_dir, subdir, "images")
            if os.path.isdir(sub_img_dir):
                print(f"\nProcessing session: {subdir}")
                total += process_directory(
                    sub_img_dir, model, processor, device, args.force
                )

        print(f"\n{'═' * 55}")
        print(f"  Total: {total} depth maps across {len(sessions)} sessions")
        print(f"{'═' * 55}")


if __name__ == "__main__":
    main()
