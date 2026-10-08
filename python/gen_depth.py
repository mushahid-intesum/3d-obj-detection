#!/usr/bin/env python3
"""
gen_depth.py — Generate MiDaS depth maps for collected images.

Reads images from a collection directory, runs MiDaS Small,
and saves depth maps as .npy files alongside the images.
"""

import os
import glob

import numpy as np
import torch
from PIL import Image

# ═══════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════
DATA_DIR   = "data/room1_start_west"   # Single collection, or parent dir
IMG_SIZE   = 48                        # Output depth map size (NxN)
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
# ═══════════════════════════════════════════════


def load_midas(device):
    """Load MiDaS Small model."""
    print("Loading MiDaS Small...")
    model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True)
    model.to(device).eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
    transform = transforms.small_transform

    print("MiDaS loaded.")
    return model, transform


def process_directory(img_dir, model, transform, device, img_size):
    """Generate depth maps for all images in a directory."""
    depth_dir = os.path.join(os.path.dirname(img_dir), "depth")
    os.makedirs(depth_dir, exist_ok=True)

    image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    if not image_files:
        print(f"  No images found in {img_dir}")
        return 0

    count = 0
    for img_path in image_files:
        basename = os.path.splitext(os.path.basename(img_path))[0]
        depth_path = os.path.join(depth_dir, f"{basename}.npy")

        # Skip if already processed
        if os.path.exists(depth_path):
            count += 1
            continue

        # Load and transform
        img = Image.open(img_path).convert("RGB")
        input_batch = transform(np.array(img)).to(device)

        # Run MiDaS
        with torch.no_grad():
            prediction = model(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=(img_size, img_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze()

        # Normalize depth to [0, 1]
        depth = prediction.cpu().numpy()
        d_min, d_max = depth.min(), depth.max()
        if d_max - d_min > 1e-6:
            depth = (depth - d_min) / (d_max - d_min)
        else:
            depth = np.zeros_like(depth)

        np.save(depth_path, depth.astype(np.float32))
        count += 1

        if count % 24 == 0:
            print(f"  Processed {count}/{len(image_files)} images")

    print(f"  Done: {count} depth maps in {depth_dir}")
    return count


def main():
    device = torch.device(DEVICE)
    print(f"Device: {device}")

    model, transform = load_midas(device)

    # Check if DATA_DIR contains images/ subdirectory (single collection)
    img_dir = os.path.join(DATA_DIR, "images")
    if os.path.isdir(img_dir):
        print(f"\nProcessing: {DATA_DIR}")
        process_directory(img_dir, model, transform, device, IMG_SIZE)
    else:
        # Process all subdirectories
        total = 0
        for subdir in sorted(os.listdir(DATA_DIR)):
            sub_img_dir = os.path.join(DATA_DIR, subdir, "images")
            if os.path.isdir(sub_img_dir):
                print(f"\nProcessing: {subdir}")
                total += process_directory(
                    sub_img_dir, model, transform, device, IMG_SIZE
                )
        print(f"\nTotal: {total} depth maps generated")


if __name__ == "__main__":
    main()
