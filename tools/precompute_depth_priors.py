"""
tools/precompute_depth_priors.py
Precompute relative depth priors for KITTI training images using
Depth Anything V2 Large (ViT-L, 335M params).

Model: https://huggingface.co/depth-anything/Depth-Anything-V2-Large

These depth maps are loaded during training when the C2 (Depth Prior
Warm-Start) enhancement is enabled. A learnable affine transform in
the model aligns the relative priors to metric depth.

Usage:
    python tools/precompute_depth_priors.py \
        --data_dir /path/to/kitti/training \
        --output_dir /path/to/kitti/training/depth_prior \
        --device cuda

Resumable: skips images that already have a .npy file (use --overwrite to redo).
"""

import os
import sys
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

import cv2
import torch


def setup_depth_anything_v2(device='cuda'):
    """
    Download and initialize Depth Anything V2 Large.

    The model code is cloned from the HuggingFace Space repo, and weights
    are fetched via huggingface_hub. Everything is cached automatically.
    """
    try:
        from depth_anything_v2.dpt import DepthAnythingV2
    except ImportError:
        print("[INFO] depth_anything_v2 package not found — installing from HuggingFace...")
        import subprocess
        # Clone the Depth Anything V2 repo for the model definition
        da_repo = os.path.join(os.path.dirname(__file__), '_depth_anything_v2_repo')
        if not os.path.isdir(da_repo):
            subprocess.run([
                'git', 'clone', '--depth', '1',
                'https://huggingface.co/spaces/depth-anything/Depth-Anything-V2',
                da_repo
            ], check=True)
        sys.path.insert(0, da_repo)
        from depth_anything_v2.dpt import DepthAnythingV2

    # Instantiate ViT-Large model
    model = DepthAnythingV2(
        encoder='vitl',
        features=256,
        out_channels=[256, 512, 1024, 1024]
    )

    # Download weights from HuggingFace Hub
    from huggingface_hub import hf_hub_download
    weights_path = hf_hub_download(
        repo_id='depth-anything/Depth-Anything-V2-Large',
        filename='depth_anything_v2_vitl.pth',
        repo_type='model'
    )
    print(f"[INFO] Weights loaded from: {weights_path}")

    state_dict = torch.load(weights_path, map_location='cpu')
    model.load_state_dict(state_dict)
    model = model.to(device).eval()

    print(f"[OK] Depth Anything V2 Large (ViT-L) ready on {device}")
    return model


def main():
    parser = argparse.ArgumentParser(
        description='Precompute depth priors for KITTI using Depth Anything V2 Large'
    )
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to KITTI training directory (containing image_2/)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for .npy files (default: <data_dir>/depth_prior)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on (cuda or cpu)')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing .npy files')
    args = parser.parse_args()

    # ── Validate paths ──
    image_dir = os.path.join(args.data_dir, 'image_2')
    if not os.path.isdir(image_dir):
        print(f"[ERROR] Image directory not found: {image_dir}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(args.data_dir, 'depth_prior')
    os.makedirs(output_dir, exist_ok=True)

    # ── Load model ──
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = setup_depth_anything_v2(device)

    # ── Process images ──
    image_files = sorted(glob(os.path.join(image_dir, '*.png')))
    print(f"Found {len(image_files)} images")
    print(f"Output: {output_dir}")

    skipped, processed, errors = 0, 0, 0

    for img_path in tqdm(image_files, desc='Computing depth priors'):
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, f'{img_id}.npy')

        # Skip if already computed
        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        try:
            # Depth Anything V2 expects BGR uint8 numpy array (cv2 format)
            raw_img = cv2.imread(img_path)
            if raw_img is None:
                print(f"\n[WARN] Could not read: {img_path}")
                errors += 1
                continue

            with torch.no_grad():
                depth = model.infer_image(raw_img)  # (H, W) numpy float

            np.save(out_path, depth.astype(np.float32))
            processed += 1

        except Exception as e:
            print(f"\n[ERROR] {img_id}: {e}")
            errors += 1

    print(f"\nDone!")
    print(f"  Processed: {processed}")
    print(f"  Skipped:   {skipped} (already exist)")
    print(f"  Errors:    {errors}")
    print(f"  Output:    {output_dir}")


if __name__ == '__main__':
    main()
