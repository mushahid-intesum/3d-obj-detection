"""
tools/precompute_depth_priors.py
Offline script to precompute relative depth priors for KITTI training images.

Uses Depth Anything V2 (ViT-S) to generate per-image relative depth maps,
saved as .npy files. These are loaded during training by the KITTI dataset
when the C2 (Depth Prior Warm-Start) enhancement is enabled.

Usage:
    python tools/precompute_depth_priors.py \
        --data_dir /path/to/kitti/training \
        --output_dir /path/to/kitti/training/depth_prior \
        --model depth_anything_v2_vits \
        --device cuda

If Depth Anything V2 is not available, falls back to a simple Laplacian-based
edge proxy or ZoeDepth. The exact model is not critical since the depth prior
is only used as a warm-start signal with learnable affine alignment.
"""

import os
import sys
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms


def load_depth_model(model_name, device='cuda'):
    """
    Load a pretrained monocular depth estimation model.
    Tries Depth Anything V2 first, falls back to MiDaS, then to a simple proxy.
    """
    if model_name == 'depth_anything_v2_vits':
        try:
            model = torch.hub.load('LiheYoung/Depth-Anything-V2', 'depth_anything_v2_vits',
                                   pretrained=True, trust_repo=True)
            model = model.to(device).eval()
            print(f"[OK] Loaded Depth Anything V2 ViT-S")
            return model, 'depth_anything_v2'
        except Exception as e:
            print(f"[WARN] Could not load Depth Anything V2: {e}")
            print("[INFO] Falling back to MiDaS...")

    if model_name in ['midas', 'MiDaS'] or model_name == 'depth_anything_v2_vits':
        try:
            model = torch.hub.load('intel-isl/MiDaS', 'MiDaS_small', trust_repo=True)
            model = model.to(device).eval()
            midas_transforms = torch.hub.load('intel-isl/MiDaS', 'transforms', trust_repo=True)
            transform = midas_transforms.small_transform
            print(f"[OK] Loaded MiDaS Small")
            return (model, transform), 'midas'
        except Exception as e:
            print(f"[WARN] Could not load MiDaS: {e}")

    # Final fallback: simple gradient-based depth proxy
    print("[INFO] Using simple gradient-based depth proxy (no pretrained model)")
    return None, 'gradient_proxy'


def predict_depth(img_tensor, model_info, model_type, device='cuda'):
    """
    Run depth prediction on a single image tensor.

    Args:
        img_tensor: (3, H, W) normalized image tensor
        model_info: model object(s)
        model_type: str identifying the model type
        device: torch device

    Returns:
        (H, W) numpy array of relative depth values
    """
    with torch.no_grad():
        if model_type == 'depth_anything_v2':
            model = model_info
            # Depth Anything V2 expects BGR uint8 numpy array (H, W, 3)
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            depth = model.infer_image(img_np)  # returns (H, W) numpy
            return depth.astype(np.float32)

        elif model_type == 'midas':
            model, transform = model_info
            # MiDaS expects its own transform
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = (img_np * 255).astype(np.uint8)
            input_batch = transform(img_np).to(device)
            prediction = model(input_batch)
            prediction = F.interpolate(
                prediction.unsqueeze(1),
                size=img_tensor.shape[1:],
                mode='bilinear',
                align_corners=False
            ).squeeze()
            return prediction.cpu().numpy().astype(np.float32)

        elif model_type == 'gradient_proxy':
            # Simple Laplacian-based edge magnitude as depth proxy
            # Not a real depth map, but provides a structured signal
            gray = img_tensor.mean(dim=0, keepdim=True)  # (1, H, W)
            # Sobel-like gradient magnitude
            kx = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).to(device)
            ky = kx.transpose(2, 3)
            gray_4d = gray.unsqueeze(0).to(device)
            gx = F.conv2d(gray_4d, kx, padding=1)
            gy = F.conv2d(gray_4d, ky, padding=1)
            grad_mag = torch.sqrt(gx**2 + gy**2).squeeze()
            # Invert: smoother regions → closer (heuristic)
            depth = 1.0 / (grad_mag + 0.01)
            depth = depth / depth.max()  # normalize to [0, 1]
            return depth.cpu().numpy().astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Precompute depth priors for KITTI')
    parser.add_argument('--data_dir', type=str, default='/mnt/Stuff/3d-mono-obj-det/kitti/training',
                        help='Path to KITTI training directory (e.g., kitti/training)')
    parser.add_argument('--output_dir', type=str, default='/mnt/Stuff/3d-mono-obj-det/3d-obj-detection/lib/backbones/depth_priors',
                        help='Output directory for .npy files (default: <data_dir>/depth_prior)')
    parser.add_argument('--model', type=str, default='depth_anything_v2_vits',
                        choices=['depth_anything_v2_vits', 'midas', 'gradient_proxy'],
                        help='Depth estimation model to use')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run inference on')
    parser.add_argument('--overwrite', action='store_true',
                        help='Overwrite existing .npy files')
    args = parser.parse_args()

    # Setup paths
    image_dir = os.path.join(args.data_dir, 'image_2')
    if not os.path.isdir(image_dir):
        print(f"[ERROR] Image directory not found: {image_dir}")
        sys.exit(1)

    output_dir = args.output_dir or os.path.join(args.data_dir, 'depth_prior')
    os.makedirs(output_dir, exist_ok=True)

    # Load model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model_info, model_type = load_depth_model(args.model, device)
    print(f"Using depth model: {model_type}")
    print(f"Output directory: {output_dir}")

    # Process all images
    image_files = sorted(glob(os.path.join(image_dir, '*.png')))
    print(f"Found {len(image_files)} images")

    skipped, processed = 0, 0
    for img_path in tqdm(image_files, desc='Computing depth priors'):
        img_id = os.path.splitext(os.path.basename(img_path))[0]
        out_path = os.path.join(output_dir, f'{img_id}.npy')

        if os.path.exists(out_path) and not args.overwrite:
            skipped += 1
            continue

        # Load image
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img).astype(np.float32) / 255.0
        img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)  # (3, H, W)

        # Predict depth
        depth = predict_depth(img_tensor, model_info, model_type, device)

        # Save
        np.save(out_path, depth)
        processed += 1

    print(f"\nDone! Processed: {processed}, Skipped: {skipped}")
    print(f"Depth priors saved to: {output_dir}")


if __name__ == '__main__':
    main()
