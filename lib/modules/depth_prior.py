"""
Module C2: Depth Prior Warm-Start
Inspired by FF-VIO-Init's feed-forward geometric bootstrapping.

Uses precomputed depth priors (from a frozen depth estimation model) as
auxiliary training targets during early epochs. This stabilizes the depth
head initialization and accelerates convergence.

The prior loss decays linearly to zero over `warmstart_epochs`, after which
the model relies entirely on its own learned depth estimation.

Precomputation: run tools/precompute_depth_priors.py once to generate
    kitti/training/depth_prior/{:06d}.npy files.

Usage in loss:
    aligned_prior = prior_scale * depth_prior + prior_shift
    prior_loss = F.mse_loss(vis_depth, aligned_prior)
    weight = max(0, 1 - epoch / warmstart_epochs) * prior_loss_weight
    total_depth_loss += weight * prior_loss
"""

import os
import numpy as np
import torch
import torch.nn.functional as F


class DepthPriorLoader:
    """
    Loads precomputed depth prior maps from disk and provides
    per-RoI depth targets via cropping and resizing.

    Depth priors are relative (not metric) depth maps produced by a frozen
    depth estimation model. A learnable affine transform (scale, shift) in
    the main model aligns them to metric depth during training.
    """

    def __init__(self, prior_dir, device='cpu'):
        """
        Args:
            prior_dir: path to directory containing {:06d}.npy depth maps
            device: device to load tensors to
        """
        self.prior_dir = prior_dir
        self.device = device
        self.cache = {}

    def load(self, img_id):
        """
        Load a precomputed depth prior map.

        Args:
            img_id: int — KITTI image index

        Returns:
            (H, W) tensor of relative depth values, or None if not available
        """
        if img_id in self.cache:
            return self.cache[img_id]

        path = os.path.join(self.prior_dir, '{:06d}.npy'.format(img_id))
        if not os.path.exists(path):
            return None

        depth_map = np.load(path).astype(np.float32)
        depth_tensor = torch.from_numpy(depth_map).to(self.device)
        self.cache[img_id] = depth_tensor
        return depth_tensor

    def get_roi_depth(self, depth_map, box2d, roi_size=7):
        """
        Crop depth map to 2D bounding box and resize to RoI grid.

        Args:
            depth_map: (H, W) tensor of depth values
            box2d: [x1, y1, x2, y2] in image coordinates
            roi_size: output spatial size (default 7 to match MonoMH RoI)

        Returns:
            (roi_size, roi_size) tensor of depth values
        """
        x1, y1, x2, y2 = int(box2d[0]), int(box2d[1]), int(box2d[2]), int(box2d[3])

        # Clamp to image boundaries
        H, W = depth_map.shape
        x1, x2 = max(0, x1), min(W, x2)
        y1, y2 = max(0, y1), min(H, y2)

        if x2 <= x1 or y2 <= y1:
            return torch.zeros(roi_size, roi_size, device=depth_map.device)

        crop = depth_map[y1:y2, x1:x2]
        # Resize to roi_size x roi_size
        crop = F.interpolate(
            crop.unsqueeze(0).unsqueeze(0),
            size=(roi_size, roi_size),
            mode='bilinear',
            align_corners=False
        )
        return crop.squeeze(0).squeeze(0)

    def clear_cache(self):
        """Clear the in-memory cache."""
        self.cache.clear()


def compute_prior_loss_weight(epoch, warmstart_epochs, initial_weight):
    """
    Compute the decaying weight for the depth prior loss.
    Linear decay from initial_weight to 0 over warmstart_epochs.

    Args:
        epoch: current training epoch
        warmstart_epochs: number of epochs over which to decay
        initial_weight: starting loss weight

    Returns:
        float — current prior loss weight
    """
    if warmstart_epochs <= 0:
        return 0.0
    return max(0.0, (1.0 - epoch / warmstart_epochs)) * initial_weight
