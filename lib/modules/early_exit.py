"""
Module D1: Hypothesis Early Exit
Novel contribution — dynamic computation gating for inference efficiency.

A lightweight MLP on the RoI feature predicts an ambiguity score in [0, 1].
During inference, this score determines how many spatial windows to evaluate:
    - EASY  (amb < easy_threshold):  skip multi-hypothesis entirely (1 depth)
    - MEDIUM (easy_threshold <= amb < hard_threshold): 3 central regions
    - HARD  (amb >= hard_threshold): all 9 regions (original behavior)

Training supervision uses depth variance across the 9 windows as a proxy
for scene ambiguity (no additional GT annotation required).

Usage:
    predictor = AmbiguityPredictor(in_channels=69, hidden_dim=64)
    amb_scores = predictor(roi_features)  # (N,) in [0, 1]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AmbiguityPredictor(nn.Module):
    """
    Predicts per-RoI ambiguity score from pooled RoI features.

    Architecture: AdaptiveAvgPool2d(1) → Linear → ReLU → Linear → Sigmoid
    Parameter count: ~4.5K for default settings (69 * 64 + 64 + 64 * 1 + 1)
    """

    def __init__(self, in_channels=69, hidden_dim=64):
        """
        Args:
            in_channels: number of channels in RoI feature map
                         (64 backbone + 2 coord + 3 class = 69 for MonoMH)
            hidden_dim: hidden layer size
        """
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()
        )

        # Initialize to predict medium ambiguity (0.5) by default
        nn.init.zeros_(self.mlp[-2].weight)
        nn.init.constant_(self.mlp[-2].bias, 0.0)  # sigmoid(0) = 0.5

    def forward(self, roi_features):
        """
        Args:
            roi_features: (N, C, 7, 7) — concatenated RoI features

        Returns:
            (N,) — ambiguity scores in [0, 1]
                   0 = unambiguous (easy), 1 = highly ambiguous (hard)
        """
        x = self.pool(roi_features).flatten(1)  # (N, C)
        return self.mlp(x).squeeze(-1)  # (N,)


def compute_ambiguity_targets(multi_vis_depth, regions):
    """
    Compute ambiguity supervision targets from depth variance across windows.
    This is a self-supervised signal — no additional GT annotation needed.

    Args:
        multi_vis_depth: (N, 7, 7) — per-pixel depth predictions within RoI
        regions: list of (sh, eh, sw, ew) tuples for the 9 spatial windows

    Returns:
        (N,) — normalized ambiguity targets in [0, 1]
    """
    window_depths = []
    for (sh, eh, sw, ew) in regions:
        # Mean depth within each window
        wd = multi_vis_depth[:, sh:eh, sw:ew].mean(dim=(-2, -1))  # (N,)
        window_depths.append(wd)

    # Variance across window depths = measure of spatial depth disagreement
    depth_var = torch.stack(window_depths, dim=-1).var(dim=-1)  # (N,)

    # Normalize to [0, 1] per-batch
    var_min = depth_var.min()
    var_max = depth_var.max()
    if var_max - var_min < 1e-8:
        return torch.zeros_like(depth_var)

    amb_targets = (depth_var - var_min) / (var_max - var_min)
    return amb_targets


def get_active_regions(amb_score, regions, easy_threshold=0.2, hard_threshold=0.7):
    """
    Determine which spatial regions to evaluate based on ambiguity score.

    Args:
        amb_score: float — predicted ambiguity for this detection
        regions: list of 9 (sh, eh, sw, ew) tuples
        easy_threshold: below this → skip all regions (use global depth)
        hard_threshold: above this → use all 9 regions

    Returns:
        list of (sh, eh, sw, ew) tuples to evaluate, or None for EASY exit
    """
    if amb_score < easy_threshold:
        return None  # EASY: skip multi-hypothesis entirely
    elif amb_score < hard_threshold:
        # MEDIUM: use 3 central regions (top-center, center, bottom-center)
        return [regions[1], regions[4], regions[7]]
    else:
        # HARD: use all 9 regions
        return regions
