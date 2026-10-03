"""
Module B2: Intrinsic-Conditioned Hypothesis Diversity
Inspired by MonoIA's focal-length-aware feature modulation.

Simplified from MonoIA's CLIP-based focal embedding to a lightweight MLP
that maps scalar focal length to a threshold modulation factor.
Shorter focal length → wider FoV → more depth ambiguity → lower τ → more hypotheses.

Usage:
    conditioner = IntrinsicConditioner(hidden_dim=32, modulation_range=0.3)
    per_image_tau = conditioner(f_y, base_threshold=0.75)  # (B,)
"""

import torch
import torch.nn as nn


class IntrinsicConditioner(nn.Module):
    """
    Maps camera focal length to a per-image confidence threshold.

    Unlike MonoIA's full CLIP+Connector pipeline, this is a lightweight MLP
    suitable for single-dataset (KITTI) training where focal length variation
    is small but still meaningful for hypothesis diversity.

    The output modulates the base confidence threshold:
        τ_adjusted = base_threshold - modulation_range * ambiguity_scale
    where ambiguity_scale ∈ (0, 1) is learned.
    """

    def __init__(self, hidden_dim=32, modulation_range=0.3):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # output ∈ (0, 1)
        )
        self.modulation_range = modulation_range

        # Initialize to near-zero modulation (preserve baseline behavior)
        nn.init.zeros_(self.mlp[-2].weight)
        nn.init.constant_(self.mlp[-2].bias, -2.0)  # sigmoid(-2) ≈ 0.12

    def forward(self, focal_length, base_threshold=0.75):
        """
        Compute per-image adjusted confidence threshold.

        Args:
            focal_length: (B,) — raw f_y values from calibration P2 matrix
            base_threshold: float — MonoMH default threshold (0.75)

        Returns:
            (B,) — per-image adjusted thresholds
        """
        # Normalize focal length to approximately [0, 1] using KITTI range
        # KITTI typical range: f_y ≈ 707-721, but can be wider for other datasets
        f_norm = (focal_length - 700.0) / 600.0
        f_norm = f_norm.clamp(0.0, 1.0).unsqueeze(-1)  # (B, 1)

        ambiguity_scale = self.mlp(f_norm).squeeze(-1)  # (B,)

        # Higher ambiguity → lower threshold → more hypotheses retained
        tau = base_threshold - self.modulation_range * ambiguity_scale

        return tau
