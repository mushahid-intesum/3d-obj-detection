"""
student_model.py — Tiny student model for ESP32-S3 deployment.

~10K parameters. Depthwise-separable CNN encoder producing 3x3x32 feature maps,
simplified 9x9 cross-correlation + L/R similarity → 83-dim cue, tiny MLP policy.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparable(nn.Module):
    """Depthwise separable convolution: depthwise + pointwise."""

    def __init__(self, in_ch, out_ch, kernel=3, stride=2):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, stride=stride,
                            padding=kernel // 2, groups=in_ch, bias=False)
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = F.relu(self.bn1(self.dw(x)))
        x = F.relu(self.bn2(self.pw(x)))
        return x


class TinyEncoder(nn.Module):
    """
    Minimal encoder for MCU deployment.

    Input:  (B, 3, 48, 48)
    Output: (B, 32, 3, 3)

    4 layers: 1 standard conv + 3 depthwise-separable.
    """

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            # Standard conv: 48x48x3 → 24x24x8
            nn.Conv2d(3, 8, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
            # DWSep: 24x24x8 → 12x12x16
            DepthwiseSeparable(8, 16),
            # DWSep: 12x12x16 → 6x6x32
            DepthwiseSeparable(16, 32),
            # DWSep: 6x6x32 → 3x3x32
            DepthwiseSeparable(32, 32),
        )

    def forward(self, x):
        return self.net(x)


class SimplifiedCorrelation(nn.Module):
    """
    Simplified correlation for MCU: 9x9 cross-correlation + 2 L/R scores = 83-dim.

    No pyramid, no conv fusion — just dot products implementable in plain C.
    """

    def __init__(self, feat_dim=32, feat_h=3, feat_w=3):
        super().__init__()
        self.feat_dim = feat_dim
        self.n_positions = feat_h * feat_w  # 9

    def forward(self, f_goal, f_obs):
        """
        Args:
            f_goal: (B, 32, 3, 3)
            f_obs:  (B, 32, 3, 3)
        Returns:
            cue: (B, 83)
        """
        B = f_goal.size(0)

        # Flatten spatial: (B, 32, 9) → (B, 9, 32)
        g = f_goal.view(B, self.feat_dim, -1).permute(0, 2, 1)
        o = f_obs.view(B, self.feat_dim, -1).permute(0, 2, 1)

        # L2 normalize
        g_norm = F.normalize(g, dim=-1)
        o_norm = F.normalize(o, dim=-1)

        # 9x9 cross-correlation: (B, 9, 9)
        cross_corr = torch.bmm(g_norm, o_norm.permute(0, 2, 1))
        cross_flat = cross_corr.reshape(B, self.n_positions ** 2)  # (B, 81)

        # Left/Right similarity
        # Left positions: col 0 → indices 0,3,6 in 3x3 grid
        # Right positions: col 2 → indices 2,5,8
        left_idx = [0, 3, 6]
        right_idx = [2, 5, 8]

        g_left = g_norm[:, left_idx, :].mean(dim=1)   # (B, 32)
        g_right = g_norm[:, right_idx, :].mean(dim=1)
        o_left = o_norm[:, left_idx, :].mean(dim=1)
        o_right = o_norm[:, right_idx, :].mean(dim=1)

        lr_sim = torch.stack([
            (g_left * o_left).sum(dim=-1),    # left similarity
            (g_right * o_right).sum(dim=-1),  # right similarity
        ], dim=-1)  # (B, 2)

        # Concatenate: 81 + 2 = 83
        cue = torch.cat([cross_flat, lr_sim], dim=-1)
        return cue


class TinyPolicy(nn.Module):
    """Tiny MLP: 83 → 64 → 32 → 4 action logits."""

    def __init__(self, cue_dim=83, num_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cue_dim, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 32), nn.ReLU(inplace=True),
            nn.Linear(32, num_actions),
        )

    def forward(self, cue):
        return self.net(cue)


class StudentModel(nn.Module):
    """
    Complete student model for ESP32-S3 deployment.

    TinyEncoder (weight-shared) → SimplifiedCorrelation → TinyPolicy
    Total: ~10K parameters → ~10KB INT8
    """

    def __init__(self, num_actions=4):
        super().__init__()
        self.encoder = TinyEncoder()
        self.correlation = SimplifiedCorrelation()
        self.policy = TinyPolicy(cue_dim=83, num_actions=num_actions)

        # Projection layer for correlation distillation (discarded after training)
        self.corr_projector = nn.Linear(83, 256)

    def encode(self, x):
        """Encode a single image to feature map."""
        return self.encoder(x)

    def get_cue(self, obs, goal):
        """Compute correlation cue from raw images."""
        f_obs = self.encoder(obs)
        f_goal = self.encoder(goal)
        return self.correlation(f_goal, f_obs)

    def forward(self, obs, goal):
        """Full forward: returns action logits."""
        cue = self.get_cue(obs, goal)
        return self.policy(cue)

    def project_cue(self, cue):
        """Project 83-dim cue to 256-dim for distillation matching."""
        return self.corr_projector(cue)

    def get_deploy_params(self):
        """Count only deployment parameters (exclude projector)."""
        total = 0
        for name, p in self.named_parameters():
            if "corr_projector" not in name:
                total += p.numel()
        return total


def print_model_summary():
    """Print parameter counts per layer."""
    model = StudentModel()
    print("Student Model Summary")
    print("=" * 50)
    total = 0
    deploy = 0
    for name, p in model.named_parameters():
        n = p.numel()
        total += n
        is_deploy = "corr_projector" not in name
        if is_deploy:
            deploy += n
        marker = "" if is_deploy else " [distill-only]"
        print(f"  {name:40s} {str(list(p.shape)):>20s} = {n:>6,}{marker}")
    print("=" * 50)
    print(f"  Total params:      {total:>10,}")
    print(f"  Deploy params:     {deploy:>10,}")
    print(f"  Deploy size (INT8): ~{deploy // 1024} KB")


if __name__ == "__main__":
    print_model_summary()

    # Quick forward pass test
    model = StudentModel()
    obs = torch.randn(2, 3, 48, 48)
    goal = torch.randn(2, 3, 48, 48)
    logits = model(obs, goal)
    print(f"\nForward pass: obs {obs.shape} → logits {logits.shape}")
    print(f"Logits: {logits}")
