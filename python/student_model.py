"""
student_model.py — Student model for ESP32-S3 deployment.

~1M parameters. 4-stage hierarchical encoder (MicroDepthAnything architecture)
producing feature maps, simplified cross-correlation → cue, scaled MLP policy.

All sizing controlled by constants below.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ═══════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════

# Encoder — 4-stage hierarchical (mirrors MicroDepthAnything)
ENC_CHANNELS       = [64, 128, 320, 640]   # per-stage widths
ENC_BLOCKS         = [2, 3, 3, 2]          # DSConv blocks per stage
FEAT_DIM           = 640                   # final feature channel dim
FEAT_SIZE          = 3                     # spatial output (3×3)

# Correlation
CUE_DIM            = FEAT_SIZE ** 2 * FEAT_SIZE ** 2 + 2  # 9×9 + 2 = 83

# Policy MLP
POLICY_HIDDEN      = [512, 256, 128]       # hidden layer widths
NUM_ACTIONS        = 4                     # F/B/L/R

# Distillation projector
PROJ_DIM           = 256                   # teacher cue dimension
# ═══════════════════════════════════════════════


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
    4-stage hierarchical encoder (MicroDepthAnything architecture).

    Input:  (B, 3, 48, 48)
    Output: (B, C3, 3, 3)

    Uses depthwise-separable convolutions with configurable stage
    widths and block counts.
    """

    def __init__(self, channels=None, blocks=None):
        super().__init__()
        if channels is None:
            channels = ENC_CHANNELS
        if blocks is None:
            blocks = ENC_BLOCKS
        c0, c1, c2, c3 = channels
        b0, b1, b2, b3 = blocks

        # Stem — 48×48 → 24×24
        self.stem = nn.Sequential(
            nn.Conv2d(3, c0, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
        )

        # Hierarchical stages — each halves spatial resolution
        self.stage1 = self._make_stage(c0, c1, b1)    # 24→12
        self.stage2 = self._make_stage(c1, c2, b2)    # 12→6
        self.stage3 = self._make_stage(c2, c3, b3)    # 6→3

    @staticmethod
    def _make_stage(in_ch, out_ch, n_blocks):
        """Build a stage: first block strides, rest preserve resolution."""
        layers = [DepthwiseSeparable(in_ch, out_ch, stride=2)]
        for _ in range(n_blocks - 1):
            layers.append(DepthwiseSeparable(out_ch, out_ch, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return x


class SimplifiedCorrelation(nn.Module):
    """
    Simplified correlation for MCU: 9x9 cross-correlation + 2 L/R scores.

    No pyramid, no conv fusion — just dot products implementable in plain C.
    Output dim = feat_h² × feat_w² + 2 = 83 (for 3×3 features).
    """

    def __init__(self, feat_dim=None, feat_h=None, feat_w=None):
        super().__init__()
        self.feat_dim = feat_dim or FEAT_DIM
        self.feat_h = feat_h or FEAT_SIZE
        self.feat_w = feat_w or FEAT_SIZE
        self.n_positions = self.feat_h * self.feat_w  # 9

    def forward(self, f_goal, f_obs):
        """
        Args:
            f_goal: (B, C, H, W)
            f_obs:  (B, C, H, W)
        Returns:
            cue: (B, 83)
        """
        B = f_goal.size(0)

        # Flatten spatial: (B, C, N) → (B, N, C)
        g = f_goal.view(B, self.feat_dim, -1).permute(0, 2, 1)
        o = f_obs.view(B, self.feat_dim, -1).permute(0, 2, 1)

        # L2 normalize
        g_norm = F.normalize(g, dim=-1)
        o_norm = F.normalize(o, dim=-1)

        # NxN cross-correlation: (B, N, N)
        cross_corr = torch.bmm(g_norm, o_norm.permute(0, 2, 1))
        cross_flat = cross_corr.reshape(B, self.n_positions ** 2)  # (B, 81)

        # Left/Right similarity
        # Left positions: col 0 → indices 0,3,6 in 3x3 grid
        # Right positions: col 2 → indices 2,5,8
        left_idx = [i * self.feat_w for i in range(self.feat_h)]
        right_idx = [i * self.feat_w + (self.feat_w - 1)
                     for i in range(self.feat_h)]

        g_left = g_norm[:, left_idx, :].mean(dim=1)   # (B, C)
        g_right = g_norm[:, right_idx, :].mean(dim=1)
        o_left = o_norm[:, left_idx, :].mean(dim=1)
        o_right = o_norm[:, right_idx, :].mean(dim=1)

        lr_sim = torch.stack([
            (g_left * o_left).sum(dim=-1),    # left similarity
            (g_right * o_right).sum(dim=-1),  # right similarity
        ], dim=-1)  # (B, 2)

        # Concatenate: N² + 2
        cue = torch.cat([cross_flat, lr_sim], dim=-1)
        return cue


class TinyPolicy(nn.Module):
    """
    Scaled MLP policy network.

    Default: 83 → 512 → 256 → 128 → 4
    Configurable via hidden layer widths.
    """

    def __init__(self, cue_dim=None, hidden=None, num_actions=None):
        super().__init__()
        if cue_dim is None:
            cue_dim = CUE_DIM
        if hidden is None:
            hidden = POLICY_HIDDEN
        if num_actions is None:
            num_actions = NUM_ACTIONS

        layers = []
        prev_dim = cue_dim
        for h in hidden:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(inplace=True)])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, num_actions))
        self.net = nn.Sequential(*layers)

    def forward(self, cue):
        return self.net(cue)


class StudentModel(nn.Module):
    """
    Complete student model for ESP32-S3 deployment.

    TinyEncoder (weight-shared) → SimplifiedCorrelation → TinyPolicy
    ~1M parameters → ~1MB INT8
    """

    def __init__(self, num_actions=None):
        super().__init__()
        if num_actions is None:
            num_actions = NUM_ACTIONS
        self.encoder = TinyEncoder()
        self.correlation = SimplifiedCorrelation()
        self.policy = TinyPolicy(num_actions=num_actions)

        # Projection layer for correlation distillation (discarded after training)
        self.corr_projector = nn.Linear(CUE_DIM, PROJ_DIM)

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
        """Project cue to teacher dim for distillation matching."""
        return self.corr_projector(cue)

    def get_deploy_params(self):
        """Count only deployment parameters (exclude projector)."""
        total = 0
        for name, p in self.named_parameters():
            if "corr_projector" not in name:
                total += p.numel()
        return total


def print_model_summary():
    """Print parameter counts per component."""
    model = StudentModel()

    enc = sum(p.numel() for p in model.encoder.parameters())
    corr = sum(p.numel() for p in model.correlation.parameters())
    pol = sum(p.numel() for p in model.policy.parameters())
    proj = sum(p.numel() for p in model.corr_projector.parameters())
    deploy = model.get_deploy_params()
    total = sum(p.numel() for p in model.parameters())

    print("Student Model Summary")
    print("=" * 55)
    print(f"  TinyEncoder:           {enc:>10,} params")
    print(f"  SimplifiedCorrelation: {corr:>10,} params")
    print(f"  TinyPolicy:            {pol:>10,} params")
    print(f"  corr_projector:        {proj:>10,} params  [distill-only]")
    print("=" * 55)
    print(f"  Total:                 {total:>10,}")
    print(f"  Deploy (no projector): {deploy:>10,}")
    print(f"  Deploy INT8:           ~{deploy // 1024} KB "
          f"({deploy / (1024*1024):.2f} MB)")
    print()
    print("Configuration:")
    print(f"  ENC_CHANNELS  = {ENC_CHANNELS}")
    print(f"  ENC_BLOCKS    = {ENC_BLOCKS}")
    print(f"  POLICY_HIDDEN = {POLICY_HIDDEN}")


if __name__ == "__main__":
    print_model_summary()

    # Quick forward pass test
    model = StudentModel()
    obs = torch.randn(2, 3, 48, 48)
    goal = torch.randn(2, 3, 48, 48)
    logits = model(obs, goal)
    print(f"\nForward pass: obs {obs.shape} → logits {logits.shape}")
    print(f"Logits: {logits}")
