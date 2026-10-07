"""
models.py — Teacher model architecture: RSRNav-style correlation encoder + IQL heads.

Components:
  - SharedEncoder:       Weight-shared ResNet-9 producing 6x6 feature maps
  - CorrelationModule:   Cross-correlation + pyramid + direction-aware lookup
  - TeacherPolicy:       Full teacher pipeline (encoder → correlation → MLP)
  - QNetwork / VNetwork: IQL critic networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Residual Block ───

class ResBlock(nn.Module):
    """Standard residual block with two 3x3 convolutions."""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + residual)


# ─── Shared Encoder (ResNet-9) ───

class SharedEncoder(nn.Module):
    """
    Weight-shared ResNet-9 encoder for both goal and observation images.

    Input:  (B, 3, 48, 48) RGB image
    Output: (B, 128, 6, 6) feature map

    Architecture:
      Conv(3→32, s=2) + ResBlock → 24x24x32
      Conv(32→64, s=2) + ResBlock → 12x12x64
      Conv(64→128, s=2) + ResBlock → 6x6x128
    """

    def __init__(self, feat_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            ResBlock(32),

            nn.Conv2d(32, 64, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            ResBlock(64),

            nn.Conv2d(64, feat_dim, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(feat_dim), nn.ReLU(inplace=True),
            ResBlock(feat_dim),
        )

    def forward(self, x):
        return self.net(x)  # (B, 128, 6, 6)


# ─── Correlation Module ───

class CorrelationModule(nn.Module):
    """
    RSRNav-style direction-aware correlation.

    Takes two feature maps (goal, obs) and produces a correlation cue vector.

    Steps:
      1. Cross-correlation matrix C ∈ (B, H_g*W_g, H_t*W_t)
      2. Correlation pyramid via avg-pooling goal dimensions
      3. 3x3 direction-aware lookup per pyramid level
      4. Conv fusion → flatten → correlation cue
    """

    def __init__(self, feat_h=6, feat_w=6, feat_dim=128, pyramid_levels=3,
                 lookup_radius=1, cue_dim=256):
        super().__init__()
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.feat_dim = feat_dim
        self.pyramid_levels = pyramid_levels
        self.r = lookup_radius
        self.k = 2 * lookup_radius + 1  # 3 for radius=1

        # Conv fusion after lookup concatenation
        lookup_ch = (pyramid_levels + 1) * (self.k ** 2)  # 4 * 9 = 36
        self.fusion = nn.Sequential(
            nn.Conv2d(lookup_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )

        # Flatten dimension after fusion
        fused_size = 32 * feat_h * feat_w
        self.fc = nn.Sequential(
            nn.Linear(fused_size, cue_dim),
            nn.ReLU(inplace=True),
        )

    def _cross_correlation(self, f_goal, f_obs):
        """Compute normalized cross-correlation matrix."""
        B, C, H, W = f_goal.shape
        N = H * W

        # Flatten spatial dims: (B, C, N)
        g = f_goal.view(B, C, N)
        o = f_obs.view(B, C, N)

        # L2 normalize along channel dim
        g = F.normalize(g, dim=1)
        o = F.normalize(o, dim=1)

        # Correlation: (B, N_g, N_o) via batched matmul
        corr = torch.bmm(g.permute(0, 2, 1), o)  # (B, N_g, N_o)
        return corr

    def _build_pyramid(self, corr):
        """
        Build correlation pyramid by pooling the goal dimensions.

        corr: (B, H_g*W_g, H_o*W_o)
        Returns list of (B, H_s*W_s, H_o*W_o) at decreasing goal resolution.
        """
        B = corr.shape[0]
        N_o = corr.shape[2]
        H, W = self.feat_h, self.feat_w

        # Reshape goal dims to 2D: (B, H_g, W_g, N_o)
        c = corr.view(B, H, W, N_o)

        pyramid = [corr]  # level 0: full resolution

        for s in range(1, self.pyramid_levels + 1):
            # Pool goal dims by factor 2
            # Reshape to (B*N_o, 1, H_s, W_s) for avg_pool2d
            cur_h, cur_w = c.shape[1], c.shape[2]
            c_pool = c.permute(0, 3, 1, 2)  # (B, N_o, H, W)
            c_pool = c_pool.reshape(B * N_o, 1, cur_h, cur_w)

            # Handle odd dimensions
            new_h = max(1, cur_h // 2)
            new_w = max(1, cur_w // 2)
            c_pool = F.adaptive_avg_pool2d(c_pool, (new_h, new_w))

            c_pool = c_pool.reshape(B, N_o, new_h, new_w)
            c = c_pool.permute(0, 2, 3, 1)  # (B, new_h, new_w, N_o)

            # Flatten goal dims
            c_flat = c.reshape(B, new_h * new_w, N_o)
            pyramid.append(c_flat)

        return pyramid

    def _direction_lookup(self, pyramid):
        """
        3x3 direction-aware lookup on each pyramid level.

        For each obs position, crop a 3x3 region centered on the
        corresponding goal position. This encodes directional offset.

        Returns (B, H_o, W_o, (S+1)*k^2) concatenated lookup features.
        """
        B = pyramid[0].shape[0]
        H, W = self.feat_h, self.feat_w
        N_o = H * W
        lookups = []

        for s, corr_s in enumerate(pyramid):
            N_g_s = corr_s.shape[1]
            H_g_s = int(N_g_s ** 0.5)
            W_g_s = H_g_s
            if H_g_s * W_g_s != N_g_s:
                # Non-square; compute dims from original aspect ratio
                scale = N_g_s / (H * W)
                H_g_s = max(1, int(H * (scale ** 0.5)))
                W_g_s = max(1, N_g_s // H_g_s)

            # Reshape: (B, H_g_s, W_g_s, N_o) → (B, N_o, H_g_s, W_g_s)
            corr_2d = corr_s.view(B, H_g_s, W_g_s, N_o).permute(0, 3, 1, 2)

            # Pad for border handling
            padded = F.pad(corr_2d, [self.r] * 4, mode='constant', value=0)

            # For each obs position (j), find corresponding goal position
            # and crop 3x3 around it
            lookup_list = []
            for oy in range(H):
                for ox in range(W):
                    j = oy * W + ox
                    # Corresponding goal position (scaled)
                    gy = int(oy * H_g_s / H)
                    gx = int(ox * W_g_s / W)

                    # Crop 3x3 from padded (accounting for pad offset)
                    patch = padded[:, j,
                                   gy:gy + self.k,
                                   gx:gx + self.k]  # (B, k, k)
                    lookup_list.append(patch.reshape(B, self.k ** 2))

            # Stack: (B, N_o, k^2) → (B, H, W, k^2)
            lookup = torch.stack(lookup_list, dim=1).view(B, H, W, self.k ** 2)
            lookups.append(lookup)

        # Concatenate across scales: (B, H, W, (S+1)*k^2)
        return torch.cat(lookups, dim=-1)

    def forward(self, f_goal, f_obs):
        """
        Args:
            f_goal: (B, C, H, W) goal feature map
            f_obs:  (B, C, H, W) observation feature map

        Returns:
            cue: (B, cue_dim) correlation cue vector
        """
        corr = self._cross_correlation(f_goal, f_obs)
        pyramid = self._build_pyramid(corr)
        lookup = self._direction_lookup(pyramid)  # (B, H, W, ch)

        # Rearrange for conv: (B, ch, H, W)
        x = lookup.permute(0, 3, 1, 2)
        x = self.fusion(x)

        # Flatten and project
        x = x.reshape(x.size(0), -1)
        cue = self.fc(x)
        return cue


# ─── Policy / Q / V Networks ───

class PolicyNetwork(nn.Module):
    """Maps correlation cue → action logits."""

    def __init__(self, cue_dim=256, hidden=128, num_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cue_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 64), nn.ReLU(inplace=True),
            nn.Linear(64, num_actions),
        )

    def forward(self, cue):
        return self.net(cue)

    def get_action(self, cue):
        logits = self.forward(cue)
        return torch.distributions.Categorical(logits=logits)


class QNetwork(nn.Module):
    """Maps (correlation cue, action) → Q-value scalar."""

    def __init__(self, cue_dim=256, hidden=128, num_actions=4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cue_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 64), nn.ReLU(inplace=True),
            nn.Linear(64, num_actions),  # output Q for all actions
        )

    def forward(self, cue, action=None):
        q_all = self.net(cue)  # (B, num_actions)
        if action is not None:
            q = q_all.gather(1, action.unsqueeze(1)).squeeze(1)
            return q
        return q_all


class VNetwork(nn.Module):
    """Maps correlation cue → state value scalar."""

    def __init__(self, cue_dim=256, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cue_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, cue):
        return self.net(cue).squeeze(-1)


# ─── Full Teacher Model ───

class TeacherModel(nn.Module):
    """
    Complete teacher model: Encoder → Correlation → Policy/Q/V heads.

    This wraps all components for convenient forward passes.
    """

    def __init__(self, feat_dim=128, cue_dim=256, num_actions=4):
        super().__init__()
        self.encoder = SharedEncoder(feat_dim=feat_dim)
        self.correlation = CorrelationModule(
            feat_h=6, feat_w=6, feat_dim=feat_dim,
            pyramid_levels=2, cue_dim=cue_dim,
        )
        self.policy = PolicyNetwork(cue_dim, num_actions=num_actions)
        self.q1 = QNetwork(cue_dim, num_actions=num_actions)
        self.q2 = QNetwork(cue_dim, num_actions=num_actions)
        self.v = VNetwork(cue_dim)

    def encode_and_correlate(self, obs, goal):
        """Encode images and compute correlation cue."""
        f_obs = self.encoder(obs)
        f_goal = self.encoder(goal)
        cue = self.correlation(f_goal, f_obs)
        return cue

    def forward(self, obs, goal):
        """Full forward: returns action logits."""
        cue = self.encode_and_correlate(obs, goal)
        return self.policy(cue)

    def get_cue(self, obs, goal):
        """Get correlation cue only (for distillation)."""
        with torch.no_grad():
            return self.encode_and_correlate(obs, goal)
