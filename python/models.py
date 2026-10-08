#!/usr/bin/env python3
"""
models.py — Depth-aware teacher model for ImageNav.

Components:
  - SharedEncoder:       ResNet-9 producing 6×6 feature maps
  - CorrelationModule:   RSRNav-style cross-correlation
  - DepthDecoder:        Auxiliary depth prediction head (training only)
  - TeacherModel:        Full pipeline: encoder → correlation → action + depth
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Residual Block ───

class ResBlock(nn.Module):
    """Standard residual block with two 3×3 convolutions."""

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
    """

    def __init__(self, feat_h=6, feat_w=6, feat_dim=128, pyramid_levels=2,
                 lookup_radius=1, cue_dim=256):
        super().__init__()
        self.feat_h = feat_h
        self.feat_w = feat_w
        self.r = lookup_radius
        self.k = 2 * lookup_radius + 1
        self.pyramid_levels = pyramid_levels

        lookup_ch = (pyramid_levels + 1) * (self.k ** 2)
        self.fusion = nn.Sequential(
            nn.Conv2d(lookup_ch, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )

        fused_size = 32 * feat_h * feat_w
        self.fc = nn.Sequential(
            nn.Linear(fused_size, cue_dim),
            nn.ReLU(inplace=True),
        )

    def _cross_correlation(self, f_goal, f_obs):
        B, C, H, W = f_goal.shape
        N = H * W
        g = F.normalize(f_goal.view(B, C, N), dim=1)
        o = F.normalize(f_obs.view(B, C, N), dim=1)
        return torch.bmm(g.permute(0, 2, 1), o)

    def _build_pyramid(self, corr):
        B, N_o = corr.shape[0], corr.shape[2]
        H, W = self.feat_h, self.feat_w
        c = corr.view(B, H, W, N_o)
        pyramid = [corr]

        for _ in range(self.pyramid_levels):
            cur_h, cur_w = c.shape[1], c.shape[2]
            c_pool = c.permute(0, 3, 1, 2).reshape(B * N_o, 1, cur_h, cur_w)
            new_h, new_w = max(1, cur_h // 2), max(1, cur_w // 2)
            c_pool = F.adaptive_avg_pool2d(c_pool, (new_h, new_w))
            c = c_pool.reshape(B, N_o, new_h, new_w).permute(0, 2, 3, 1)
            pyramid.append(c.reshape(B, new_h * new_w, N_o))

        return pyramid

    def _direction_lookup(self, pyramid):
        B = pyramid[0].shape[0]
        H, W = self.feat_h, self.feat_w
        lookups = []

        for corr_s in pyramid:
            N_g_s = corr_s.shape[1]
            H_g_s = max(1, int(N_g_s ** 0.5))
            W_g_s = max(1, N_g_s // H_g_s)

            corr_2d = corr_s.view(B, H_g_s, W_g_s, H * W).permute(0, 3, 1, 2)
            padded = F.pad(corr_2d, [self.r] * 4, mode="constant", value=0)

            patches = []
            for oy in range(H):
                for ox in range(W):
                    j = oy * W + ox
                    gy = int(oy * H_g_s / H)
                    gx = int(ox * W_g_s / W)
                    patch = padded[:, j, gy:gy + self.k, gx:gx + self.k]
                    patches.append(patch.reshape(B, self.k ** 2))

            lookup = torch.stack(patches, dim=1).view(B, H, W, self.k ** 2)
            lookups.append(lookup)

        return torch.cat(lookups, dim=-1)

    def forward(self, f_goal, f_obs):
        corr = self._cross_correlation(f_goal, f_obs)
        pyramid = self._build_pyramid(corr)
        lookup = self._direction_lookup(pyramid)
        x = lookup.permute(0, 3, 1, 2)
        x = self.fusion(x)
        x = x.reshape(x.size(0), -1)
        return self.fc(x)


# ─── Depth Decoder (auxiliary, training only) ───

class DepthDecoder(nn.Module):
    """
    Lightweight depth prediction head.

    Takes encoder feature maps (B, 128, 6, 6) and upsamples to (B, 1, 48, 48).
    Only used during training — stripped for deployment.
    """

    def __init__(self, feat_dim=128):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(feat_dim, 64, 4, stride=2, padding=1),  # 6→12
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),       # 12→24
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),

            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),        # 24→48
            nn.Sigmoid(),  # depth in [0, 1]
        )

    def forward(self, features):
        return self.decoder(features)  # (B, 1, 48, 48)


# ─── Policy Head ───

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


# ─── Q-Network (IQL) ───

class QNetwork(nn.Module):
    """State-action Q-value estimator for IQL.

    Takes correlation cue and discrete action index,
    returns scalar Q-value.
    """

    def __init__(self, cue_dim=256, hidden=128, num_actions=4):
        super().__init__()
        self.action_embed = nn.Embedding(num_actions, 32)
        self.net = nn.Sequential(
            nn.Linear(cue_dim + 32, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, cue, actions):
        """Args:
            cue:     (B, cue_dim) correlation cue
            actions: (B,) int64 action indices
        Returns:
            (B,) scalar Q-values
        """
        a_emb = self.action_embed(actions)          # (B, 32)
        x = torch.cat([cue, a_emb], dim=-1)         # (B, cue_dim+32)
        return self.net(x).squeeze(-1)               # (B,)


# ─── V-Network (IQL) ───

class VNetwork(nn.Module):
    """State value estimator for IQL.

    Takes correlation cue, returns scalar V-value.
    """

    def __init__(self, cue_dim=256, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cue_dim, hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, 64), nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, cue):
        """Args:
            cue: (B, cue_dim) correlation cue
        Returns:
            (B,) scalar V-values
        """
        return self.net(cue).squeeze(-1)


# ─── Full Teacher Model ───

class TeacherModel(nn.Module):
    """
    Depth-aware teacher model.

    Architecture:
      obs → Encoder → f_obs ─┬→ Correlation(f_goal, f_obs) → cue → ActionHead
      goal → Encoder → f_goal┘                                 ↑
                        f_obs ─→ DepthDecoder → depth_pred     (auxiliary)

    At deployment, DepthDecoder is removed. The encoder has learned
    depth-aware features through the auxiliary loss.
    """

    def __init__(self, feat_dim=128, cue_dim=256, num_actions=4,
                 use_depth_head=True):
        super().__init__()
        self.encoder = SharedEncoder(feat_dim=feat_dim)
        self.correlation = CorrelationModule(
            feat_h=6, feat_w=6, feat_dim=feat_dim,
            pyramid_levels=2, cue_dim=cue_dim,
        )
        self.policy = PolicyNetwork(cue_dim, num_actions=num_actions)

        # IQL value networks
        self.q1 = QNetwork(cue_dim, num_actions=num_actions)
        self.q2 = QNetwork(cue_dim, num_actions=num_actions)
        self.v = VNetwork(cue_dim)

        self.use_depth_head = use_depth_head
        if use_depth_head:
            self.depth_decoder = DepthDecoder(feat_dim=feat_dim)

    def encode_and_correlate(self, obs, goal):
        """Encode obs/goal and compute correlation cue (no policy head).

        Used by IQL trainer which feeds the cue to Q/V/Policy separately.

        Returns:
            cue: (B, cue_dim) correlation cue
        """
        f_obs = self.encoder(obs)
        f_goal = self.encoder(goal)
        return self.correlation(f_goal, f_obs)

    def forward(self, obs, goal, return_depth=False):
        """
        Args:
            obs:  (B, 3, 48, 48) current observation
            goal: (B, 3, 48, 48) goal image
            return_depth: if True, also return depth prediction

        Returns:
            action_logits: (B, num_actions)
            depth_pred:    (B, 1, 48, 48) if return_depth=True
        """
        f_obs = self.encoder(obs)
        f_goal = self.encoder(goal)

        cue = self.correlation(f_goal, f_obs)
        action_logits = self.policy(cue)

        if return_depth and self.use_depth_head:
            depth_pred = self.depth_decoder(f_obs)
            return action_logits, depth_pred

        return action_logits

    def get_action(self, obs, goal):
        """Get action for inference (no depth)."""
        with torch.no_grad():
            logits = self.forward(obs, goal, return_depth=False)
            return logits.argmax(dim=-1)

    def strip_depth_head(self):
        """Remove depth decoder for deployment."""
        if hasattr(self, "depth_decoder"):
            del self.depth_decoder
        self.use_depth_head = False
        print("[Model] Depth head stripped for deployment")

    def count_params(self):
        """Count parameters by component."""
        counts = {}
        counts["encoder"] = sum(p.numel() for p in self.encoder.parameters())
        counts["correlation"] = sum(
            p.numel() for p in self.correlation.parameters()
        )
        counts["policy"] = sum(p.numel() for p in self.policy.parameters())
        if self.use_depth_head:
            counts["depth_decoder"] = sum(
                p.numel() for p in self.depth_decoder.parameters()
            )
        counts["total"] = sum(counts.values())
        return counts
