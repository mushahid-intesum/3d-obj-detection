"""
Module A1: Prototype-Based Hypothesis Filtering
Adapted from MonoSAOD's TwoStageBin + PrototypeBank for MonoMH's 64-dim DLA34 features.

Replaces the scalar confidence threshold (0.75) in decode_detections() with a
dual-criteria filter: prototype cosine similarity AND depth reliability.

Usage:
    bank = PrototypeBank(dim=64, K=192)
    # During training: update bank with GT RoI features
    bank.update(gt_features, alpha=0.005)
    # During inference: score hypothesis features
    score = bank.score(hypothesis_feature)
"""

import torch
import torch.nn.functional as F


class TwoStageBin:
    """
    Adapted from MonoSAOD's TwoStageBin.
    Maintains K prototype centroids with EMA updates and a FIFO member buffer.
    """

    def __init__(self, dim, K=192, m=512, alpha=0.05, device='cuda'):
        self.dim = dim
        self.K = K
        self.m = m
        self.alpha = alpha
        self.device = torch.device(device)
        self.eps = 1e-12

        # Prototype centroids (float32 for numerical stability)
        self.M = torch.empty(0, dim, dtype=torch.float32, device=self.device)
        self.count = torch.empty(0, dtype=torch.int32, device=self.device)
        self.members = []  # list of tensors, each [<=m, dim]

    @torch.no_grad()
    def _add_member(self, j, feat):
        """Add a feature vector to prototype j's member buffer (FIFO)."""
        mem = self.members[j]
        if mem.size(0) < self.m:
            self.members[j] = torch.cat([mem, feat[None]], dim=0)
        else:
            self.members[j] = torch.roll(mem, shifts=-1, dims=0)
            self.members[j][-1].copy_(feat)

    @property
    def num_prototypes(self):
        return self.M.size(0)

    @property
    def num_members(self):
        return sum(m.size(0) for m in self.members) if self.num_prototypes > 0 else 0

    def state_dict(self):
        return {
            'M': self.M.cpu(),
            'count': self.count.cpu(),
            'members': [m.cpu() for m in self.members],
        }

    def load_state_dict(self, state):
        self.M = state['M'].to(self.device)
        self.count = state['count'].to(self.device)
        self.members = [m.to(self.device) for m in state['members']]


class PrototypeBank:
    """
    Unified prototype bank for MonoMH hypothesis filtering.
    Adapted from MonoSAOD's PrototypeBank to work with 64-dim DLA34 features
    instead of 256-dim DETR multi-level features.
    """

    def __init__(self, dim=64, K=192, m=512, alpha=0.005,
                 similarity_threshold=0.85, depth_reliability_threshold=0.3,
                 device='cuda'):
        self.dim = dim
        self.device = device
        self.similarity_threshold = similarity_threshold
        self.depth_reliability_threshold = depth_reliability_threshold
        self.bank = TwoStageBin(dim=dim, K=K, m=m, alpha=alpha, device=device)
        # Cosine similarity threshold for creating new prototypes
        self._tau_new = 0.8

    @torch.no_grad()
    def update(self, features, alpha=None):
        """
        Update prototype bank with a batch of L2-normalized feature vectors.

        Args:
            features: (N, dim) tensor of L2-normalized features
            alpha: EMA update rate (overrides bank default if provided)
        """
        if features.numel() == 0:
            return

        bin_obj = self.bank
        if alpha is not None:
            update_alpha = alpha
        else:
            update_alpha = bin_obj.alpha

        for i in range(features.size(0)):
            feat = features[i]  # (dim,)

            # First prototype — just add it
            if bin_obj.M.size(0) == 0:
                bin_obj.M = feat[None].clone()
                bin_obj.count = torch.tensor([1], dtype=torch.int32, device=self.device)
                bin_obj.members = [feat[None].clone()]
                continue

            # Compute cosine similarity to all existing prototypes
            sims = torch.mm(bin_obj.M, feat.unsqueeze(1)).squeeze(1)
            high_sim = torch.where(sims >= self._tau_new)[0]

            if len(high_sim) == 0 and bin_obj.M.size(0) < bin_obj.K:
                # Create new prototype
                bin_obj.M = torch.cat([bin_obj.M, feat[None]], dim=0)
                bin_obj.count = torch.cat(
                    [bin_obj.count, torch.tensor([1], dtype=torch.int32, device=self.device)]
                )
                bin_obj.members.append(feat[None].clone())
            elif len(high_sim) > 0:
                # EMA update matching prototypes
                for idx in high_sim:
                    mu = bin_obj.M[idx]
                    mu = (1 - update_alpha) * mu + update_alpha * feat
                    mu = mu / (mu.norm() + bin_obj.eps)
                    bin_obj.M[idx] = mu
                    bin_obj.count[idx] += 1
                    bin_obj._add_member(idx, feat)
            else:
                # Bank full — update nearest prototype
                best_idx = int(torch.argmax(sims))
                mu = bin_obj.M[best_idx]
                mu = (1 - update_alpha) * mu + update_alpha * feat
                mu = mu / (mu.norm() + bin_obj.eps)
                bin_obj.M[best_idx] = mu
                bin_obj.count[best_idx] += 1
                bin_obj._add_member(best_idx, feat)

    @torch.no_grad()
    def score(self, features):
        """
        Compute prototype similarity scores for a batch of features.

        Args:
            features: (N, dim) tensor of L2-normalized features

        Returns:
            (N,) tensor of max cosine similarity scores
        """
        if features.numel() == 0 or self.bank.M.size(0) == 0:
            return torch.zeros(features.size(0), device=self.device)

        # (N, K) = features @ M.T
        sims = torch.mm(features, self.bank.M.t())
        return sims.max(dim=1).values

    def should_accept_hypothesis(self, proto_score, depth_reliability):
        """
        Dual-criteria hypothesis acceptance check.

        Args:
            proto_score: cosine similarity to nearest prototype
            depth_reliability: exp(-uncertainty), higher = more reliable

        Returns:
            bool — whether this hypothesis should be accepted
        """
        return (proto_score > self.similarity_threshold and
                depth_reliability > self.depth_reliability_threshold)

    def __repr__(self):
        return (f"PrototypeBank(dim={self.dim}, "
                f"prototypes={self.bank.num_prototypes}/{self.bank.K}, "
                f"members={self.bank.num_members})")

    def state_dict(self):
        return self.bank.state_dict()

    def load_state_dict(self, state):
        self.bank.load_state_dict(state)


def extract_roi_window_features(roi_features, regions, backbone_channels=64):
    """
    Extract per-window pooled features from RoI feature maps.
    Used for prototype scoring of individual hypothesis regions.

    Args:
        roi_features: (N, C, 7, 7) — full RoI features (69 channels in MonoMH)
        regions: list of (sh, eh, sw, ew) tuples defining spatial windows
        backbone_channels: number of backbone feature channels (64 for DLA34)

    Returns:
        (N, num_regions, backbone_channels) — L2-normalized per-window features
    """
    # Use only backbone channels (strip coord maps + class one-hot)
    backbone_feat = roi_features[:, :backbone_channels, :, :]  # (N, 64, 7, 7)

    window_features = []
    for (sh, eh, sw, ew) in regions:
        # Global average pool within the window
        window_feat = backbone_feat[:, :, sh:eh, sw:ew].mean(dim=(-2, -1))  # (N, 64)
        window_feat = F.normalize(window_feat, dim=-1)
        window_features.append(window_feat)

    return torch.stack(window_features, dim=1)  # (N, num_regions, 64)
