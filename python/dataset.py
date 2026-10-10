#!/usr/bin/env python3
"""
dataset.py — PyTorch Dataset for offline RL image-goal navigation.

Loads the index-based transition dataset produced by hindsight_relabel.py:
  - frames.npz:      shared image bank (N, 128, 128, 3) uint8
  - transitions.npz: obs_idx, next_obs_idx, goal_idx, actions, rewards, dones

Serves (obs, goal, action, reward, next_obs, done) batches for IQL training.
"""

import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# ─── Action Space ───
ACTION_FORWARD    = 0
ACTION_TURN_RIGHT = 1
ACTION_TURN_LEFT  = 2
ACTION_STOP       = 3
NUM_ACTIONS       = 4

ACTION_NAMES = ["FORWARD", "TURN_RIGHT", "TURN_LEFT", "STOP"]


class OfflineNavDataset(Dataset):
    """
    PyTorch Dataset for offline goal-conditioned RL navigation.

    Loads pre-built transitions from hindsight_relabel.py output.
    Each sample contains (obs, goal, action, reward, next_obs, done).

    Images are stored once in a shared frame bank and referenced by index
    to avoid duplication.
    """

    def __init__(self, dataset_dir, img_size=128, augment=True):
        """
        Args:
            dataset_dir: Path to offline_dataset/ (output of hindsight_relabel.py).
            img_size: Image size (NxN). Frames are assumed to be this size.
            augment: Whether to apply data augmentation.
        """
        self.img_size = img_size
        self.augment = augment

        # Load shared frame bank
        frames_path = os.path.join(dataset_dir, "frames.npz")
        print(f"[Dataset] Loading frames from {frames_path}")
        self.frames = np.load(frames_path)["frames"]  # (N, 128, 128, 3) uint8

        # Load transitions (indices + labels)
        trans_path = os.path.join(dataset_dir, "transitions.npz")
        print(f"[Dataset] Loading transitions from {trans_path}")
        trans = np.load(trans_path)
        self.obs_idx      = trans["obs_idx"]       # (M,) int32
        self.next_obs_idx = trans["next_obs_idx"]  # (M,) int32
        self.goal_idx     = trans["goal_idx"]      # (M,) int32
        self.actions      = trans["actions"]       # (M,) int32
        self.rewards      = trans["rewards"]       # (M,) float32
        self.dones        = trans["dones"]         # (M,) bool

        # Load dataset info
        info_path = os.path.join(dataset_dir, "dataset_info.json")
        if os.path.exists(info_path):
            with open(info_path) as f:
                self.info = json.load(f)
        else:
            self.info = {}

        print(f"[Dataset] {len(self.frames)} frames, "
              f"{len(self.obs_idx)} transitions")
        if "action_distribution" in self.info:
            print(f"[Dataset] Action distribution: {self.info['action_distribution']}")

    def __len__(self):
        return len(self.obs_idx)

    def _frame_to_tensor(self, idx):
        """Load frame by index and convert to (3, H, W) float32 tensor."""
        img = self.frames[idx]  # (128, 128, 3) uint8
        arr = img.astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

    def _augment(self, obs, goal, next_obs, action):
        """Consistent augmentation: horizontal flip swaps left/right."""
        if np.random.random() < 0.5:
            obs = obs.flip(-1)
            goal = goal.flip(-1)
            next_obs = next_obs.flip(-1)
            # Swap TURN_RIGHT ↔ TURN_LEFT
            if action == ACTION_TURN_RIGHT:
                action = ACTION_TURN_LEFT
            elif action == ACTION_TURN_LEFT:
                action = ACTION_TURN_RIGHT
        return obs, goal, next_obs, action

    def __getitem__(self, idx):
        obs      = self._frame_to_tensor(self.obs_idx[idx])
        next_obs = self._frame_to_tensor(self.next_obs_idx[idx])
        goal     = self._frame_to_tensor(self.goal_idx[idx])
        action   = int(self.actions[idx])
        reward   = float(self.rewards[idx])
        done     = float(self.dones[idx])

        if self.augment:
            obs, goal, next_obs, action = self._augment(
                obs, goal, next_obs, action
            )

        return {
            "obs":      obs,                                          # (3, 128, 128)
            "goal":     goal,                                         # (3, 128, 128)
            "next_obs": next_obs,                                     # (3, 128, 128)
            "action":   torch.tensor(action, dtype=torch.long),       # scalar
            "reward":   torch.tensor(reward, dtype=torch.float32),    # scalar
            "done":     torch.tensor(done, dtype=torch.float32),      # scalar
        }


def get_dataloader(dataset_dir, batch_size=256, augment=True, num_workers=4):
    """Create DataLoader for offline RL training.

    Args:
        dataset_dir: Path to offline_dataset/ directory.
    """
    ds = OfflineNavDataset(dataset_dir, augment=augment)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


if __name__ == "__main__":
    # Quick test
    DATASET_DIR = "./data/offline_dataset"

    ds = OfflineNavDataset(DATASET_DIR, augment=False)
    sample = ds[0]
    print(f"\nSample:")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} ({v.dtype})")
        else:
            print(f"  {k}: {v}")

    loader = get_dataloader(DATASET_DIR, batch_size=8)
    batch = next(iter(loader))
    print(f"\nBatch:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape} ({v.dtype})")
