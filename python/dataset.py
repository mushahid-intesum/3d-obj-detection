#!/usr/bin/env python3
"""
dataset.py — PyTorch Dataset for the offline RL image-goal navigation dataset.

Loads the index-based dataset produced by hindsight_relabel.py and serves
(obs, goal, action, reward, next_obs, done) batches for training.

Includes on-the-fly data augmentation (color jitter, horizontal flip).
"""

import os

import numpy as np
import torch
from torch.utils.data import Dataset


class ImageNavDataset(Dataset):
    """
    Goal-conditioned offline RL dataset.

    Frames are stored once in a shared array. Transitions reference
    frames by index, making the dataset memory-efficient.
    """

    def __init__(self, dataset_dir, augment=True):
        """
        Args:
            dataset_dir: Path to the offline dataset (output of hindsight_relabel.py).
            augment:     Whether to apply data augmentation.
        """
        # Load shared frame bank
        frames_data = np.load(os.path.join(dataset_dir, "frames.npz"))
        self.frames = frames_data["frames"]  # (N_frames, 48, 48, 3) uint8

        # Load transitions
        trans_data = np.load(os.path.join(dataset_dir, "transitions.npz"))
        self.obs_idx = trans_data["obs_idx"]            # (N_trans,) int32
        self.next_obs_idx = trans_data["next_obs_idx"]  # (N_trans,) int32
        self.goal_idx = trans_data["goal_idx"]          # (N_trans,) int32
        self.actions = trans_data["actions"]             # (N_trans,) int32
        self.rewards = trans_data["rewards"]             # (N_trans,) float32
        self.dones = trans_data["dones"]                 # (N_trans,) bool

        self.augment = augment
        self.n_transitions = len(self.actions)

        print(f"[Dataset] {self.n_transitions} transitions, "
              f"{len(self.frames)} frames, augment={augment}")

    def __len__(self):
        return self.n_transitions

    def _to_tensor(self, img_uint8):
        """Convert (H, W, 3) uint8 image to (3, H, W) float32 tensor in [0, 1]."""
        return torch.from_numpy(img_uint8).permute(2, 0, 1).float() / 255.0

    def _augment_pair(self, obs, goal):
        """
        Apply consistent augmentation to an obs-goal pair.

        - Random horizontal flip (flip action accordingly — handled in __getitem__)
        - Random brightness/contrast jitter (applied independently)
        """
        flip = False
        if np.random.random() < 0.5:
            obs = np.flip(obs, axis=1).copy()
            goal = np.flip(goal, axis=1).copy()
            flip = True

        # Color jitter: random brightness and contrast
        for img in [obs, goal]:
            if np.random.random() < 0.3:
                # Brightness: shift by ±30
                shift = np.random.randint(-30, 31)
                img[:] = np.clip(img.astype(np.int16) + shift, 0, 255).astype(np.uint8)
            if np.random.random() < 0.3:
                # Contrast: scale by 0.7-1.3
                factor = np.random.uniform(0.7, 1.3)
                mean = img.mean()
                img[:] = np.clip((img.astype(np.float32) - mean) * factor + mean,
                                 0, 255).astype(np.uint8)

        return obs, goal, flip

    def __getitem__(self, idx):
        obs = self.frames[self.obs_idx[idx]].copy()
        next_obs = self.frames[self.next_obs_idx[idx]].copy()
        goal = self.frames[self.goal_idx[idx]].copy()
        action = int(self.actions[idx])
        reward = float(self.rewards[idx])
        done = bool(self.dones[idx])

        if self.augment:
            obs, goal, flip = self._augment_pair(obs, goal)
            next_obs_aug, _, _ = self._augment_pair(next_obs, goal)
            next_obs = next_obs_aug

            # If flipped, swap left/right actions
            if flip:
                if action == 1:     # left → right
                    action = 2
                elif action == 2:   # right → left
                    action = 1

        return {
            "obs": self._to_tensor(obs),            # (3, 48, 48)
            "goal": self._to_tensor(goal),           # (3, 48, 48)
            "next_obs": self._to_tensor(next_obs),   # (3, 48, 48)
            "action": torch.tensor(action, dtype=torch.long),
            "reward": torch.tensor(reward, dtype=torch.float32),
            "done": torch.tensor(done, dtype=torch.float32),
        }


def get_dataloader(dataset_dir, batch_size=256, augment=True, num_workers=4):
    """Create a DataLoader for the offline RL dataset."""
    dataset = ImageNavDataset(dataset_dir, augment=augment)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    return loader


if __name__ == "__main__":
    # ─── Quick test ───
    DATASET_DIR = "./data/offline_dataset"

    ds = ImageNavDataset(DATASET_DIR, augment=False)
    print(f"\nDataset size: {len(ds)} transitions")

    sample = ds[0]
    print(f"\nSample keys: {list(sample.keys())}")
    print(f"  obs shape:      {sample['obs'].shape}")
    print(f"  goal shape:     {sample['goal'].shape}")
    print(f"  next_obs shape: {sample['next_obs'].shape}")
    print(f"  action:         {sample['action'].item()}")
    print(f"  reward:         {sample['reward'].item()}")
    print(f"  done:           {sample['done'].item()}")

    loader = get_dataloader(DATASET_DIR, batch_size=32, augment=True, num_workers=0)
    batch = next(iter(loader))
    print(f"\nBatch shapes:")
    for k, v in batch.items():
        print(f"  {k}: {v.shape}")
