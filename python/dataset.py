
import os
import json

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

ACTION_FORWARD    = 0
ACTION_TURN_RIGHT = 1
ACTION_TURN_LEFT  = 2
ACTION_STOP       = 3
NUM_ACTIONS       = 4

ACTION_NAMES = ["FORWARD", "TURN_RIGHT", "TURN_LEFT", "STOP"]

PROX_MAX = 255.0


class OfflineNavDataset(Dataset):

    def __init__(self, dataset_dir, img_size=128, augment=True):
        self.img_size = img_size
        self.augment = augment

        frames_path = os.path.join(dataset_dir, "frames.npz")
        print(f"[Dataset] Loading frames from {frames_path}")
        self.frames = np.load(frames_path)["frames"]

        trans_path = os.path.join(dataset_dir, "transitions.npz")
        print(f"[Dataset] Loading transitions from {trans_path}")
        trans = np.load(trans_path)
        self.obs_idx      = trans["obs_idx"]
        self.next_obs_idx = trans["next_obs_idx"]
        self.goal_idx     = trans["goal_idx"]
        self.actions      = trans["actions"]
        self.rewards      = trans["rewards"]
        self.dones        = trans["dones"]
        self.obs_proximity    = trans["obs_proximity"]
        self.goal_proximity   = trans["goal_proximity"]

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
        print(f"[Dataset] Proximity: obs_mean={self.obs_proximity.mean():.1f}, "
              f"goal_mean={self.goal_proximity.mean():.1f}")

    def __len__(self):
        return len(self.obs_idx)

    def _frame_to_tensor(self, idx):
        img = self.frames[idx]
        arr = img.astype(np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)

    def _augment(self, obs, goal, next_obs, action):
        if np.random.random() < 0.5:
            obs = obs.flip(-1)
            goal = goal.flip(-1)
            next_obs = next_obs.flip(-1)
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

        obs_proximity  = min(float(self.obs_proximity[idx])  / PROX_MAX, 1.0)
        goal_proximity = min(float(self.goal_proximity[idx]) / PROX_MAX, 1.0)

        if self.augment:
            obs, goal, next_obs, action = self._augment(
                obs, goal, next_obs, action
            )

        return {
            "obs":        obs,
            "goal":       goal,
            "next_obs":   next_obs,
            "action":     torch.tensor(action, dtype=torch.long),
            "reward":     torch.tensor(reward, dtype=torch.float32),
            "done":       torch.tensor(done, dtype=torch.float32),
            "obs_proximity":  torch.tensor(obs_proximity, dtype=torch.float32),
            "goal_proximity": torch.tensor(goal_proximity, dtype=torch.float32),
        }


def get_dataloader(dataset_dir, batch_size=256, augment=True, num_workers=4):
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
