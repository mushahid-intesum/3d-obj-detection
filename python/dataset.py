#!/usr/bin/env python3
"""
dataset.py — Navigation graph + PyTorch Dataset for depth-aware ImageNav.

Builds a topological graph from collected data:
  - Nodes: (position, direction) with image + heading
  - Edges: FORWARD, TURN_RIGHT, TURN_LEFT actions
  - BFS shortest path provides supervision signal

Serves (obs, goal, action, depth) batches for training.
"""

import os
import json
from collections import deque

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

# ─── Action Space ───
ACTION_FORWARD    = 0
ACTION_TURN_RIGHT = 1
ACTION_TURN_LEFT  = 2
ACTION_STOP       = 3
NUM_ACTIONS       = 4

ACTION_NAMES = ["FORWARD", "TURN_RIGHT", "TURN_LEFT", "STOP"]
DIR_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
NUM_DIRS = 8


class NavGraph:
    """
    Topological navigation graph from collected data.

    Nodes: (position_id, direction_index) → image path + heading
    Edges: actions connecting nodes

    Graph structure:
      - TURN_RIGHT: (pos, d) → (pos, (d+1)%8)
      - TURN_LEFT:  (pos, d) → (pos, (d-1)%8)
      - FORWARD:    (pos, 0) → (pos+1, 0)  [only from forward dir]
      - BACKWARD:   (pos, 4) → (pos-1, 4)  [only from reverse dir]
    """

    def __init__(self, collection_dir, photos_per_dir=3):
        self.collection_dir = collection_dir
        self.img_dir = os.path.join(collection_dir, "images")
        self.depth_dir = os.path.join(collection_dir, "depth")

        # Load metadata
        meta_path = os.path.join(collection_dir, "metadata.json")
        with open(meta_path) as f:
            self.metadata = json.load(f)

        # Build node index: (pos, dir) → {image_paths, heading}
        self.nodes = {}
        self.num_positions = 0

        for frame in self.metadata["frames"]:
            pos = frame["position"]
            d = frame["dir_index"]
            key = (pos, d)

            if key not in self.nodes:
                self.nodes[key] = {
                    "images": [],
                    "depths": [],
                    "headings": [],
                    "position": pos,
                    "dir_index": d,
                    "dir_name": frame["dir_name"],
                }
            self.nodes[key]["images"].append(
                os.path.join(self.img_dir, frame["filename"])
            )
            depth_file = os.path.splitext(frame["filename"])[0] + ".npy"
            self.nodes[key]["depths"].append(
                os.path.join(self.depth_dir, depth_file)
            )
            self.nodes[key]["headings"].append(frame["heading_deg"])
            self.num_positions = max(self.num_positions, pos + 1)

        # Build adjacency list
        self.adj = {}  # node_key → [(neighbor_key, action)]
        for (pos, d) in self.nodes:
            self.adj[(pos, d)] = []

            # Turn edges (always available)
            right = (pos, (d + 1) % NUM_DIRS)
            left = (pos, (d - 1) % NUM_DIRS)
            if right in self.nodes:
                self.adj[(pos, d)].append((right, ACTION_TURN_RIGHT))
            if left in self.nodes:
                self.adj[(pos, d)].append((left, ACTION_TURN_LEFT))

            # Forward edge (only from dir 0 → next position dir 0)
            if d == 0 and (pos + 1, 0) in self.nodes:
                self.adj[(pos, d)].append(((pos + 1, 0), ACTION_FORWARD))

            # Backward edge (only from dir 4 → prev position dir 4)
            if d == 4 and (pos - 1, 4) in self.nodes:
                self.adj[(pos, d)].append(((pos - 1, 4), ACTION_FORWARD))

        print(f"[NavGraph] {len(self.nodes)} nodes, "
              f"{self.num_positions} positions, "
              f"{sum(len(v) for v in self.adj.values())} edges")

    def shortest_path_action(self, start, goal):
        """
        BFS shortest path from start to goal.

        Returns:
            The first action on the optimal path, or ACTION_STOP if
            start == goal or no path exists.
        """
        if start == goal:
            return ACTION_STOP

        visited = {start}
        queue = deque()

        # Enqueue neighbors with the action taken from start
        for neighbor, action in self.adj.get(start, []):
            if neighbor not in visited:
                queue.append((neighbor, action))
                visited.add(neighbor)

        while queue:
            current, first_action = queue.popleft()

            if current == goal:
                return first_action

            for neighbor, _ in self.adj.get(current, []):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, first_action))

        # No path found — shouldn't happen in a connected graph
        return ACTION_STOP

    def get_all_node_keys(self):
        """Return list of all node keys."""
        return list(self.nodes.keys())

    def sample_image(self, node_key, rng=None):
        """Sample a random image path for a node (from the 3 photos)."""
        images = self.nodes[node_key]["images"]
        idx = (rng or np.random).randint(0, len(images))
        return images[idx]

    def sample_depth(self, node_key, rng=None):
        """Sample a random depth map path for a node."""
        depths = self.nodes[node_key]["depths"]
        idx = (rng or np.random).randint(0, len(depths))
        return depths[idx]


def discover_collections(data_root):
    """
    Auto-discover all collection directories under a root.

    Scans for subdirectories containing metadata.json.
    Works with structure:
        data/
          room1_start_west/
            metadata.json
            images/
            depth/
          room1_start_east/
            metadata.json
            ...

    Returns list of absolute paths to collection directories.
    """
    collections = []

    # Check if data_root itself is a collection
    if os.path.isfile(os.path.join(data_root, "metadata.json")):
        collections.append(data_root)
        return collections

    # Scan subdirectories
    for entry in sorted(os.listdir(data_root)):
        subdir = os.path.join(data_root, entry)
        if os.path.isdir(subdir) and os.path.isfile(
            os.path.join(subdir, "metadata.json")
        ):
            collections.append(subdir)

    if not collections:
        raise FileNotFoundError(
            f"No collections found under {data_root}. "
            f"Expected subdirectories with metadata.json."
        )

    print(f"[Discover] Found {len(collections)} collections in {data_root}:")
    for c in collections:
        print(f"  - {os.path.basename(c)}")

    return collections


class ImageNavDataset(Dataset):
    """
    PyTorch Dataset for depth-aware image-goal navigation.

    Samples random (start, goal) pairs from the navigation graph,
    computes the optimal first action via BFS, and returns
    (obs, goal, action, depth) for training.
    """

    def __init__(self, data_root, samples_per_epoch=10000,
                 img_size=48, augment=True):
        """
        Args:
            data_root: Root data directory (auto-discovers all collections).
            samples_per_epoch: Number of (start, goal) pairs per epoch.
            img_size: Image size (NxN).
            augment: Whether to apply data augmentation.
        """
        collection_dirs = discover_collections(data_root)

        self.graphs = []
        self.all_nodes = []  # (graph_idx, node_key) tuples

        for cdir in collection_dirs:
            graph = NavGraph(cdir)
            graph_idx = len(self.graphs)
            self.graphs.append(graph)
            for key in graph.get_all_node_keys():
                self.all_nodes.append((graph_idx, key))

        self.samples_per_epoch = samples_per_epoch
        self.img_size = img_size
        self.augment = augment
        self.rng = np.random.RandomState(42)

        print(f"[Dataset] {len(self.all_nodes)} total nodes from "
              f"{len(self.graphs)} collections, "
              f"{samples_per_epoch} samples/epoch")

    def __len__(self):
        return self.samples_per_epoch

    def _load_image(self, path):
        """Load and normalize image to (3, H, W) float32 tensor."""
        img = Image.open(path).convert("RGB")
        if img.size != (self.img_size, self.img_size):
            img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        return torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)

    def _load_depth(self, path):
        """Load depth map as (1, H, W) float32 tensor."""
        if os.path.exists(path):
            depth = np.load(path).astype(np.float32)
            return torch.from_numpy(depth).unsqueeze(0)  # (1, H, W)
        else:
            return torch.zeros(1, self.img_size, self.img_size)

    def _augment(self, obs, goal, depth, action):
        """Consistent augmentation: horizontal flip swaps left/right."""
        if self.rng.random() < 0.5:
            obs = obs.flip(-1)      # flip width
            goal = goal.flip(-1)
            depth = depth.flip(-1)
            # Swap TURN_RIGHT ↔ TURN_LEFT
            if action == ACTION_TURN_RIGHT:
                action = ACTION_TURN_LEFT
            elif action == ACTION_TURN_LEFT:
                action = ACTION_TURN_RIGHT
        return obs, goal, depth, action

    def __getitem__(self, idx):
        # Sample random start and goal from same graph
        gi = self.rng.randint(0, len(self.graphs))
        graph = self.graphs[gi]
        keys = graph.get_all_node_keys()

        start_key = keys[self.rng.randint(0, len(keys))]
        goal_key = keys[self.rng.randint(0, len(keys))]

        # Get optimal action
        action = graph.shortest_path_action(start_key, goal_key)

        # Load images
        obs = self._load_image(graph.sample_image(start_key, self.rng))
        goal_img = self._load_image(graph.sample_image(goal_key, self.rng))
        depth = self._load_depth(graph.sample_depth(start_key, self.rng))

        # Augment
        if self.augment:
            obs, goal_img, depth, action = self._augment(
                obs, goal_img, depth, action
            )

        return {
            "obs": obs,                                          # (3, 48, 48)
            "goal": goal_img,                                    # (3, 48, 48)
            "depth": depth,                                      # (1, 48, 48)
            "action": torch.tensor(action, dtype=torch.long),    # scalar
        }


def get_dataloader(data_root, batch_size=128, samples_per_epoch=10000,
                   augment=True, num_workers=4):
    """Create DataLoader for ImageNav training.

    Args:
        data_root: Root data directory (auto-discovers all collections).
    """
    ds = ImageNavDataset(
        data_root,
        samples_per_epoch=samples_per_epoch,
        augment=augment,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )


if __name__ == "__main__":
    # Quick test — point at data root
    DATA_ROOT = "data"

    collections = discover_collections(DATA_ROOT)
    for cdir in collections:
        graph = NavGraph(cdir)
        keys = graph.get_all_node_keys()
        if len(keys) >= 2:
            s, g = keys[0], keys[-1]
            action = graph.shortest_path_action(s, g)
            print(f"  Shortest path {s} → {g}: {ACTION_NAMES[action]}")

    ds = ImageNavDataset(DATA_ROOT, samples_per_epoch=100, augment=False)
    sample = ds[0]
    print(f"\nSample:")
    for k, v in sample.items():
        if isinstance(v, torch.Tensor):
            print(f"  {k}: {v.shape} ({v.dtype})")
        else:
            print(f"  {k}: {v}")
