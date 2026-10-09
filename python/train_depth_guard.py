#!/usr/bin/env python3
"""
train_depth_guard.py — Distill Depth Anything V2 → tiny obstacle detector.

Training data:  NYU Depth V2 (47K indoor images) with DA-V2 pseudo-depth labels.
Student model:  TinyDepthNet — depthwise-separable CNN (~15K params).
Output:         INT8 TFLite model (~18KB) for ESP32-S3 Camera Board.

Pipeline:
    1. gen_depth.py generates DA-V2 depth maps for NYU images (48×48 .npy)
    2. This script labels each image as blocked/clear from the depth map
    3. Trains student to match teacher labels
    4. Exports INT8 TFLite + C header for firmware embedding

Usage:
    # Step 1: Generate depth maps (run once, set SOURCE="nyu" in gen_depth.py)
    python gen_depth.py

    # Step 2: Train + export
    python train_depth_guard.py

    # Step 3: Fine-tune on robot data (set SOURCE="mcu", RESUME_FROM below)
    python train_depth_guard.py
"""

import os
import glob
import time


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image

# ═══════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════
NYU_CACHE_DIR      = "./data/nyu_depth_v2_cache"
MCU_DATA_DIR       = "./data"
OUTPUT_DIR         = "./checkpoints/depth_guard"
IMG_SIZE           = 48
DEVICE             = "cuda" if torch.cuda.is_available() else "cpu"

# Data source
SOURCE             = "nyu"        # "nyu" or "mcu"
RESUME_FROM        = None         # Path to checkpoint for fine-tuning, e.g.
                                  # "./checkpoints/depth_guard/best.pt"

# Depth labeling
CENTER_STRIP_FRAC  = 0.4          # fraction of width treated as "ahead"
OBSTACLE_THRESHOLD = 0.65         # DA-V2 disparity > this → blocked
BALANCE_WEIGHT     = True

# Model
STUDENT_CHANNELS   = [8, 16, 32]  # depthwise-separable layers

# Training
LR                 = 3e-4
BATCH_SIZE         = 64
EPOCHS             = 80
VAL_SPLIT          = 0.15
SEED               = 42
# ═══════════════════════════════════════════════


# ═══════════════════════════════════════════════
#  Student Model: TinyDepthNet
# ═══════════════════════════════════════════════

class DepthwiseSeparableConv(nn.Module):
    """Depthwise-separable convolution — efficient for MCUs."""

    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_ch, in_ch, 3, stride=stride, padding=1,
            groups=in_ch, bias=False
        )
        self.pointwise = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)


class TinyDepthNet(nn.Module):
    """
    Ultra-lightweight obstacle detector for ESP32-S3.

    Input:  (B, 3, 48, 48) RGB image
    Output: (B, 1) obstacle logit (apply sigmoid for probability)

    Uses depthwise-separable convolutions for minimum parameter count.
    Target: <20KB INT8.
    """

    def __init__(self, channels=None):
        super().__init__()
        if channels is None:
            channels = STUDENT_CHANNELS

        self.features = nn.Sequential(
            # 48×48 → 24×24
            nn.Conv2d(3, channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),

            # 24×24 → 12×12
            DepthwiseSeparableConv(channels[0], channels[1], stride=2),

            # 12×12 → 6×6
            DepthwiseSeparableConv(channels[1], channels[2], stride=2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[2], 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def predict(self, x):
        return torch.sigmoid(self.forward(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def model_size_kb(self):
        return self.count_params() / 1024


# ═══════════════════════════════════════════════
#  Dataset
# ═══════════════════════════════════════════════

def label_from_depth(depth_map, center_frac=CENTER_STRIP_FRAC,
                     threshold=OBSTACLE_THRESHOLD):
    """
    Label image as blocked/clear from DA-V2 depth map.

    DA-V2 outputs disparity: high values = close objects.
    Check bottom-center strip (where robot would drive toward).

    Returns: 1.0 if blocked, 0.0 if clear
    """
    h, w = depth_map.shape
    margin = int(w * (1 - center_frac) / 2)
    center = depth_map[:, margin:w - margin]
    bottom_center = center[h // 2:, :]
    max_depth = np.max(bottom_center) if bottom_center.size > 0 else 0.0
    return 1.0 if max_depth > threshold else 0.0


def discover_nyu_pairs(cache_dir):
    """Find (image, depth) pairs from NYU cache directory."""
    pairs = []
    for split in ["train", "validation"]:
        img_dir = os.path.join(cache_dir, split, "images")
        depth_dir = os.path.join(cache_dir, split, "depth")
        if not os.path.isdir(img_dir) or not os.path.isdir(depth_dir):
            continue
        for img_file in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
            basename = os.path.splitext(os.path.basename(img_file))[0]
            depth_file = os.path.join(depth_dir, f"{basename}.npy")
            if os.path.exists(depth_file):
                pairs.append((img_file, depth_file))
    return pairs


def discover_mcu_pairs(data_dir):
    """Find (image, depth) pairs from MCU-collected sessions."""
    pairs = []
    for session in sorted(os.listdir(data_dir)):
        img_dir = os.path.join(data_dir, session, "images")
        depth_dir = os.path.join(data_dir, session, "depth")
        if not os.path.isdir(img_dir) or not os.path.isdir(depth_dir):
            continue
        for img_file in sorted(glob.glob(os.path.join(img_dir, "*.jpg"))):
            basename = os.path.splitext(os.path.basename(img_file))[0]
            depth_file = os.path.join(depth_dir, f"{basename}.npy")
            if os.path.exists(depth_file):
                pairs.append((img_file, depth_file))
    return pairs


class DepthGuardDataset(Dataset):
    """Dataset with binary obstacle labels from DA-V2 depth maps."""

    def __init__(self, pairs, augment=True):
        self.pairs = pairs
        self.augment = augment

        # Pre-compute labels
        self.labels = []
        blocked = 0
        for _, depth_path in pairs:
            depth = np.load(depth_path)
            label = label_from_depth(depth)
            self.labels.append(label)
            blocked += int(label)

        self.blocked_ratio = blocked / max(len(pairs), 1)
        clear = len(pairs) - blocked
        print(f"[Data] Labels: {blocked} blocked / {clear} clear "
              f"({self.blocked_ratio:.1%} blocked)")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, _ = self.pairs[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert("RGB")
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0

        if self.augment:
            # Random brightness jitter
            if np.random.random() < 0.5:
                arr = np.clip(arr * np.random.uniform(0.7, 1.3), 0, 1)
            # Random horizontal flip
            if np.random.random() < 0.5:
                arr = arr[:, ::-1, :].copy()
            # Random small color shift
            if np.random.random() < 0.3:
                shift = np.random.uniform(-0.05, 0.05, (1, 1, 3)).astype(
                    np.float32
                )
                arr = np.clip(arr + shift, 0, 1)

        tensor = torch.from_numpy(arr).permute(2, 0, 1)  # (3, H, W)
        return tensor, torch.tensor([label], dtype=torch.float32)


# ═══════════════════════════════════════════════
#  Training
# ═══════════════════════════════════════════════

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(imgs)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(imgs)

    return total_loss / total, correct / total


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    tp = fp = tn = fn = 0

    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        logits = model(imgs)
        loss = criterion(logits, labels)

        total_loss += loss.item() * len(imgs)
        preds = (torch.sigmoid(logits) > 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(imgs)

        tp += ((preds == 1) & (labels == 1)).sum().item()
        fp += ((preds == 1) & (labels == 0)).sum().item()
        tn += ((preds == 0) & (labels == 0)).sum().item()
        fn += ((preds == 0) & (labels == 1)).sum().item()

    acc = correct / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-8)

    return total_loss / max(total, 1), acc, precision, recall, f1



    # Final save
    torch.save(model.state_dict(),
               os.path.join(OUTPUT_DIR, "final.pt"))

    print(f"\n{'═' * 60}")
    print(f"  Training complete!")
    print(f"  Best F1:   {best_f1:.3f}")
    print(f"  Params:    {n_params:,} (~{est_size:.1f} KB INT8)")
    print(f"  Output:    {OUTPUT_DIR}")
    print(f"  Source:    {SOURCE.upper()}")
    print(f"{'═' * 60}")
    print(f"\n  Next steps:")
    if SOURCE == "nyu":
        print(f"  • Fine-tune on MCU data:")
        print(f"    Set SOURCE='mcu', RESUME_FROM='{OUTPUT_DIR}/best.pt'")
    print(f"  • Quantize + export:")
    print(f"    python quantize_depth_guard.py")


if __name__ == "__main__":
    main()

