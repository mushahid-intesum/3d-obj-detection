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
import struct


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
MODEL_DIR          = "../esp32/cam_board/main/model"
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

# TFLite export
TFLITE_OPSET       = 13
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


# ═══════════════════════════════════════════════
#  TFLite INT8 Export
# ═══════════════════════════════════════════════

def export_tflite_int8(model, output_path, calibration_loader, device,
                       n_cal=200):
    """
    Export PyTorch model to INT8 TFLite via ONNX → TensorFlow → TFLite.

    Requires: pip install onnx onnx2tf tensorflow
    """
    import onnx

    model.eval().cpu()
    dummy = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    # Step 1: PyTorch → ONNX
    onnx_path = output_path.replace(".tflite", ".onnx")
    torch.onnx.export(
        model, dummy, onnx_path,
        input_names=["image"],
        output_names=["obstacle_logit"],
        opset_version=TFLITE_OPSET,
        dynamic_axes=None,
    )
    print(f"[Export] ONNX saved: {onnx_path}")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"[Export] ONNX validation passed")

    # Step 2: ONNX → TFLite INT8
    try:
        import tensorflow as tf

        savedmodel_dir = output_path.replace(".tflite", "_saved_model")
        os.system(f"onnx2tf -i {onnx_path} -o {savedmodel_dir} -osd -nuo")

        # Collect calibration data from NYU training set
        cal_data = []
        for imgs, _ in calibration_loader:
            for img in imgs:
                cal_data.append(img.numpy())
                if len(cal_data) >= n_cal:
                    break
            if len(cal_data) >= n_cal:
                break
        cal_data = np.array(cal_data, dtype=np.float32)
        print(f"[Export] Calibration samples: {len(cal_data)}")

        def representative_dataset():
            for i in range(len(cal_data)):
                yield [cal_data[i:i + 1]]

        # Full integer quantization
        converter = tf.lite.TFLiteConverter.from_saved_model(savedmodel_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8
        ]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()

        with open(output_path, "wb") as f:
            f.write(tflite_model)

        size_kb = len(tflite_model) / 1024
        print(f"[Export] TFLite INT8 saved: {output_path} ({size_kb:.1f} KB)")

        return output_path

    except ImportError:
        print("[Export] tensorflow or onnx2tf not installed.")
        print(f"  To convert manually:")
        print(f"    pip install onnx2tf tensorflow")
        print(f"    onnx2tf -i {onnx_path} -o saved_model -osd")
        print(f"    Then quantize with tf.lite.TFLiteConverter.")
        return onnx_path


def export_c_header(tflite_path, header_path):
    """Convert .tflite to C byte array for ESP32 firmware embedding."""
    with open(tflite_path, "rb") as f:
        data = f.read()

    with open(header_path, "w") as f:
        f.write("/**\n")
        f.write(" * @file depth_guard_model.h\n")
        f.write(" * @brief INT8 TFLite model — depth guard obstacle detector.\n")
        f.write(" *\n")
        f.write(" * Distilled from Depth Anything V2 Small.\n")
        f.write(" * Trained on NYU Depth V2 (47K indoor images).\n")
        f.write(f" * Size: {len(data)} bytes ({len(data) / 1024:.1f} KB)\n")
        f.write(f" * Input:  1×48×48×3 INT8 image\n")
        f.write(f" * Output: 1 INT8 obstacle logit\n")
        f.write(" */\n")
        f.write("#ifndef DEPTH_GUARD_MODEL_H\n")
        f.write("#define DEPTH_GUARD_MODEL_H\n\n")
        f.write("#include <stddef.h>\n\n")
        f.write(f"#define DEPTH_GUARD_MODEL_LEN {len(data)}\n\n")
        f.write("alignas(16) static const unsigned char\n")
        f.write("depth_guard_model_data[] = {\n")

        for i in range(0, len(data), 16):
            chunk = data[i:i + 16]
            hex_str = ", ".join(f"0x{b:02x}" for b in chunk)
            f.write(f"    {hex_str},\n")

        f.write("};\n\n")
        f.write("#endif /* DEPTH_GUARD_MODEL_H */\n")

    print(f"[Export] C header saved: {header_path}")


# ═══════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  Depth Guard — Depth Anything V2 Distillation")
    print(f"  Teacher:  Depth Anything V2 Small (25M params)")
    print(f"  Student:  TinyDepthNet ({STUDENT_CHANNELS})")
    print(f"  Source:   {SOURCE.upper()}")
    print(f"  Device:   {DEVICE}")
    print("═" * 60)

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Discover data
    if SOURCE == "nyu":
        pairs = discover_nyu_pairs(NYU_CACHE_DIR)
    else:
        pairs = discover_mcu_pairs(MCU_DATA_DIR)

    print(f"[Data] Found {len(pairs)} image-depth pairs")

    if len(pairs) == 0:
        print("\n[ERROR] No image-depth pairs found.")
        if SOURCE == "nyu":
            print("Run first:  python gen_depth.py  (with SOURCE='nyu')")
        else:
            print("Run first:  python gen_depth.py  (with SOURCE='mcu')")
        return

    # Create dataset
    full_ds = DepthGuardDataset(pairs, augment=True)

    # Split
    val_size = int(len(full_ds) * VAL_SPLIT)
    train_size = len(full_ds) - val_size
    train_ds, val_ds = random_split(full_ds, [train_size, val_size])
    val_ds_no_aug = DepthGuardDataset(
        [pairs[i] for i in val_ds.indices], augment=False
    )

    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds_no_aug, batch_size=BATCH_SIZE, shuffle=False, num_workers=4
    )

    # Model
    device = torch.device(DEVICE)
    model = TinyDepthNet().to(device)
    n_params = model.count_params()
    est_size = model.model_size_kb()

    # Resume from checkpoint (for fine-tuning)
    if RESUME_FROM:
        print(f"[Model] Loading weights from: {RESUME_FROM}")
        model.load_state_dict(
            torch.load(RESUME_FROM, map_location=device, weights_only=True)
        )

    print(f"\n[Model] TinyDepthNet: {n_params:,} params "
          f"(~{est_size:.1f} KB INT8)")

    # Class-balanced loss
    pos_weight = None
    if BALANCE_WEIGHT and full_ds.blocked_ratio > 0:
        pw = (1.0 - full_ds.blocked_ratio) / full_ds.blocked_ratio
        pos_weight = torch.tensor([pw], device=device)
        print(f"[Train] Positive weight: {pw:.2f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LR, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, EPOCHS
    )

    # Training loop
    best_f1 = 0.0
    print(f"\n[Train] {EPOCHS} epochs, batch={BATCH_SIZE}, "
          f"train={train_size}, val={val_size}\n")

    for epoch in range(1, EPOCHS + 1):
        t0 = time.time()
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device
        )
        val_loss, val_acc, prec, rec, f1 = eval_epoch(
            model, val_loader, criterion, device
        )
        scheduler.step()
        dt = time.time() - t0

        print(f"E{epoch:>3}/{EPOCHS} │ "
              f"TrL={train_loss:.4f} TrA={train_acc:.3f} │ "
              f"VlL={val_loss:.4f} VlA={val_acc:.3f} │ "
              f"P={prec:.3f} R={rec:.3f} F1={f1:.3f} │ "
              f"{dt:.1f}s")

        if f1 > best_f1:
            best_f1 = f1
            ckpt_path = os.path.join(OUTPUT_DIR, "best.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"    ↳ New best F1={f1:.3f}, saved!")

    # Final save
    torch.save(model.state_dict(),
               os.path.join(OUTPUT_DIR, "final.pt"))

    # Load best for export
    model.load_state_dict(
        torch.load(os.path.join(OUTPUT_DIR, "best.pt"),
                   map_location="cpu", weights_only=True)
    )

    # Export TFLite INT8
    print(f"\n{'─' * 60}")
    print(f"  Exporting to TFLite INT8...")
    print(f"{'─' * 60}")

    tflite_path = os.path.join(OUTPUT_DIR, "depth_guard.tflite")
    result_path = export_tflite_int8(
        model, tflite_path, train_loader, device
    )

    # Export C header if TFLite was created
    if result_path.endswith(".tflite") and os.path.exists(result_path):
        os.makedirs(MODEL_DIR, exist_ok=True)
        header_path = os.path.join(MODEL_DIR, "depth_guard_model.h")
        export_c_header(result_path, header_path)

        import shutil
        shutil.copy2(result_path,
                     os.path.join(MODEL_DIR, "depth_guard.tflite"))
        print(f"[Export] Model copied to {MODEL_DIR}")

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
    print(f"  • Enable model in firmware:")
    print(f"    Uncomment #define DEPTH_GUARD_MODEL_AVAILABLE in inference.c")
    print(f"  • Rebuild: cd esp32/cam_board && idf.py fullclean && idf.py build")


if __name__ == "__main__":
    main()
