#!/usr/bin/env python3
"""
quantize_depth_guard.py — Convert trained depth guard to INT8 LiteRT (.tflite).

Two-step decoupled workflow (Google recommended):
    1. litert_torch.convert()  → FP32 .tflite
    2. ai_edge_quantizer       → INT8 .tflite

Then generates C header for ESP32-S3 firmware embedding.

Requires:
    pip install litert-torch ai-edge-quantizer

Usage:
    python quantize_depth_guard.py
"""

import os
import glob
import time

import numpy as np
import torch
from PIL import Image

# ═══════════════════════════════════════════════
#  Configuration — edit these constants directly
# ═══════════════════════════════════════════════

# Input: trained PyTorch checkpoint
CHECKPOINT_PATH    = "./checkpoints/depth_guard/best.pt"

# Output paths
OUTPUT_DIR         = "./checkpoints/depth_guard"
FP32_TFLITE_PATH   = "./checkpoints/depth_guard/depth_guard_fp32.tflite"
INT8_TFLITE_PATH   = "./checkpoints/depth_guard/depth_guard_int8.tflite"
C_HEADER_PATH      = "../esp32/cam_board/main/model/depth_guard_model.h"
MODEL_COPY_DIR     = "../esp32/cam_board/main/model"

# Calibration data
NYU_CACHE_DIR      = "./data/nyu_depth_v2_cache"
N_CALIBRATION      = 200         # Number of calibration samples for INT8

# Model architecture (must match train_depth_guard.py)
IMG_SIZE           = 48
STUDENT_CHANNELS   = [8, 16, 32]
# ═══════════════════════════════════════════════


# ─── Import TinyDepthNet from training script ───
# Re-define here to keep this script self-contained.

import torch.nn as nn
import torch.nn.functional as F


class DepthwiseSeparableConv(nn.Module):
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
    def __init__(self, channels=None):
        super().__init__()
        if channels is None:
            channels = STUDENT_CHANNELS
        self.features = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
            DepthwiseSeparableConv(channels[0], channels[1], stride=2),
            DepthwiseSeparableConv(channels[1], channels[2], stride=2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels[2], 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# ═══════════════════════════════════════════════
#  Step 1: PyTorch → FP32 LiteRT
# ═══════════════════════════════════════════════

def convert_to_fp32_tflite(model, output_path):
    """Convert PyTorch model to FP32 .tflite using litert_torch."""
    import litert_torch

    model.eval()
    sample_inputs = (torch.randn(1, 3, IMG_SIZE, IMG_SIZE),)

    print("[Step 1] Converting PyTorch → FP32 LiteRT...")
    edge_model = litert_torch.convert(model, sample_inputs)

    # Validate conversion
    test_input = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        pt_output = model(test_input).numpy()
    edge_output = edge_model(test_input)

    if np.allclose(pt_output, edge_output, atol=1e-4, rtol=1e-4):
        print("  ✓ FP32 conversion validated — outputs match")
    else:
        max_diff = np.max(np.abs(pt_output - edge_output))
        print(f"  ⚠ Output mismatch (max diff: {max_diff:.6f})")

    edge_model.export(output_path)
    size_kb = os.path.getsize(output_path) / 1024
    print(f"  ✓ FP32 .tflite saved: {output_path} ({size_kb:.1f} KB)")
    return output_path


# ═══════════════════════════════════════════════
#  Step 2: FP32 .tflite → INT8 .tflite
# ═══════════════════════════════════════════════

def collect_calibration_data():
    """Load calibration images from NYU cache."""
    img_dir = os.path.join(NYU_CACHE_DIR, "train", "images")
    if not os.path.isdir(img_dir):
        print(f"  [WARN] No calibration data at {img_dir}")
        return None

    image_files = sorted(glob.glob(os.path.join(img_dir, "*.jpg")))
    n = min(len(image_files), N_CALIBRATION)

    cal_data = []
    for img_path in image_files[:n]:
        img = Image.open(img_path).convert("RGB")
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        arr = np.array(img, dtype=np.float32) / 255.0
        # (H, W, C) → (1, C, H, W)
        arr = np.transpose(arr, (2, 0, 1))[np.newaxis, ...]
        cal_data.append(arr)

    cal_data = np.concatenate(cal_data, axis=0)
    print(f"  Calibration data: {cal_data.shape} ({n} samples)")
    return cal_data


def quantize_to_int8(fp32_path, int8_path):
    """Quantize FP32 .tflite to INT8 using ai-edge-quantizer."""
    from ai_edge_quantizer import quantizer

    print("[Step 2] Quantizing FP32 → INT8...")

    qt = quantizer.Quantizer(fp32_path)

    # Apply full integer static quantization recipe
    # This quantizes both weights and activations to INT8
    qt.update_quantization_recipe(
        regex=".*",
        operation_name=quantizer.OperationName.ALL_SUPPORTED,
        algorithm_key=quantizer.AlgorithmName.MIN_MAX_UNIFORM_QUANT,
        op_config=quantizer.OpQuantizationConfig(
            weight_tensor_config=quantizer.TensorQuantizationConfig(
                num_bits=8,
                symmetric=True,
                granularity=quantizer.QuantGranularity.CHANNELWISE,
            ),
            activation_tensor_config=quantizer.TensorQuantizationConfig(
                num_bits=8,
                symmetric=True,
                granularity=quantizer.QuantGranularity.TENSORWISE,
            ),
        ),
    )

    # Collect calibration data
    cal_data = collect_calibration_data()
    if cal_data is not None:
        # Build calibration dataset as dict: {input_name: data}
        # The ai-edge-quantizer needs numpy arrays for calibration
        calibration_dataset = {"serving_default_x:0": cal_data}
        qt.calibrate(calibration_dataset)

    # Quantize and export
    qt.quantize()
    qt.export_model(int8_path)

    size_kb = os.path.getsize(int8_path) / 1024
    print(f"  ✓ INT8 .tflite saved: {int8_path} ({size_kb:.1f} KB)")
    return int8_path


# ═══════════════════════════════════════════════
#  Step 3: Generate C header
# ═══════════════════════════════════════════════

def export_c_header(tflite_path, header_path):
    """Convert .tflite to C byte array for ESP32 firmware embedding."""
    with open(tflite_path, "rb") as f:
        data = f.read()

    os.makedirs(os.path.dirname(header_path), exist_ok=True)

    with open(header_path, "w") as f:
        f.write("/**\n")
        f.write(" * @file depth_guard_model.h\n")
        f.write(" * @brief INT8 LiteRT model — depth guard obstacle detector.\n")
        f.write(" *\n")
        f.write(" * Distilled from Depth Anything V2 Small.\n")
        f.write(" * Trained on NYU Depth V2 (47K indoor images).\n")
        f.write(" * Quantized with ai-edge-quantizer (full INT8).\n")
        f.write(f" * Size: {len(data)} bytes ({len(data) / 1024:.1f} KB)\n")
        f.write(" * Input:  1×48×48×3 INT8 image\n")
        f.write(" * Output: 1 INT8 obstacle logit\n")
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

    print(f"  ✓ C header saved: {header_path}")


# ═══════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════

def main():
    print("═" * 60)
    print("  Depth Guard — LiteRT INT8 Quantization")
    print("═" * 60)

    # Load trained model
    print(f"\n[Load] Checkpoint: {CHECKPOINT_PATH}")
    if not os.path.exists(CHECKPOINT_PATH):
        print(f"  [ERROR] Checkpoint not found: {CHECKPOINT_PATH}")
        print(f"  Run train_depth_guard.py first.")
        return

    model = TinyDepthNet()
    model.load_state_dict(
        torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    )
    model.eval()

    n_params = model.count_params()
    print(f"  TinyDepthNet: {n_params:,} params")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Step 1: PyTorch → FP32 .tflite
    print()
    convert_to_fp32_tflite(model, FP32_TFLITE_PATH)

    # Step 2: FP32 → INT8 .tflite
    print()
    quantize_to_int8(FP32_TFLITE_PATH, INT8_TFLITE_PATH)

    # Step 3: Generate C header
    print()
    print("[Step 3] Generating C header for firmware...")
    export_c_header(INT8_TFLITE_PATH, C_HEADER_PATH)

    # Copy .tflite to firmware model dir
    import shutil
    os.makedirs(MODEL_COPY_DIR, exist_ok=True)
    shutil.copy2(INT8_TFLITE_PATH,
                 os.path.join(MODEL_COPY_DIR, "depth_guard.tflite"))
    print(f"  ✓ Model copied to {MODEL_COPY_DIR}")

    # Summary
    fp32_kb = os.path.getsize(FP32_TFLITE_PATH) / 1024
    int8_kb = os.path.getsize(INT8_TFLITE_PATH) / 1024

    print(f"\n{'═' * 60}")
    print(f"  Quantization complete!")
    print(f"  FP32 model:  {fp32_kb:.1f} KB")
    print(f"  INT8 model:  {int8_kb:.1f} KB")
    print(f"  Compression: {fp32_kb / max(int8_kb, 0.1):.1f}×")
    print(f"  Params:      {n_params:,}")
    print(f"{'═' * 60}")
    print(f"\n  Next steps:")
    print(f"  1. Uncomment #define DEPTH_GUARD_MODEL_AVAILABLE in inference.c")
    print(f"  2. Rebuild: cd esp32/cam_board && idf.py fullclean && idf.py build")


if __name__ == "__main__":
    main()
