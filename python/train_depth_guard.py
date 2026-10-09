import os
import glob

import litert_torch
from litert_torch.interpreter import Interpreter
from ai_edge_quantizer import algorithm_manager, calibrator, qtyping, quantizer, recipe


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

# Model — 4-stage hierarchical encoder (mini Depth Anything)
STAGE_CHANNELS     = [16, 24, 48, 96]

# Training
LR                 = 3e-4
BATCH_SIZE         = 64
EPOCHS             = 80
VAL_SPLIT          = 0.15
SEED               = 42
# ═══════════════════════════════════════════════


# ═══════════════════════════════════════════════
#  Student Model: MicroDepthAnything
#
#  Miniature Depth Anything V2 — 4-stage hierarchical encoder
#  mirroring DA-V2's DINOv2 backbone + DPT head structure.
#
#  Stage channels: [16, 24, 48, 96] at resolutions [24², 12², 6², 3²]
#  ~25K params → ~25KB INT8 for ESP32-S3
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


class MicroDepthAnything(nn.Module):
    """
    Miniature Depth Anything V2 student model (~25K params).

    Mirrors DA-V2's hierarchical design:
      - Stem: patch embedding (3→C0) like DINOv2's patch projection
      - Stage 1–3: paired depthwise-separable blocks at decreasing
        resolutions, analogous to DINOv2's multi-scale features
      - Head: global pool → 2-layer MLP classifier

    Input:  (B, 3, 48, 48)  RGB image
    Output: (B, 1)          obstacle logit (sigmoid → probability)

    48×48 → stem[24²] → S1[12²] → S2[6²] → S3[3²] → GAP → MLP → 1
    """

    def __init__(self, channels=None):
        super().__init__()
        if channels is None:
            channels = STAGE_CHANNELS
        c0, c1, c2, c3 = channels

        # Stem — patch embedding (like DINOv2 patch projection)
        # 48×48 → 24×24
        self.stem = nn.Sequential(
            nn.Conv2d(3, c0, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c0),
            nn.ReLU(inplace=True),
        )

        # Stage 1 — 24×24 → 12×12
        self.stage1 = nn.Sequential(
            DepthwiseSeparableConv(c0, c1, stride=2),
            DepthwiseSeparableConv(c1, c1, stride=1),
        )

        # Stage 2 — 12×12 → 6×6
        self.stage2 = nn.Sequential(
            DepthwiseSeparableConv(c1, c2, stride=2),
            DepthwiseSeparableConv(c2, c2, stride=1),
        )

        # Stage 3 — 6×6 → 3×3
        self.stage3 = nn.Sequential(
            DepthwiseSeparableConv(c2, c3, stride=2),
            DepthwiseSeparableConv(c3, c3, stride=1),
        )

        # Classification head — DPT-style projection + classifier
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(c3, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        return self.head(x)

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


<<<<<<< HEAD
=======
# ═══════════════════════════════════════════════
#  LiteRT INT8 Export (via ai_edge_torch)
# ═══════════════════════════════════════════════

def export_tflite_int8(model, output_path, calibration_loader, device,
                       n_cal=200):
    """
    Export PyTorch model to INT8 LiteRT (.tflite) via ai_edge_torch.

    Direct path: PyTorch → ai_edge_torch.convert() → .tflite
    With PT2E quantization for full INT8.

    Requires: pip install ai-edge-torch
    """
    model.eval().cpu()
    sample_input = (torch.randn(1, 3, IMG_SIZE, IMG_SIZE),)

    # Collect calibration data
    cal_data = []
    for imgs, _ in calibration_loader:
        for img in imgs:
            cal_data.append(img.unsqueeze(0))
            if len(cal_data) >= n_cal:
                break
        if len(cal_data) >= n_cal:
            break
    print(f"[Export] Calibration samples: {len(cal_data)}")

    try:
        model = model.eval().cpu()
        sample_args = (sample_input,)  # convert() expects a tuple of args
    
        # Step 1: convert to an unquantized float .tflite
        float_path = os.path.splitext(output_path)[0] + "_float.tflite"
        litert_torch.convert(model, sample_args).export(float_path)
    
        # Read the signature key, input name and input shape from the converted model
        interp = Interpreter(model_path=float_path)
        signatures = interp.get_signature_list()
        sig_key = "serving_default" if "serving_default" in signatures else next(iter(signatures))
        input_name = signatures[sig_key]["inputs"][0]  # single input model assumed
        input_shape = tuple(interp.get_input_details()[0]["shape"])
    
        # Step 2: static INT8 recipe
        qt = quantizer.Quantizer(float_path)
        qt.load_quantization_recipe(recipe.static_wi8_ai8())
    
        if float_io:
            # Leave the model's input and output tensors in float32
            for op_name in (qtyping.TFLOperationName.INPUT, qtyping.TFLOperationName.OUTPUT):
                qt.update_quantization_recipe(
                    regex=".*",
                    operation_name=op_name,
                    algorithm_key=algorithm_manager.AlgorithmName.NO_QUANTIZE,
                )
    
        # Step 3: calibrate with real data, as numpy float32 matching the .tflite input shape
        samples = []
        for x in cal_data:
            arr = x.detach().cpu().numpy().astype(np.float32)
            if tuple(arr.shape) != input_shape:
                raise ValueError(
                    f"Calibration sample shape {tuple(arr.shape)} does not match "
                    f"the .tflite input shape {input_shape}"
                )
            samples.append({input_name: arr})
    
        calibration_result = qt.calibrate(
            {sig_key: samples},
            mode=calibrator.CalibrationMode.CALIBRATION_PROFILER_BASED,
        )
    
        # Step 4: quantize and save
        if os.path.exists(output_path):
            os.remove(output_path)
        qt.quantize(calibration_result=calibration_result).export_model(output_path)
    
        size_kb = os.path.getsize(output_path) / 1024
        print(f"[Export] LiteRT INT8 saved: {output_path} ({size_kb:.1f} KB)")
        return output_path
    except ImportError as e:
        print("[Export] ai_edge_torch not installed.")
        print("  Install: pip install ai-edge-torch")

        print(e)

        # Fallback: save FP32 .pt for manual conversion
        fallback_path = output_path.replace(".tflite", ".pt")
        torch.save(model.state_dict(), fallback_path)
        print(f"[Export] Saved FP32 weights: {fallback_path}")
        print(f"  Convert manually with ai_edge_torch later.")
        return fallback_path

    except Exception as e:
        print(f"[Export] PT2E quantization failed: {e}")
        print("[Export] Falling back to FP32 LiteRT export...")

        try:
            import litert_torch
            edge_model = litert_torch.convert(model, sample_input)
            fp32_path = output_path.replace(".tflite", "_fp32.tflite")
            edge_model.export(fp32_path)

            size_kb = os.path.getsize(fp32_path) / 1024
            print(f"[Export] FP32 LiteRT saved: {fp32_path} ({size_kb:.1f} KB)")
            print("  Quantize with: ai-edge-quantizer")
            return fp32_path

        except Exception as e2:
            print(f"[Export] FP32 export also failed: {e2}")
            fallback_path = output_path.replace(".tflite", ".pt")
            torch.save(model.state_dict(), fallback_path)
            return fallback_path


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
    print(f"  Student:  MicroDepthAnything ({STAGE_CHANNELS})")
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
    model = MicroDepthAnything().to(device)
    n_params = model.count_params()
    est_size = model.model_size_kb()

    # Resume from checkpoint (for fine-tuning)
    if RESUME_FROM:
        print(f"[Model] Loading weights from: {RESUME_FROM}")
        model.load_state_dict(
            torch.load(RESUME_FROM, map_location=device, weights_only=True)
        )

    print(f"\n[Model] MicroDepthAnything: {n_params:,} params "
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

    # for epoch in range(1, EPOCHS + 1):
    #     t0 = time.time()
    #     train_loss, train_acc = train_epoch(
    #         model, train_loader, optimizer, criterion, device
    #     )
    #     val_loss, val_acc, prec, rec, f1 = eval_epoch(
    #         model, val_loader, criterion, device
    #     )
    #     scheduler.step()
    #     dt = time.time() - t0

    #     print(f"E{epoch:>3}/{EPOCHS} │ "
    #           f"TrL={train_loss:.4f} TrA={train_acc:.3f} │ "
    #           f"VlL={val_loss:.4f} VlA={val_acc:.3f} │ "
    #           f"P={prec:.3f} R={rec:.3f} F1={f1:.3f} │ "
    #           f"{dt:.1f}s")

    #     if f1 > best_f1:
    #         best_f1 = f1
    #         ckpt_path = os.path.join(OUTPUT_DIR, "best.pt")
    #         torch.save(model.state_dict(), ckpt_path)
    #         print(f"    ↳ New best F1={f1:.3f}, saved!")
>>>>>>> 5e663bb (some changes)

    # # Final save
    # torch.save(model.state_dict(),
    #            os.path.join(OUTPUT_DIR, "final.pt"))

<<<<<<< HEAD
=======
    # Load best for export
    model.load_state_dict(
        torch.load(os.path.join(OUTPUT_DIR, "best.pt"),
                   map_location="cpu", weights_only=True)
    )

    # Export TFLite INT8
    print(f"\n{'─' * 60}")
    print(f"  Exporting to TFLite INT8...")
    print(f"{'─' * 60}")

    model.eval()

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

>>>>>>> 5e663bb (some changes)
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

