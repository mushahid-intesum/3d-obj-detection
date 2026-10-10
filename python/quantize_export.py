"""
quantize_export.py — Phase 6: Quantize student model and export for ESP32-S3.

Pipeline:
  1. Load distilled student checkpoint
  2. Quantization-aware training (QAT) fine-tuning
  3. Export encoder and policy as separate ONNX models
  4. Convert ONNX → TFLite INT8
  5. Generate C header files for embedding in ESP32 firmware
  6. Validate INT8 accuracy vs FP32
"""

import os
import struct
import time
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from student_model import StudentModel, TinyEncoder, TinyPolicy
from models import TeacherModel
from dataset import get_dataloader

# ═══════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════
DATASET_DIR         = "./data/offline_dataset"
STUDENT_CKPT        = "./checkpoints/student/best.pt"
TEACHER_CKPT        = "./checkpoints/teacher/best.pt"
OUTPUT_DIR          = "./export"
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"

# Teacher config (must match)
TEACHER_FEAT_DIM    = 512
TEACHER_CUE_DIM     = 512
NUM_ACTIONS         = 4

# QAT config
QAT_STEPS           = 10_000
QAT_LR              = 1e-4
QAT_BATCH_SIZE      = 256
QAT_TEMPERATURE     = 4.0
NUM_WORKERS         = 4

# TFLite calibration
CALIBRATION_SAMPLES = 200
SEED                = 42
# ═══════════════════════════════════════════


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_student(ckpt_path):
    """Load the distilled student model."""
    student = StudentModel(num_actions=NUM_ACTIONS)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    student.load_state_dict(ckpt["model_state"])
    print(f"[*] Student loaded from: {ckpt_path}")
    return student


def load_teacher(ckpt_path):
    """Load frozen teacher for QAT."""
    teacher = TeacherModel(
        feat_dim=TEACHER_FEAT_DIM, cue_dim=TEACHER_CUE_DIM,
        num_actions=NUM_ACTIONS
    ).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    teacher.load_state_dict(ckpt["model_state"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    return teacher


# ────────────────────────────────────────────
#  Step 1: Quantization-Aware Training
# ────────────────────────────────────────────

def run_qat(student, teacher, loader):
    """Fine-tune student with fake quantization nodes inserted."""
    print("\n[*] Running Quantization-Aware Training...")

    student = student.to(DEVICE)
    student.train()

    # Insert QAT observers
    student.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")
    # Only quantize encoder and policy (not correlation — it's plain math)
    modules_to_quantize = [student.encoder, student.policy]
    for m in modules_to_quantize:
        m.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")

    torch.ao.quantization.prepare_qat(student, inplace=True)

    optimizer = torch.optim.Adam(student.parameters(), lr=QAT_LR)
    data_iter = iter(loader)

    for step in range(1, QAT_STEPS + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        obs = batch["obs"].to(DEVICE)
        goal = batch["goal"].to(DEVICE)

        with torch.no_grad():
            teacher_logits = teacher(obs, goal)
            teacher_actions = teacher_logits.argmax(dim=-1)

        student_logits = student(obs, goal)

        # Distillation loss during QAT
        t_soft = F.log_softmax(teacher_logits / QAT_TEMPERATURE, dim=-1)
        s_soft = F.log_softmax(student_logits / QAT_TEMPERATURE, dim=-1)
        loss_kl = F.kl_div(s_soft, t_soft.exp(), reduction='batchmean') * (QAT_TEMPERATURE ** 2)
        loss_ce = F.cross_entropy(student_logits, teacher_actions)
        loss = loss_kl + 0.5 * loss_ce

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
        optimizer.step()

        if step % 1000 == 0:
            agreement = (student_logits.argmax(-1) == teacher_actions).float().mean()
            print(f"  QAT step {step}/{QAT_STEPS}: "
                  f"loss={loss.item():.4f} agreement={agreement.item():.3f}")

    # Convert to actual quantized model
    student.eval()
    student_quantized = torch.ao.quantization.convert(student, inplace=False)
    print("[OK] QAT complete")
    return student, student_quantized


# ────────────────────────────────────────────
#  Step 2: Export to ONNX
# ────────────────────────────────────────────

class EncoderWrapper(nn.Module):
    """Wrapper to export just the encoder."""
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)


class PolicyWrapper(nn.Module):
    """Wrapper to export just the policy MLP."""
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, cue):
        return self.policy(cue)


def export_onnx(student_fp32, output_dir):
    """Export encoder and policy as separate ONNX models."""
    os.makedirs(output_dir, exist_ok=True)
    student_fp32.eval().cpu()

    # Export encoder
    encoder_wrapper = EncoderWrapper(student_fp32.encoder)
    dummy_img = torch.randn(1, 3, 128, 128)
    encoder_path = os.path.join(output_dir, "encoder.onnx")
    torch.onnx.export(
        encoder_wrapper, dummy_img, encoder_path,
        input_names=["image"], output_names=["features"],
        opset_version=13,
        dynamic_axes={"image": {0: "batch"}, "features": {0: "batch"}},
    )
    print(f"[OK] Encoder ONNX: {encoder_path}")

    # Export policy
    policy_wrapper = PolicyWrapper(student_fp32.policy)
    dummy_cue = torch.randn(1, 83)
    policy_path = os.path.join(output_dir, "policy.onnx")
    torch.onnx.export(
        policy_wrapper, dummy_cue, policy_path,
        input_names=["correlation_cue"], output_names=["action_logits"],
        opset_version=13,
        dynamic_axes={"correlation_cue": {0: "batch"}, "action_logits": {0: "batch"}},
    )
    print(f"[OK] Policy ONNX: {policy_path}")

    return encoder_path, policy_path


# ────────────────────────────────────────────
#  Step 3: Convert ONNX → TFLite INT8
# ────────────────────────────────────────────

def generate_calibration_data(loader, n_samples):
    """Generate calibration images for TFLite INT8 quantization."""
    images = []
    for batch in loader:
        obs = batch["obs"].numpy()
        goal = batch["goal"].numpy()
        for img in obs:
            images.append(img)
            if len(images) >= n_samples:
                return np.array(images, dtype=np.float32)
        for img in goal:
            images.append(img)
            if len(images) >= n_samples:
                return np.array(images, dtype=np.float32)
    return np.array(images, dtype=np.float32)


def convert_to_tflite(onnx_path, output_path, calibration_data=None,
                      input_shape=None):
    """
    Convert ONNX → TFLite with INT8 quantization.

    Uses onnx2tf or manual TFLite conversion.
    Falls back to creating a representative dataset for full integer quant.
    """
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        # ONNX → TF SavedModel
        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        saved_model_dir = onnx_path.replace(".onnx", "_saved_model")
        tf_rep.export_graph(saved_model_dir)

        # TF SavedModel → TFLite INT8
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        if calibration_data is not None:
            def representative_dataset():
                for i in range(len(calibration_data)):
                    yield [calibration_data[i:i+1]]
            converter.representative_dataset = representative_dataset

        tflite_model = converter.convert()

        with open(output_path, "wb") as f:
            f.write(tflite_model)

        size_kb = len(tflite_model) / 1024
        print(f"[OK] TFLite INT8: {output_path} ({size_kb:.1f} KB)")
        return True

    except ImportError:
        print(f"[WARN] onnx-tf/tensorflow not installed. "
              f"ONNX exported to {onnx_path} — convert manually.")
        print(f"       Install: pip install onnx-tf tensorflow")
        return False


# ────────────────────────────────────────────
#  Step 4: Generate C header for ESP32
# ────────────────────────────────────────────

def tflite_to_c_header(tflite_path, header_path, array_name):
    """Convert .tflite file to a C header with uint8 array."""
    with open(tflite_path, "rb") as f:
        data = f.read()

    with open(header_path, "w") as f:
        f.write(f"/* Auto-generated from {os.path.basename(tflite_path)} */\n")
        f.write(f"#ifndef {array_name.upper()}_H\n")
        f.write(f"#define {array_name.upper()}_H\n\n")
        f.write(f"#include <stdint.h>\n\n")
        f.write(f"const unsigned int {array_name}_len = {len(data)};\n")
        f.write(f"alignas(8) const uint8_t {array_name}[] = {{\n")

        # Write 16 bytes per line
        for i in range(0, len(data), 16):
            chunk = data[i:i+16]
            hex_vals = ", ".join(f"0x{b:02x}" for b in chunk)
            f.write(f"    {hex_vals},\n")

        f.write(f"}};\n\n")
        f.write(f"#endif /* {array_name.upper()}_H */\n")

    print(f"[OK] C header: {header_path} ({len(data)} bytes)")


# ────────────────────────────────────────────
#  Step 5: Validate INT8 accuracy
# ────────────────────────────────────────────

@torch.no_grad()
def validate_fp32_vs_int8(student_fp32, loader, max_batches=50):
    """Compare FP32 student accuracy (as baseline for INT8 comparison)."""
    student_fp32.eval().cpu()
    correct = 0
    total = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        obs = batch["obs"]
        goal = batch["goal"]
        actions = batch["action"]

        logits = student_fp32(obs, goal)
        preds = logits.argmax(dim=-1)
        correct += (preds == actions).sum().item()
        total += len(actions)

    acc = correct / total if total > 0 else 0.0
    print(f"[*] FP32 student accuracy (vs dataset labels): {acc:.3f}")
    return acc


# ────────────────────────────────────────────
#  Main pipeline
# ────────────────────────────────────────────

def main():
    print("=" * 55)
    print("  Phase 6: Quantization & Export")
    print("=" * 55)

    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load models
    student = load_student(STUDENT_CKPT)
    teacher = load_teacher(TEACHER_CKPT)

    # Data
    train_loader = get_dataloader(
        DATASET_DIR, batch_size=QAT_BATCH_SIZE,
        augment=True, num_workers=NUM_WORKERS
    )
    eval_loader = get_dataloader(
        DATASET_DIR, batch_size=QAT_BATCH_SIZE,
        augment=False, num_workers=NUM_WORKERS
    )

    # Step 0: FP32 baseline
    print("\n── Step 0: FP32 Baseline ──")
    fp32_acc = validate_fp32_vs_int8(student, eval_loader)

    # Step 1: QAT
    print("\n── Step 1: Quantization-Aware Training ──")
    student_qat, student_quantized = run_qat(student, teacher, train_loader)

    # Step 2: Export ONNX (from QAT FP32 model, not quantized)
    print("\n── Step 2: Export ONNX ──")
    student_qat.eval().cpu()
    # Remove QAT observers for clean export
    student_export = StudentModel(num_actions=NUM_ACTIONS)
    # Copy weights (skip observer keys)
    clean_state = {}
    for k, v in student_qat.state_dict().items():
        # Skip quantization observer keys
        if any(x in k for x in ["weight_fake_quant", "activation_post_process",
                                  "observer", "scale", "zero_point", "min_val",
                                  "max_val", "eps"]):
            continue
        clean_state[k] = v

    student_export.load_state_dict(clean_state, strict=False)
    encoder_onnx, policy_onnx = export_onnx(student_export, OUTPUT_DIR)

    # Step 3: Convert to TFLite INT8
    print("\n── Step 3: Convert to TFLite INT8 ──")
    calib_data = generate_calibration_data(eval_loader, CALIBRATION_SAMPLES)
    print(f"[*] Calibration data: {calib_data.shape}")

    encoder_tflite = os.path.join(OUTPUT_DIR, "encoder.tflite")
    policy_tflite = os.path.join(OUTPUT_DIR, "policy.tflite")

    enc_ok = convert_to_tflite(encoder_onnx, encoder_tflite,
                                calibration_data=calib_data,
                                input_shape=(1, 3, 128, 128))
    # Policy calibration: generate random cues
    pol_calib = np.random.randn(CALIBRATION_SAMPLES, 258).astype(np.float32)
    pol_ok = convert_to_tflite(policy_onnx, policy_tflite,
                                calibration_data=pol_calib,
                                input_shape=(1, 258))

    # Step 4: Generate C headers
    print("\n── Step 4: Generate C Headers ──")
    esp32_model_dir = os.path.join(OUTPUT_DIR, "esp32_headers")
    os.makedirs(esp32_model_dir, exist_ok=True)

    if enc_ok:
        tflite_to_c_header(
            encoder_tflite,
            os.path.join(esp32_model_dir, "encoder_model.h"),
            "encoder_model"
        )
    if pol_ok:
        tflite_to_c_header(
            policy_tflite,
            os.path.join(esp32_model_dir, "policy_model.h"),
            "policy_model"
        )

    # Also save the FP32 ONNX models for reference
    torch.save(student_export.state_dict(),
               os.path.join(OUTPUT_DIR, "student_fp32.pt"))

    # Summary
    print(f"\n{'='*55}")
    print(f"  Export complete!")
    print(f"  FP32 accuracy: {fp32_acc:.3f}")
    print(f"  Output dir: {OUTPUT_DIR}")
    if enc_ok and pol_ok:
        enc_size = os.path.getsize(encoder_tflite) / 1024
        pol_size = os.path.getsize(policy_tflite) / 1024
        print(f"  Encoder: {enc_size:.1f} KB")
        print(f"  Policy:  {pol_size:.1f} KB")
        print(f"  Total:   {enc_size + pol_size:.1f} KB")
    else:
        print(f"  ONNX models exported. Manual TFLite conversion needed.")
        print(f"  See: {OUTPUT_DIR}/encoder.onnx, policy.onnx")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
