
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

DATASET_DIR         = "./data/offline_dataset"
STUDENT_CKPT        = "./checkpoints/student/best.pt"
TEACHER_CKPT        = "./checkpoints/teacher/best.pt"
OUTPUT_DIR          = "./export"
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"

TEACHER_FEAT_DIM    = 512
TEACHER_CUE_DIM     = 512
NUM_ACTIONS         = 4

QAT_STEPS           = 10_000
QAT_LR              = 1e-4
QAT_BATCH_SIZE      = 256
QAT_TEMPERATURE     = 4.0
NUM_WORKERS         = 4

CALIBRATION_SAMPLES = 200
SEED                = 42


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def load_student(ckpt_path):
    student = StudentModel(num_actions=NUM_ACTIONS)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    student.load_state_dict(ckpt["model_state"])
    print(f"[*] Student loaded from: {ckpt_path}")
    return student


def load_teacher(ckpt_path):
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


def run_qat(student, teacher, loader):
    print("\n[*] Running Quantization-Aware Training...")

    student = student.to(DEVICE)
    student.train()

    student.qconfig = torch.ao.quantization.get_default_qat_qconfig("x86")
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
        obs_proximity = batch["obs_proximity"].to(DEVICE)

        with torch.no_grad():
            teacher_logits = teacher(obs, goal, obs_proximity)
            teacher_actions = teacher_logits.argmax(dim=-1)

        student_logits = student(obs, goal, obs_proximity)

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

    student.eval()
    student_quantized = torch.ao.quantization.convert(student, inplace=False)
    print("[OK] QAT complete")
    return student, student_quantized


class EncoderWrapper(nn.Module):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def forward(self, x):
        return self.encoder(x)


class PolicyWrapper(nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, cue):
        return self.policy(cue)


def export_onnx(student_fp32, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    student_fp32.eval().cpu()

    from student_model import CUE_DIM

    encoder_wrapper = EncoderWrapper(student_fp32.encoder)
    dummy_img = torch.randn(1, 3, 96, 96)
    encoder_path = os.path.join(output_dir, "encoder.onnx")
    torch.onnx.export(
        encoder_wrapper, dummy_img, encoder_path,
        input_names=["image"], output_names=["features"],
        opset_version=13,
        dynamic_axes={"image": {0: "batch"}, "features": {0: "batch"}},
    )
    print(f"[OK] Encoder ONNX: {encoder_path}")

    policy_wrapper = PolicyWrapper(student_fp32.policy)
    dummy_cue = torch.randn(1, CUE_DIM)
    policy_path = os.path.join(output_dir, "policy.onnx")
    torch.onnx.export(
        policy_wrapper, dummy_cue, policy_path,
        input_names=["cue_with_proximity"], output_names=["action_logits"],
        opset_version=13,
        dynamic_axes={"cue_with_proximity": {0: "batch"}, "action_logits": {0: "batch"}},
    )
    print(f"[OK] Policy ONNX: {policy_path} (input_dim={CUE_DIM})")

    return encoder_path, policy_path


def generate_calibration_data(loader, n_samples):
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
    try:
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        onnx_model = onnx.load(onnx_path)
        tf_rep = prepare(onnx_model)
        saved_model_dir = onnx_path.replace(".onnx", "_saved_model")
        tf_rep.export_graph(saved_model_dir)

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


def tflite_to_c_header(tflite_path, header_path, array_name):
    with open(tflite_path, "rb") as f:
        data = f.read()

    with open(header_path, "w") as f:
        f.write(f"/* Auto-generated from {os.path.basename(tflite_path)} */\n")
        f.write(f"/* Size: {len(data)} bytes ({len(data)/1024:.1f} KB) */\n")
        f.write(f"#ifndef {array_name.upper()}_H\n")
        f.write(f"#define {array_name.upper()}_H\n\n")
        f.write(f"#include <stdint.h>\n")
        f.write(f"#include <stdalign.h>\n\n")
        f.write(f"/* Enable model in inference.cpp */\n")
        f.write(f"#ifndef INFERENCE_HAS_MODEL\n")
        f.write(f"#define INFERENCE_HAS_MODEL 1\n")
        f.write(f"#endif\n\n")
        f.write(f"const unsigned int {array_name}_len = {len(data)};\n")
        f.write(f"alignas(16) const uint8_t {array_name}[] = {{\n")

        for i in range(0, len(data), 16):
            chunk = data[i:i+16]
            hex_vals = ", ".join(f"0x{b:02x}" for b in chunk)
            f.write(f"    {hex_vals},\n")

        f.write(f"}};\n\n")
        f.write(f"#endif /* {array_name.upper()}_H */\n")

    print(f"[OK] C header: {header_path} ({len(data)} bytes, {len(data)/1024:.1f} KB)")


@torch.no_grad()
def validate_fp32_vs_int8(student_fp32, loader, max_batches=50):
    student_fp32.eval().cpu()
    correct = 0
    total = 0

    for i, batch in enumerate(loader):
        if i >= max_batches:
            break
        obs = batch["obs"]
        goal = batch["goal"]
        actions = batch["action"]
        obs_proximity = batch["obs_proximity"]

        logits = student_fp32(obs, goal, obs_proximity)
        preds = logits.argmax(dim=-1)
        correct += (preds == actions).sum().item()
        total += len(actions)

    acc = correct / total if total > 0 else 0.0
    print(f"[*] FP32 student accuracy (vs dataset labels): {acc:.3f}")
    return acc


def main():
    print("=" * 55)
    print("  Phase 6: Quantization & Export")
    print("=" * 55)

    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    student = load_student(STUDENT_CKPT)
    teacher = load_teacher(TEACHER_CKPT)

    train_loader = get_dataloader(
        DATASET_DIR, batch_size=QAT_BATCH_SIZE,
        augment=True, num_workers=NUM_WORKERS
    )
    eval_loader = get_dataloader(
        DATASET_DIR, batch_size=QAT_BATCH_SIZE,
        augment=False, num_workers=NUM_WORKERS
    )

    print("\n── Step 0: FP32 Baseline ──")
    fp32_acc = validate_fp32_vs_int8(student, eval_loader)

    print("\n── Step 1: Quantization-Aware Training ──")
    student_qat, student_quantized = run_qat(student, teacher, train_loader)

    print("\n── Step 2: Export ONNX ──")
    student_qat.eval().cpu()
    student_export = StudentModel(num_actions=NUM_ACTIONS)
    clean_state = {}
    for k, v in student_qat.state_dict().items():
        if any(x in k for x in ["weight_fake_quant", "activation_post_process",
                                  "observer", "scale", "zero_point", "min_val",
                                  "max_val", "eps"]):
            continue
        clean_state[k] = v

    student_export.load_state_dict(clean_state, strict=False)
    encoder_onnx, policy_onnx = export_onnx(student_export, OUTPUT_DIR)

    print("\n── Step 3: Convert to TFLite INT8 ──")
    calib_data = generate_calibration_data(eval_loader, CALIBRATION_SAMPLES)
    print(f"[*] Calibration data: {calib_data.shape}")

    encoder_tflite = os.path.join(OUTPUT_DIR, "encoder.tflite")
    policy_tflite = os.path.join(OUTPUT_DIR, "policy.tflite")

    enc_ok = convert_to_tflite(encoder_onnx, encoder_tflite,
                                calibration_data=calib_data,
                                input_shape=(1, 3, 96, 96))
    from student_model import CUE_DIM, CORR_FLAT_DIM
    pol_calib = np.random.randn(CALIBRATION_SAMPLES, CUE_DIM).astype(np.float32)
    pol_ok = convert_to_tflite(policy_onnx, policy_tflite,
                                calibration_data=pol_calib,
                                input_shape=(1, CUE_DIM))

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

    torch.save(student_export.state_dict(),
               os.path.join(OUTPUT_DIR, "student_fp32.pt"))

    print(f"\n{'='*55}")
    print(f"  Export complete!")
    print(f"  FP32 accuracy: {fp32_acc:.3f}")
    print(f"  Output dir: {OUTPUT_DIR}")
    if enc_ok and pol_ok:
        enc_size = os.path.getsize(encoder_tflite) / 1024
        pol_size = os.path.getsize(policy_tflite) / 1024
        print(f"  Encoder: {enc_size:.1f} KB (input: 1×3×96×96 INT8)")
        print(f"  Policy:  {pol_size:.1f} KB (input: 1×{CUE_DIM} INT8)")
        print(f"  Total:   {enc_size + pol_size:.1f} KB")
        print(f"")
        print(f"  Tensor dimensions (verify against inference.cpp):")
        print(f"    FEAT_DIM={student_export.encoder.net[-1].pw.out_channels}")
        print(f"    CORR_FLAT_DIM={CORR_FLAT_DIM}  CUE_DIM={CUE_DIM}")
        print(f"")
        print(f"  Deployment:")
        print(f"    1. Copy {esp32_model_dir}/*.h → esp32/motor_board/main/include/")
        print(f"    2. Headers auto-set INFERENCE_HAS_MODEL=1")
        print(f"    3. idf.py build && idf.py flash")
    else:
        print(f"  ONNX models exported. Manual TFLite conversion needed.")
        print(f"  See: {OUTPUT_DIR}/encoder.onnx, policy.onnx")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
