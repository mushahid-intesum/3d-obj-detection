#!/usr/bin/env python3
"""
compression_metrics.py — Tier 1: Parameter counts, FLOPs, model sizes.

Outputs tables comparing teacher vs student and a LaTeX-ready table.

Usage:
    python3 -m eval.compression_metrics
"""

import os
import sys
from collections import OrderedDict

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TeacherModel
from student_model import StudentModel


# ═══════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════
IMG_SIZE            = 96
PRINT_LATEX         = True

# Paths to exported TFLite models (set when available)
TFLITE_PATHS        = [
    # "./exports/encoder_int8.tflite",
    # "./exports/policy_int8.tflite",
]
# ═══════════════════════════════════════════


def count_params(model):
    """Count parameters per named child module."""
    counts = OrderedDict()
    for name, module in model.named_children():
        counts[name] = sum(p.numel() for p in module.parameters())
    counts["total"] = sum(p.numel() for p in model.parameters())
    return counts


def fmt(n):
    """Format parameter count with commas and M suffix."""
    return f"{n:>12,}  ({n / 1e6:.2f}M)"


def print_param_table(teacher_counts, student_counts, deploy_params):
    """Print side-by-side parameter comparison."""
    print("\n" + "=" * 72)
    print("  PARAMETER COUNT COMPARISON")
    print("=" * 72)
    print(f"  {'Component':<25} {'Teacher':>20} {'Student':>20}")
    print("-" * 72)

    all_keys = list(OrderedDict.fromkeys(
        list(teacher_counts.keys()) + list(student_counts.keys())
    ))

    for key in all_keys:
        if key == "total":
            continue
        t = teacher_counts.get(key, 0)
        s = student_counts.get(key, 0)
        label = key
        if key == "corr_projector":
            label = "corr_projector [discard]"
        print(f"  {label:<25} {fmt(t)} {fmt(s)}")

    print("-" * 72)
    t_total = teacher_counts["total"]
    s_total = student_counts["total"]
    print(f"  {'Total':<25} {fmt(t_total)} {fmt(s_total)}")
    print(f"  {'Deployable':<25} {fmt(t_total)} {fmt(deploy_params)}")
    print(f"  {'Compression ratio':<25} {'1.00x':>20} {f'{t_total / deploy_params:.2f}x':>20}")
    print("=" * 72)


# ═══════════════════════════════════════════
#  FLOPs / MACs estimation
# ═══════════════════════════════════════════

def estimate_encoder_macs(encoder, img_size):
    """Estimate total encoder MACs by hooking Conv2d and Linear layers."""
    mac_log = []

    def hook_fn(module, inp, out):
        if isinstance(module, nn.Conv2d):
            B, C_in, H, W = inp[0].shape
            K = module.kernel_size[0]
            C_out = module.out_channels
            stride = module.stride[0]
            groups = module.groups
            H_out = H // stride
            W_out = W // stride
            macs = K * K * (C_in // groups) * C_out * H_out * W_out
            mac_log.append(macs)
        elif isinstance(module, nn.Linear):
            mac_log.append(module.in_features * module.out_features)

    hooks = []
    for m in encoder.modules():
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            hooks.append(m.register_forward_hook(hook_fn))

    with torch.no_grad():
        encoder(torch.randn(1, 3, img_size, img_size))

    for h in hooks:
        h.remove()

    return sum(mac_log)


def estimate_correlation_macs_teacher(feat_h=12, feat_w=12, feat_dim=512):
    """MACs for teacher correlation (pairwise dots + conv fusion + projection)."""
    hw = feat_h * feat_w
    dot_macs = hw * hw * feat_dim
    conv_macs = 128 * 9 * 8 * 8 + 64 * 128 * 9 * 8 * 8
    linear_macs = 64 * 8 * 8 * 512
    return dot_macs + conv_macs + linear_macs


def estimate_correlation_macs_student(pool_size=4, feat_dim=1024):
    """MACs for student parameterless correlation (pool + norm + dots)."""
    positions = pool_size * pool_size
    norm_macs = 2 * positions * feat_dim * 2
    dot_macs = positions * positions * feat_dim
    lr_macs = 2 * pool_size * feat_dim
    return norm_macs + dot_macs + lr_macs


def estimate_policy_macs(hidden_dims, input_dim, num_actions=4):
    """MACs for MLP policy."""
    macs = 0
    prev = input_dim
    for h in hidden_dims:
        macs += prev * h
        prev = h
    macs += prev * num_actions
    return macs


def print_macs_table(teacher_macs, student_macs):
    """Print MACs comparison table."""
    print("\n" + "=" * 72)
    print("  MULTIPLY-ACCUMULATE OPERATIONS (MACs) PER INFERENCE STEP")
    print("=" * 72)
    print(f"  {'Stage':<30} {'Teacher':>16} {'Student':>16}")
    print("-" * 72)

    for stage in ["Encoder (1 image)", "Correlation", "Policy MLP"]:
        t = teacher_macs[stage]
        s = student_macs[stage]
        print(f"  {stage:<30} {t / 1e6:>13.1f}M {s / 1e6:>13.1f}M")

    print("-" * 72)
    t_total = teacher_macs["Total per step"]
    s_total = student_macs["Total per step"]
    ratio = t_total / s_total if s_total > 0 else float('inf')
    print(f"  {'Total per step':<30} {t_total / 1e6:>13.1f}M {s_total / 1e6:>13.1f}M")
    print(f"  {'Reduction':<30} {'1.0x':>14} {f'{ratio:.1f}x':>14}")
    print("=" * 72)


# ═══════════════════════════════════════════
#  TFLite model sizes
# ═══════════════════════════════════════════

def print_tflite_sizes(paths):
    """Print file sizes of TFLite models."""
    valid = [p for p in paths if os.path.exists(p)]
    if not valid:
        return
    print("\n" + "=" * 72)
    print("  TFLITE MODEL SIZES")
    print("=" * 72)
    for p in valid:
        size = os.path.getsize(p)
        print(f"  {os.path.basename(p)}: {size:,} bytes ({size / 1024:.1f} KB)")
    print("=" * 72)


# ═══════════════════════════════════════════
#  LaTeX output
# ═══════════════════════════════════════════

def print_latex_table(teacher_counts, student_counts, deploy_params,
                      teacher_macs, student_macs):
    """Print LaTeX table to stdout."""
    t_total = teacher_counts["total"]
    t_enc = teacher_counts.get("encoder", 0)
    t_corr = teacher_counts.get("correlation", 0)
    t_pol = teacher_counts.get("policy", 0)
    t_val = sum(teacher_counts.get(k, 0) for k in ["q1", "q2", "v"])

    s_enc = student_counts.get("encoder", 0)
    s_corr = student_counts.get("correlation", 0)
    s_pol = student_counts.get("policy", 0)

    t_mac = teacher_macs["Total per step"]
    s_mac = student_macs["Total per step"]

    print("\n% ─── LaTeX Table ───")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Compression metrics. Deployable parameters exclude the")
    print(r"distillation projector and IQL value heads.}")
    print(r"\label{tab:compression}")
    print(r"\begin{tabular}{lrr}")
    print(r"\toprule")
    print(r"\textbf{Metric} & \textbf{Teacher} & \textbf{Student} \\")
    print(r"\midrule")
    print(f"Encoder params      & {t_enc/1e6:.2f}M & {s_enc/1e6:.2f}M " + r"\\")
    print(f"Correlation params  & {t_corr/1e6:.2f}M & {s_corr} " + r"\\")
    print(f"Policy params       & {t_pol/1e6:.2f}M & {s_pol/1e6:.2f}M " + r"\\")
    print(f"Value heads (IQL)   & {t_val/1e6:.2f}M & --- " + r"\\")
    print(r"\midrule")
    print(f"Total params        & {t_total/1e6:.2f}M & {deploy_params/1e6:.2f}M " + r"\\")
    print(f"Compression ratio   & 1.0$\\times$ & {t_total/deploy_params:.1f}$\\times$ " + r"\\")
    print(r"\midrule")
    print(f"MACs / step         & {t_mac/1e6:.0f}M & {s_mac/1e6:.0f}M " + r"\\")
    print(f"MAC reduction       & 1.0$\\times$ & {t_mac/s_mac:.1f}$\\times$ " + r"\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ═══════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════

if __name__ == "__main__":
    teacher = TeacherModel()
    student = StudentModel()

    teacher_counts = count_params(teacher)
    student_counts = count_params(student)
    deploy_params = student.get_deploy_params()

    print_param_table(teacher_counts, student_counts, deploy_params)

    # MACs
    t_enc_macs = estimate_encoder_macs(teacher.encoder, IMG_SIZE)
    t_corr_macs = estimate_correlation_macs_teacher()
    t_pol_macs = estimate_policy_macs([256, 128], input_dim=513)

    s_enc_macs = estimate_encoder_macs(student.encoder, IMG_SIZE)
    s_corr_macs = estimate_correlation_macs_student()
    s_pol_macs = estimate_policy_macs([512, 256, 128], input_dim=259)

    teacher_macs = {
        "Encoder (1 image)": t_enc_macs,
        "Correlation": t_corr_macs,
        "Policy MLP": t_pol_macs,
        "Total per step": t_enc_macs + t_corr_macs + t_pol_macs,
    }
    student_macs = {
        "Encoder (1 image)": s_enc_macs,
        "Correlation": s_corr_macs,
        "Policy MLP": s_pol_macs,
        "Total per step": s_enc_macs + s_corr_macs + s_pol_macs,
    }

    print_macs_table(teacher_macs, student_macs)

    # Architecture comparison
    print("\n" + "=" * 72)
    print("  ARCHITECTURE COMPARISON")
    print("=" * 72)
    rows = [
        ("Encoder backbone",    "ResNet (standard conv)", "DSConv hierarchy"),
        ("Encoder output",      "12x12x512",              "6x6x1024"),
        ("Correlation",         "Learned pyramid+conv",   "Parameterless dot-product"),
        ("Correlation out dim", "512",                    "258"),
        ("Policy input dim",    "513 (512+prox)",         "259 (258+prox)"),
        ("Policy MLP",          "513->256->128->4",       "259->512->256->128->4"),
        ("Value heads",         "Twin Q + V (IQL)",       "None (distilled)"),
    ]
    print(f"  {'Component':<25} {'Teacher':<28} {'Student':<28}")
    print("-" * 72)
    for component, teacher_val, student_val in rows:
        print(f"  {component:<25} {teacher_val:<28} {student_val:<28}")
    print("=" * 72)

    # TFLite sizes
    print_tflite_sizes(TFLITE_PATHS)

    # LaTeX
    if PRINT_LATEX:
        print_latex_table(teacher_counts, student_counts, deploy_params,
                          teacher_macs, student_macs)
