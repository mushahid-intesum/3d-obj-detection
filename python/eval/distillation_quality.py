
import os
import sys
import json
from collections import OrderedDict

import torch
import torch.nn.functional as F
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models import TeacherModel
from student_model import StudentModel


DATASET_DIR         = "./data/offline_dataset"
TEACHER_CKPT        = "./checkpoints/teacher/best.pt"
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"

NUM_ACTIONS         = 4
TEMPERATURE         = 4.0
BATCH_SIZE          = 512
PRINT_LATEX         = True

STUDENT_CKPT        = "./checkpoints/student/best.pt"

ABLATION_CKPTS      = {
}

QAT_CKPT            = None

SAVE_JSON           = None


ACTION_NAMES = ["FWD", "RIGHT", "LEFT", "STOP"]


def load_teacher(ckpt_path):
    teacher = TeacherModel(feat_dim=512, cue_dim=512,
                            num_actions=NUM_ACTIONS).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    teacher.load_state_dict(ckpt["model_state"])
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False
    step = ckpt.get("step", 0)
    print(f"[*] Teacher loaded from step {step}")
    return teacher


def load_student(ckpt_path):
    student = StudentModel(num_actions=NUM_ACTIONS).to(DEVICE)
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
    student.load_state_dict(ckpt["model_state"])
    student.eval()
    return student


def load_eval_batch():
    try:
        from dataset import get_dataloader
        loader = get_dataloader(DATASET_DIR, batch_size=BATCH_SIZE,
                                shuffle=True, num_workers=0)
        batch = next(iter(loader))
        print(f"[*] Loaded real data: {batch['obs'].shape[0]} samples")
        return batch
    except Exception as e:
        print(f"[!] Dataset unavailable ({e}), using synthetic data")
        return {
            "obs": torch.randn(BATCH_SIZE, 3, 96, 96),
            "goal": torch.randn(BATCH_SIZE, 3, 96, 96),
            "obs_proximity": torch.rand(BATCH_SIZE),
        }


@torch.no_grad()
def evaluate_student(teacher, student, batch):
    obs = batch["obs"].to(DEVICE)
    goal = batch["goal"].to(DEVICE)
    prox = batch["obs_proximity"].to(DEVICE)

    teacher_cue = teacher.encode_and_correlate(obs, goal)
    teacher_logits = teacher.policy(teacher_cue, prox)
    teacher_actions = teacher_logits.argmax(dim=-1)

    student_corr_cue = student.get_cue(obs, goal)
    student_logits = student.forward_with_cue(student_corr_cue, prox)
    student_actions = student_logits.argmax(dim=-1)

    student_cue_proj = student.project_cue(student_corr_cue)

    B = obs.shape[0]
    metrics = OrderedDict()

    agree = (teacher_actions == student_actions).float()
    metrics["top1_agreement"] = agree.mean().item()

    per_action = {}
    for a in range(NUM_ACTIONS):
        mask = teacher_actions == a
        if mask.sum() > 0:
            per_action[ACTION_NAMES[a]] = agree[mask].mean().item()
        else:
            per_action[ACTION_NAMES[a]] = float("nan")
    metrics["per_action_agreement"] = per_action

    teacher_soft = F.softmax(teacher_logits / TEMPERATURE, dim=-1)
    student_log_soft = F.log_softmax(student_logits / TEMPERATURE, dim=-1)
    kl_t = F.kl_div(student_log_soft, teacher_soft, reduction="batchmean")
    metrics["kl_divergence_T4"] = kl_t.item()

    teacher_soft_1 = F.softmax(teacher_logits, dim=-1)
    student_log_soft_1 = F.log_softmax(student_logits, dim=-1)
    kl_1 = F.kl_div(student_log_soft_1, teacher_soft_1, reduction="batchmean")
    metrics["kl_divergence_T1"] = kl_1.item()

    corr_mse = F.mse_loss(student_cue_proj, teacher_cue)
    metrics["correlation_mse"] = corr_mse.item()

    t_hist = torch.bincount(teacher_actions, minlength=NUM_ACTIONS).float() / B
    s_hist = torch.bincount(student_actions, minlength=NUM_ACTIONS).float() / B
    metrics["teacher_action_dist"] = {
        ACTION_NAMES[i]: f"{t_hist[i].item():.3f}" for i in range(NUM_ACTIONS)
    }
    metrics["student_action_dist"] = {
        ACTION_NAMES[i]: f"{s_hist[i].item():.3f}" for i in range(NUM_ACTIONS)
    }

    return metrics


def print_metrics(metrics, label="Student"):
    print(f"\n{'=' * 60}")
    print(f"  DISTILLATION QUALITY: {label}")
    print(f"{'=' * 60}")

    print(f"  Top-1 action agreement:   {metrics['top1_agreement']:.1%}")
    print(f"  KL divergence (T=4):      {metrics['kl_divergence_T4']:.4f}")
    print(f"  KL divergence (T=1):      {metrics['kl_divergence_T1']:.4f}")
    print(f"  Correlation MSE:          {metrics['correlation_mse']:.4f}")

    print(f"\n  Per-action agreement:")
    for action, acc in metrics["per_action_agreement"].items():
        bar = "█" * int(acc * 30) if not np.isnan(acc) else "N/A"
        val = f"{acc:.1%}" if not np.isnan(acc) else "N/A"
        print(f"    {action:<8} {val:>8}  {bar}")

    print(f"\n  Action distributions:")
    print(f"    {'Action':<8} {'Teacher':>10} {'Student':>10}")
    print(f"    {'-' * 30}")
    for action in ACTION_NAMES:
        t = metrics["teacher_action_dist"][action]
        s = metrics["student_action_dist"][action]
        print(f"    {action:<8} {t:>10} {s:>10}")

    print(f"{'=' * 60}")


def print_comparison_table(all_results):
    print(f"\n{'=' * 72}")
    print(f"  DISTILLATION COMPARISON")
    print(f"{'=' * 72}")
    print(f"  {'Config':<20} {'Agreement':>12} {'KL (T=1)':>12} {'Corr MSE':>12}")
    print(f"  {'-' * 60}")
    for label, metrics in all_results.items():
        agree = f"{metrics['top1_agreement']:.1%}"
        kl = f"{metrics['kl_divergence_T1']:.4f}"
        mse = f"{metrics['correlation_mse']:.4f}"
        print(f"  {label:<20} {agree:>12} {kl:>12} {mse:>12}")
    print(f"{'=' * 72}")


def print_latex_table(all_results):
    print("\n% ─── LaTeX Table ───")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\caption{Distillation quality metrics. Agreement measures top-1 action")
    print(r"match with the teacher. KL divergence is measured at $T{=}1$.}")
    print(r"\label{tab:distillation}")
    print(r"\begin{tabular}{lccc}")
    print(r"\toprule")
    print(r"\textbf{Configuration} & \textbf{Agreement (\%)} & \textbf{KL Div} & \textbf{Corr MSE} \\")
    print(r"\midrule")
    for label, metrics in all_results.items():
        agree = f"{metrics['top1_agreement'] * 100:.1f}"
        kl = f"{metrics['kl_divergence_T1']:.4f}"
        mse = f"{metrics['correlation_mse']:.4f}"
        print(f"{label} & {agree} & {kl} & {mse} " + r"\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


if __name__ == "__main__":
    teacher = load_teacher(TEACHER_CKPT)
    batch = load_eval_batch()

    all_results = OrderedDict()

    if os.path.exists(STUDENT_CKPT):
        print(f"\nLoading student from {STUDENT_CKPT}...")
        student = load_student(STUDENT_CKPT)
        metrics = evaluate_student(teacher, student, batch)
        print_metrics(metrics, label=os.path.basename(STUDENT_CKPT))
        all_results["Student (FP32)"] = metrics

    if QAT_CKPT and os.path.exists(QAT_CKPT):
        print(f"\nLoading QAT student from {QAT_CKPT}...")
        qat_student = load_student(QAT_CKPT)
        qat_metrics = evaluate_student(teacher, qat_student, batch)
        print_metrics(qat_metrics, label="QAT")
        all_results["Student (QAT)"] = qat_metrics

    for label, path in ABLATION_CKPTS.items():
        if os.path.exists(path):
            print(f"\nLoading {label} from {path}...")
            variant = load_student(path)
            variant_metrics = evaluate_student(teacher, variant, batch)
            all_results[label] = variant_metrics

    if not all_results:
        print("\n[!] No checkpoints found. Running with untrained student...")
        student = StudentModel(num_actions=NUM_ACTIONS).to(DEVICE)
        student.eval()
        metrics = evaluate_student(teacher, student, batch)
        print_metrics(metrics, label="Untrained (random)")
        all_results["Untrained"] = metrics

    if len(all_results) > 1:
        print_comparison_table(all_results)

    if PRINT_LATEX:
        print_latex_table(all_results)

    if SAVE_JSON:
        with open(SAVE_JSON, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[*] Results saved to {SAVE_JSON}")
