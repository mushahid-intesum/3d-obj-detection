"""
distill.py — Phase 5: Knowledge distillation from teacher to student.

Dual-level distillation:
  1. Action distribution matching (KL divergence with temperature)
  2. Correlation feature matching (MSE between projected cues)
  3. Hard label loss (cross-entropy with teacher's argmax)
"""

import os
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import TeacherModel
from student_model import StudentModel
from dataset import get_dataloader

# ═══════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════
DATASET_DIR         = "./data/offline_dataset"
TEACHER_CKPT        = "./checkpoints/teacher/best.pt"
CHECKPOINT_DIR      = "./checkpoints/student"
DEVICE              = "cuda" if torch.cuda.is_available() else "cpu"

# Teacher config (must match training)
TEACHER_FEAT_DIM    = 512
TEACHER_CUE_DIM     = 512
NUM_ACTIONS         = 4

# Distillation hyperparameters
TEMPERATURE         = 4.0       # softmax temperature for KD
ALPHA_ACTION        = 1.0       # weight for KL divergence loss
ALPHA_CORR          = 0.5       # weight for correlation matching loss
ALPHA_HARD          = 0.5       # weight for hard label CE loss

# Training
LR                  = 1e-3
BATCH_SIZE          = 256
TOTAL_STEPS         = 100_000
EVAL_EVERY          = 2_000
SAVE_EVERY          = 20_000
NUM_WORKERS         = 4
SEED                = 42
# ═══════════════════════════════════════════


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_teacher(ckpt_path):
    """Load the trained teacher model (frozen)."""
    teacher = TeacherModel(
        feat_dim=TEACHER_FEAT_DIM, cue_dim=TEACHER_CUE_DIM,
        num_actions=NUM_ACTIONS
    ).to(DEVICE)

    ckpt = torch.load(ckpt_path, map_location=DEVICE)
    teacher.load_state_dict(ckpt["model_state"])
    teacher.eval()

    # Freeze all parameters
    for p in teacher.parameters():
        p.requires_grad = False

    step = ckpt.get("step", 0)
    metrics = ckpt.get("metrics", {})
    print(f"[*] Teacher loaded from step {step}, metrics: {metrics}")
    return teacher


class DistillationTrainer:
    """Dual-level knowledge distillation trainer."""

    def __init__(self, teacher):
        self.teacher = teacher
        self.student = StudentModel(num_actions=NUM_ACTIONS).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.student.parameters(), lr=LR)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=TOTAL_STEPS, eta_min=LR * 0.01
        )

    def distill_step(self, batch):
        """One distillation gradient step."""
        obs = batch["obs"].to(DEVICE)
        goal = batch["goal"].to(DEVICE)

        # ── Teacher forward (no grad) ──
        with torch.no_grad():
            teacher_cue = self.teacher.encode_and_correlate(obs, goal)
            teacher_logits = self.teacher.policy(teacher_cue)
            teacher_actions = teacher_logits.argmax(dim=-1)

        # ── Student forward ──
        student_cue = self.student.get_cue(obs, goal)
        student_logits = self.student.policy(student_cue)

        # ── Loss 1: Action distribution matching (KL divergence) ──
        teacher_soft = F.log_softmax(teacher_logits / TEMPERATURE, dim=-1)
        student_soft = F.log_softmax(student_logits / TEMPERATURE, dim=-1)
        loss_action = F.kl_div(
            student_soft, teacher_soft.exp(),
            reduction='batchmean'
        ) * (TEMPERATURE ** 2)

        # ── Loss 2: Correlation feature matching (MSE) ──
        student_cue_proj = self.student.project_cue(student_cue)
        loss_corr = F.mse_loss(student_cue_proj, teacher_cue)

        # ── Loss 3: Hard label CE ──
        loss_hard = F.cross_entropy(student_logits, teacher_actions)

        # ── Combined loss ──
        total_loss = (ALPHA_ACTION * loss_action +
                      ALPHA_CORR * loss_corr +
                      ALPHA_HARD * loss_hard)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        # ── Compute agreement ──
        student_actions = student_logits.argmax(dim=-1)
        agreement = (student_actions == teacher_actions).float().mean().item()

        return {
            "loss_action": loss_action.item(),
            "loss_corr": loss_corr.item(),
            "loss_hard": loss_hard.item(),
            "total": total_loss.item(),
            "agreement": agreement,
            "lr": self.scheduler.get_last_lr()[0],
        }

    @torch.no_grad()
    def evaluate(self, loader, max_batches=50):
        """Evaluate student-teacher agreement and per-action accuracy."""
        self.student.eval()
        correct = 0
        total = 0
        per_action = {i: {"correct": 0, "total": 0} for i in range(NUM_ACTIONS)}

        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            obs = batch["obs"].to(DEVICE)
            goal = batch["goal"].to(DEVICE)

            teacher_logits = self.teacher(obs, goal)
            student_logits = self.student(obs, goal)

            t_actions = teacher_logits.argmax(dim=-1)
            s_actions = student_logits.argmax(dim=-1)

            matches = (s_actions == t_actions)
            correct += matches.sum().item()
            total += len(t_actions)

            for a in range(NUM_ACTIONS):
                mask = (t_actions == a)
                per_action[a]["total"] += mask.sum().item()
                per_action[a]["correct"] += (matches & mask).sum().item()

        self.student.train()

        overall_acc = correct / total if total > 0 else 0.0
        per_action_acc = {}
        for a in range(NUM_ACTIONS):
            t = per_action[a]["total"]
            c = per_action[a]["correct"]
            per_action_acc[a] = c / t if t > 0 else 0.0

        return overall_acc, per_action_acc

    def save(self, path, step, metrics):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Save only deploy-relevant weights (exclude corr_projector)
        deploy_state = {
            k: v for k, v in self.student.state_dict().items()
            if "corr_projector" not in k
        }
        torch.save({
            "step": step,
            "model_state": self.student.state_dict(),
            "deploy_state": deploy_state,
            "metrics": metrics,
        }, path)


def main():
    print("=" * 55)
    print("  Phase 5: Knowledge Distillation")
    print(f"  Device: {DEVICE}")
    print("=" * 55)

    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Load teacher
    print(f"\n[*] Loading teacher from: {TEACHER_CKPT}")
    teacher = load_teacher(TEACHER_CKPT)

    # Data
    print(f"[*] Loading dataset from: {DATASET_DIR}")
    train_loader = get_dataloader(
        DATASET_DIR, batch_size=BATCH_SIZE,
        augment=True, num_workers=NUM_WORKERS
    )
    eval_loader = get_dataloader(
        DATASET_DIR, batch_size=BATCH_SIZE,
        augment=False, num_workers=NUM_WORKERS
    )

    # Trainer
    trainer = DistillationTrainer(teacher)
    deploy_params = trainer.student.get_deploy_params()
    print(f"[*] Student model: {deploy_params:,} deploy params "
          f"(~{deploy_params // 1024} KB INT8)")

    # Training
    action_names = {0: "FWD", 1: "LFT", 2: "RGT", 3: "STP"}
    print(f"\n[*] Distilling for {TOTAL_STEPS} steps")
    print(f"    T={TEMPERATURE}, alpha_action={ALPHA_ACTION}, "
          f"alpha_corr={ALPHA_CORR}, alpha_hard={ALPHA_HARD}")
    print()

    step = 0
    best_agreement = 0.0
    log_history = []
    data_iter = iter(train_loader)
    t_start = time.time()

    while step < TOTAL_STEPS:
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        metrics = trainer.distill_step(batch)
        step += 1

        if step % 500 == 0:
            elapsed = time.time() - t_start
            sps = step / elapsed
            print(f"[{step:>7}/{TOTAL_STEPS}] "
                  f"L={metrics['total']:.4f} "
                  f"L_kl={metrics['loss_action']:.4f} "
                  f"L_corr={metrics['loss_corr']:.4f} "
                  f"L_hard={metrics['loss_hard']:.4f} "
                  f"agr={metrics['agreement']:.3f} "
                  f"lr={metrics['lr']:.1e} "
                  f"| {sps:.0f} s/s")

        if step % EVAL_EVERY == 0:
            acc, per_action_acc = trainer.evaluate(eval_loader)
            pa_str = " ".join(
                f"{action_names[a]}:{v:.2f}" for a, v in per_action_acc.items()
            )
            print(f"  >> Agreement: {acc:.3f}  [{pa_str}]")

            log_history.append({"step": step, "agreement": acc, **metrics})

            if acc > best_agreement:
                best_agreement = acc
                trainer.save(
                    os.path.join(CHECKPOINT_DIR, "best.pt"),
                    step, {"agreement": acc, "per_action": per_action_acc}
                )
                print(f"  >> New best! Saved.")

        if step % SAVE_EVERY == 0:
            trainer.save(
                os.path.join(CHECKPOINT_DIR, f"step_{step}.pt"), step, metrics)

    trainer.save(os.path.join(CHECKPOINT_DIR, "final.pt"), step, metrics)

    with open(os.path.join(CHECKPOINT_DIR, "distill_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'='*55}")
    print(f"  Distillation complete!")
    print(f"  Steps: {step}, Time: {elapsed/60:.1f} min")
    print(f"  Best agreement: {best_agreement:.3f}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
