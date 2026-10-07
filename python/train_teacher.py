"""
train_teacher.py — Phase 4: Train the teacher model with IQL offline RL.

IQL (Implicit Q-Learning) trains a policy from offline data without
querying out-of-distribution actions. Three losses:
  - L_V: Expectile regression (pushes V toward upper quantile of Q)
  - L_Q: Standard Bellman TD error
  - L_π: Advantage-weighted regression (upweights good actions)
"""

import os
import time
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models import TeacherModel
from dataset import get_dataloader

# ═══════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════
DATASET_DIR     = "./data/offline_dataset"
CHECKPOINT_DIR  = "./checkpoints/teacher"
DEVICE          = "cuda" if torch.cuda.is_available() else "cpu"

# Model
FEAT_DIM        = 128       # encoder feature channels
CUE_DIM         = 256       # correlation cue dimension
NUM_ACTIONS     = 4         # forward, left, right, stop

# IQL hyperparameters
IQL_TAU         = 0.7       # expectile for V loss (>0.5 → optimistic)
IQL_BETA        = 3.0       # temperature for advantage weighting
DISCOUNT        = 0.99      # reward discount factor

# Training
LR              = 3e-4
BATCH_SIZE      = 256
TOTAL_STEPS     = 200_000
EVAL_EVERY      = 5_000
SAVE_EVERY      = 20_000
TARGET_UPDATE   = 0.005     # polyak averaging coefficient
NUM_WORKERS     = 4
SEED            = 42
# ═══════════════════════════════════════════


def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def soft_update(target, source, tau):
    """Polyak averaging: target = (1-tau)*target + tau*source."""
    for tp, sp in zip(target.parameters(), source.parameters()):
        tp.data.copy_((1 - tau) * tp.data + tau * sp.data)


def expectile_loss(diff, tau):
    """Asymmetric squared loss for expectile regression."""
    weight = torch.where(diff > 0, tau, 1.0 - tau)
    return (weight * diff.pow(2)).mean()


class IQLTrainer:
    """Implicit Q-Learning trainer for the teacher model."""

    def __init__(self):
        self.model = TeacherModel(
            feat_dim=FEAT_DIM, cue_dim=CUE_DIM, num_actions=NUM_ACTIONS
        ).to(DEVICE)

        # Target networks (for stable bootstrapping)
        self.q1_target = TeacherModel(
            feat_dim=FEAT_DIM, cue_dim=CUE_DIM, num_actions=NUM_ACTIONS
        ).q1.to(DEVICE)
        self.q2_target = TeacherModel(
            feat_dim=FEAT_DIM, cue_dim=CUE_DIM, num_actions=NUM_ACTIONS
        ).q2.to(DEVICE)

        # Copy initial weights
        self.q1_target.load_state_dict(self.model.q1.state_dict())
        self.q2_target.load_state_dict(self.model.q2.state_dict())

        # Separate optimizers for each component
        self.opt_encoder = torch.optim.Adam(
            self.model.encoder.parameters(), lr=LR)
        self.opt_corr = torch.optim.Adam(
            self.model.correlation.parameters(), lr=LR)
        self.opt_q1 = torch.optim.Adam(self.model.q1.parameters(), lr=LR)
        self.opt_q2 = torch.optim.Adam(self.model.q2.parameters(), lr=LR)
        self.opt_v = torch.optim.Adam(self.model.v.parameters(), lr=LR)
        self.opt_pi = torch.optim.Adam(self.model.policy.parameters(), lr=LR)

        self.all_opts = [
            self.opt_encoder, self.opt_corr,
            self.opt_q1, self.opt_q2, self.opt_v, self.opt_pi
        ]

    def compute_cue(self, obs, goal):
        """Encode obs and goal, compute correlation cue."""
        return self.model.encode_and_correlate(obs, goal)

    def train_step(self, batch):
        """One IQL gradient step. Returns dict of loss values."""
        obs = batch["obs"].to(DEVICE)
        goal = batch["goal"].to(DEVICE)
        next_obs = batch["next_obs"].to(DEVICE)
        actions = batch["action"].to(DEVICE)
        rewards = batch["reward"].to(DEVICE)
        dones = batch["done"].to(DEVICE)

        # ── Compute correlation cues ──
        cue = self.compute_cue(obs, goal)
        with torch.no_grad():
            cue_next = self.compute_cue(next_obs, goal)

        # ── Q targets (from target networks) ──
        with torch.no_grad():
            q1_target = self.q1_target(cue, actions)
            q2_target = self.q2_target(cue, actions)
            q_target = torch.min(q1_target, q2_target)

        # ── V loss: expectile regression ──
        v_pred = self.model.v(cue)
        v_diff = q_target - v_pred
        loss_v = expectile_loss(v_diff, IQL_TAU)

        # ── Q losses: Bellman TD error ──
        with torch.no_grad():
            v_next = self.model.v(cue_next)
            td_target = rewards + DISCOUNT * (1.0 - dones) * v_next

        q1_pred = self.model.q1(cue, actions)
        q2_pred = self.model.q2(cue, actions)
        loss_q1 = F.mse_loss(q1_pred, td_target)
        loss_q2 = F.mse_loss(q2_pred, td_target)

        # ── Policy loss: advantage-weighted regression ──
        with torch.no_grad():
            advantage = q_target - v_pred
            # Clamp for numerical stability
            weights = torch.exp(IQL_BETA * advantage).clamp(max=100.0)

        log_probs = F.log_softmax(self.model.policy(cue), dim=-1)
        log_prob_actions = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
        loss_pi = -(weights * log_prob_actions).mean()

        # ── Combined backward pass ──
        total_loss = loss_v + loss_q1 + loss_q2 + loss_pi

        for opt in self.all_opts:
            opt.zero_grad()
        total_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        for opt in self.all_opts:
            opt.step()

        # ── Soft-update target networks ──
        soft_update(self.q1_target, self.model.q1, TARGET_UPDATE)
        soft_update(self.q2_target, self.model.q2, TARGET_UPDATE)

        return {
            "loss_v": loss_v.item(),
            "loss_q1": loss_q1.item(),
            "loss_q2": loss_q2.item(),
            "loss_pi": loss_pi.item(),
            "total": total_loss.item(),
            "v_mean": v_pred.mean().item(),
            "q_mean": q_target.mean().item(),
            "adv_mean": advantage.mean().item(),
        }

    @torch.no_grad()
    def evaluate(self, loader, max_batches=50):
        """Evaluate action prediction accuracy on dataset."""
        self.model.eval()
        correct = 0
        total = 0

        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            obs = batch["obs"].to(DEVICE)
            goal = batch["goal"].to(DEVICE)
            actions = batch["action"].to(DEVICE)

            logits = self.model(obs, goal)
            preds = logits.argmax(dim=-1)
            correct += (preds == actions).sum().item()
            total += len(actions)

        self.model.train()
        return correct / total if total > 0 else 0.0

    def save(self, path, step, metrics):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "step": step,
            "model_state": self.model.state_dict(),
            "q1_target_state": self.q1_target.state_dict(),
            "q2_target_state": self.q2_target.state_dict(),
            "metrics": metrics,
        }, path)

    def load(self, path):
        """Load model checkpoint."""
        ckpt = torch.load(path, map_location=DEVICE)
        self.model.load_state_dict(ckpt["model_state"])
        self.q1_target.load_state_dict(ckpt["q1_target_state"])
        self.q2_target.load_state_dict(ckpt["q2_target_state"])
        return ckpt["step"], ckpt.get("metrics", {})


def main():
    print("=" * 55)
    print("  Phase 4: Teacher Training (IQL)")
    print(f"  Device: {DEVICE}")
    print("=" * 55)

    set_seed(SEED)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    # Data
    print(f"\n[*] Loading dataset from: {DATASET_DIR}")
    train_loader = get_dataloader(
        DATASET_DIR, batch_size=BATCH_SIZE,
        augment=True, num_workers=NUM_WORKERS
    )
    eval_loader = get_dataloader(
        DATASET_DIR, batch_size=BATCH_SIZE,
        augment=False, num_workers=NUM_WORKERS
    )

    # Model
    trainer = IQLTrainer()
    total_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"[*] Teacher model: {total_params:,} parameters")

    # Training loop
    print(f"\n[*] Training for {TOTAL_STEPS} steps, batch_size={BATCH_SIZE}")
    print(f"    IQL: tau={IQL_TAU}, beta={IQL_BETA}, gamma={DISCOUNT}")
    print(f"    LR={LR}, target_update={TARGET_UPDATE}")
    print()

    step = 0
    best_acc = 0.0
    log_history = []
    data_iter = iter(train_loader)

    t_start = time.time()

    while step < TOTAL_STEPS:
        # Get batch (cycle through dataset)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(train_loader)
            batch = next(data_iter)

        metrics = trainer.train_step(batch)
        step += 1

        # Logging
        if step % 500 == 0:
            elapsed = time.time() - t_start
            sps = step / elapsed
            print(f"[{step:>7}/{TOTAL_STEPS}] "
                  f"L_total={metrics['total']:.4f} "
                  f"L_v={metrics['loss_v']:.4f} "
                  f"L_q={metrics['loss_q1']:.4f} "
                  f"L_pi={metrics['loss_pi']:.4f} "
                  f"V={metrics['v_mean']:.2f} "
                  f"Q={metrics['q_mean']:.2f} "
                  f"| {sps:.0f} steps/s")

        # Evaluation
        if step % EVAL_EVERY == 0:
            acc = trainer.evaluate(eval_loader)
            print(f"  >> Eval action accuracy: {acc:.3f}")

            log_history.append({
                "step": step, "accuracy": acc, **metrics
            })

            if acc > best_acc:
                best_acc = acc
                trainer.save(
                    os.path.join(CHECKPOINT_DIR, "best.pt"),
                    step, {"accuracy": acc}
                )
                print(f"  >> New best! Saved to best.pt")

        # Periodic checkpoint
        if step % SAVE_EVERY == 0:
            trainer.save(
                os.path.join(CHECKPOINT_DIR, f"step_{step}.pt"),
                step, metrics
            )

    # Final save
    trainer.save(
        os.path.join(CHECKPOINT_DIR, "final.pt"),
        step, metrics
    )

    # Save training log
    with open(os.path.join(CHECKPOINT_DIR, "train_log.json"), "w") as f:
        json.dump(log_history, f, indent=2)

    elapsed = time.time() - t_start
    print(f"\n{'='*55}")
    print(f"  Training complete!")
    print(f"  Steps: {step}, Time: {elapsed/3600:.1f}h")
    print(f"  Best accuracy: {best_acc:.3f}")
    print(f"  Checkpoints: {CHECKPOINT_DIR}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
