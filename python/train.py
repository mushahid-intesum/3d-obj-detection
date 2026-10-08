#!/usr/bin/env python3
"""
train.py — Training loop for depth-aware ImageNav policy.

Trains the TeacherModel with:
  1. Action loss:   Cross-entropy on BFS shortest-path supervision
  2. Depth loss:    MSE on MiDaS pseudo-depth maps (auxiliary)
  3. Action weights: Class-balanced weighting
"""

import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models import TeacherModel
from dataset import get_dataloader, NUM_ACTIONS, ACTION_NAMES

# ═══════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════
DATA_ROOT = "data"                     # Root dir — auto-discovers all collections
EPOCHS            = 200
BATCH_SIZE        = 128
LR                = 3e-4
DEPTH_WEIGHT      = 0.1               # Weight for depth auxiliary loss
SAMPLES_PER_EPOCH = 10000
SAVE_DIR          = "checkpoints"
LOG_DIR           = "runs"
RESUME_PATH       = None               # Set to "checkpoints/latest.pt" to resume
DEVICE            = "cuda" if torch.cuda.is_available() else "cpu"
# ═══════════════════════════════════════════════


def compute_class_weights(loader, num_actions=NUM_ACTIONS, num_batches=20):
    """Compute inverse-frequency class weights from a few batches."""
    counts = torch.zeros(num_actions)
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        actions = batch["action"]
        for a in range(num_actions):
            counts[a] += (actions == a).sum().item()

    total = counts.sum()
    weights = total / (num_actions * counts.clamp(min=1))
    print(f"[Train] Action distribution: "
          f"{dict(zip(ACTION_NAMES, counts.long().tolist()))}")
    print(f"[Train] Class weights: "
          f"{dict(zip(ACTION_NAMES, weights.numpy().round(2).tolist()))}")
    return weights


def train_epoch(model, loader, optimizer, device, depth_weight,
                class_weights=None):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_action_loss = 0
    total_depth_loss = 0
    correct = 0
    total = 0

    ce_loss_fn = nn.CrossEntropyLoss(
        weight=class_weights.to(device) if class_weights is not None else None
    )

    for batch in loader:
        obs = batch["obs"].to(device)
        goal = batch["goal"].to(device)
        depth_gt = batch["depth"].to(device)
        action_gt = batch["action"].to(device)

        # Forward
        action_logits, depth_pred = model(obs, goal, return_depth=True)

        # Losses
        action_loss = ce_loss_fn(action_logits, action_gt)
        depth_loss = F.mse_loss(depth_pred, depth_gt)
        loss = action_loss + depth_weight * depth_loss

        # Backward
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        total_action_loss += action_loss.item()
        total_depth_loss += depth_loss.item()
        pred = action_logits.argmax(dim=-1)
        correct += (pred == action_gt).sum().item()
        total += action_gt.size(0)

    n = len(loader)
    return {
        "loss": total_loss / n,
        "action_loss": total_action_loss / n,
        "depth_loss": total_depth_loss / n,
        "accuracy": correct / total if total > 0 else 0,
    }


@torch.no_grad()
def eval_epoch(model, loader, device, depth_weight):
    """Evaluate for one epoch."""
    model.eval()
    total_loss = 0
    total_action_loss = 0
    total_depth_loss = 0
    correct = 0
    total = 0
    per_action_correct = torch.zeros(NUM_ACTIONS)
    per_action_total = torch.zeros(NUM_ACTIONS)

    for batch in loader:
        obs = batch["obs"].to(device)
        goal = batch["goal"].to(device)
        depth_gt = batch["depth"].to(device)
        action_gt = batch["action"].to(device)

        action_logits, depth_pred = model(obs, goal, return_depth=True)

        action_loss = F.cross_entropy(action_logits, action_gt)
        depth_loss = F.mse_loss(depth_pred, depth_gt)
        loss = action_loss + depth_weight * depth_loss

        total_loss += loss.item()
        total_action_loss += action_loss.item()
        total_depth_loss += depth_loss.item()

        pred = action_logits.argmax(dim=-1)
        correct += (pred == action_gt).sum().item()
        total += action_gt.size(0)

        for a in range(NUM_ACTIONS):
            mask = action_gt == a
            per_action_correct[a] += (pred[mask] == a).sum().item()
            per_action_total[a] += mask.sum().item()

    n = len(loader)
    per_action_acc = {}
    for a in range(NUM_ACTIONS):
        if per_action_total[a] > 0:
            per_action_acc[ACTION_NAMES[a]] = (
                per_action_correct[a] / per_action_total[a]
            ).item()

    return {
        "loss": total_loss / n,
        "action_loss": total_action_loss / n,
        "depth_loss": total_depth_loss / n,
        "accuracy": correct / total if total > 0 else 0,
        "per_action_acc": per_action_acc,
    }


def main():
    device = torch.device(DEVICE)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"{'═' * 50}")
    print(f"  Depth-Aware ImageNav Training")
    print(f"  Data:     {DATA_ROOT}")
    print(f"  Device:   {device}")
    print(f"  Epochs:   {EPOCHS}")
    print(f"  Batch:    {BATCH_SIZE}")
    print(f"  LR:       {LR}")
    print(f"  Depth w:  {DEPTH_WEIGHT}")
    print(f"{'═' * 50}\n")

    # ── Data ──
    train_loader = get_dataloader(
        DATA_ROOT,
        batch_size=BATCH_SIZE,
        samples_per_epoch=SAMPLES_PER_EPOCH,
        augment=True,
        num_workers=4,
    )
    val_loader = get_dataloader(
        DATA_ROOT,
        batch_size=BATCH_SIZE,
        samples_per_epoch=SAMPLES_PER_EPOCH // 5,
        augment=False,
        num_workers=2,
    )

    # ── Model ──
    model = TeacherModel(
        feat_dim=128,
        cue_dim=256,
        num_actions=NUM_ACTIONS,
        use_depth_head=True,
    ).to(device)

    params = model.count_params()
    print(f"\n[Model] Parameters:")
    for k, v in params.items():
        print(f"  {k:20s}: {v:>8,}")

    # ── Optimizer ──
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-6
    )

    # ── Class weights ──
    class_weights = compute_class_weights(train_loader)

    # ── Resume ──
    start_epoch = 0
    best_acc = 0.0
    if RESUME_PATH and os.path.exists(RESUME_PATH):
        ckpt = torch.load(RESUME_PATH, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0)
        print(f"[Resume] from epoch {start_epoch}, best_acc={best_acc:.3f}")

    # ── TensorBoard ──
    writer = SummaryWriter(LOG_DIR)

    # ── Training Loop ──
    print(f"\n{'Epoch':>5} | {'Loss':>7} | {'A.Loss':>7} | {'D.Loss':>7} | "
          f"{'Acc':>6} | {'V.Acc':>6} | {'LR':>8} | {'Time':>5}")
    print("─" * 70)

    for epoch in range(start_epoch, EPOCHS):
        t0 = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            depth_weight=DEPTH_WEIGHT,
            class_weights=class_weights,
        )

        # Eval (every 5 epochs)
        val_metrics = None
        if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
            val_metrics = eval_epoch(
                model, val_loader, device, depth_weight=DEPTH_WEIGHT,
            )

        scheduler.step()
        dt = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        # Print
        val_acc_str = (f"{val_metrics['accuracy']:.3f}"
                       if val_metrics else "  -  ")
        print(f"{epoch:5d} | {train_metrics['loss']:7.4f} | "
              f"{train_metrics['action_loss']:7.4f} | "
              f"{train_metrics['depth_loss']:7.4f} | "
              f"{train_metrics['accuracy']:.3f} | {val_acc_str} | "
              f"{lr:.2e} | {dt:5.1f}s")

        if val_metrics and "per_action_acc" in val_metrics:
            parts = [f"{k}={v:.2f}" for k, v in
                     val_metrics["per_action_acc"].items()]
            print(f"        Per-action: {', '.join(parts)}")

        # TensorBoard
        writer.add_scalar("train/loss", train_metrics["loss"], epoch)
        writer.add_scalar("train/action_loss",
                          train_metrics["action_loss"], epoch)
        writer.add_scalar("train/depth_loss",
                          train_metrics["depth_loss"], epoch)
        writer.add_scalar("train/accuracy",
                          train_metrics["accuracy"], epoch)
        writer.add_scalar("lr", lr, epoch)
        if val_metrics:
            writer.add_scalar("val/accuracy",
                              val_metrics["accuracy"], epoch)
            writer.add_scalar("val/loss", val_metrics["loss"], epoch)

        # Save checkpoint
        is_best = val_metrics and val_metrics["accuracy"] > best_acc
        if is_best:
            best_acc = val_metrics["accuracy"]

        ckpt = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "best_acc": best_acc,
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
        }

        torch.save(ckpt, os.path.join(SAVE_DIR, "latest.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(SAVE_DIR, "best.pt"))
            print(f"        ★ New best: {best_acc:.3f}")

        # Save stripped model every 50 epochs
        if (epoch + 1) % 50 == 0:
            deploy_model = TeacherModel(
                feat_dim=128, cue_dim=256,
                num_actions=NUM_ACTIONS, use_depth_head=True,
            )
            deploy_model.load_state_dict(model.state_dict())
            deploy_model.strip_depth_head()
            deploy_path = os.path.join(
                SAVE_DIR, f"deploy_epoch{epoch + 1}.pt"
            )
            torch.save(deploy_model.state_dict(), deploy_path)
            print(f"        Saved deploy model: {deploy_path}")

    writer.close()

    # ── Final export ──
    print(f"\n{'═' * 50}")
    print(f"  Training complete!")
    print(f"  Best validation accuracy: {best_acc:.3f}")
    print(f"  Checkpoints: {SAVE_DIR}/")
    print(f"{'═' * 50}")

    model.strip_depth_head()
    final_path = os.path.join(SAVE_DIR, "deploy_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"  Deployment model: {final_path}")


if __name__ == "__main__":
    main()
