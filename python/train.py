#!/usr/bin/env python3
"""
train.py — Training loop for depth-aware ImageNav policy.

Trains the TeacherModel with:
  1. Action loss:   Cross-entropy on BFS shortest-path supervision
  2. Depth loss:    MSE on MiDaS pseudo-depth maps (auxiliary)
  3. Action weights: Class-balanced weighting (STOP is overrepresented)

Usage:
    # Single collection
    python train.py --data data/room1_start_west

    # Multiple collections (combined graph)
    python train.py --data data/room1_start_west data/room1_start_east

    # Resume training
    python train.py --data data/room1_start_west --resume checkpoints/latest.pt
"""

import os
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from models import TeacherModel
from dataset import get_dataloader, NUM_ACTIONS, ACTION_NAMES


def compute_class_weights(loader, num_actions=NUM_ACTIONS, num_batches=20):
    """Compute inverse-frequency class weights from a few batches."""
    counts = torch.zeros(num_actions)
    for i, batch in enumerate(loader):
        if i >= num_batches:
            break
        actions = batch["action"]
        for a in range(num_actions):
            counts[a] += (actions == a).sum().item()

    # Inverse frequency, normalized
    total = counts.sum()
    weights = total / (num_actions * counts.clamp(min=1))
    print(f"[Train] Action distribution: {dict(zip(ACTION_NAMES, counts.long().tolist()))}")
    print(f"[Train] Class weights: {dict(zip(ACTION_NAMES, weights.numpy().round(2).tolist()))}")
    return weights


def train_epoch(model, loader, optimizer, device, depth_weight=0.1,
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
def eval_epoch(model, loader, device, depth_weight=0.1):
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
    parser = argparse.ArgumentParser(description="Train ImageNav policy")
    parser.add_argument("--data", nargs="+", required=True,
                        help="Collection directory path(s)")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--depth-weight", type=float, default=0.1,
                        help="Weight for depth auxiliary loss")
    parser.add_argument("--samples-per-epoch", type=int, default=10000)
    parser.add_argument("--save-dir", default="checkpoints")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available()
                        else "cpu")
    parser.add_argument("--log-dir", default="runs")
    args = parser.parse_args()

    device = torch.device(args.device)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"{'═' * 50}")
    print(f"  Depth-Aware ImageNav Training")
    print(f"  Data:     {args.data}")
    print(f"  Device:   {device}")
    print(f"  Epochs:   {args.epochs}")
    print(f"  Batch:    {args.batch_size}")
    print(f"  LR:       {args.lr}")
    print(f"  Depth w:  {args.depth_weight}")
    print(f"{'═' * 50}\n")

    # ── Data ──
    train_loader = get_dataloader(
        args.data,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch,
        augment=True,
        num_workers=4,
    )
    val_loader = get_dataloader(
        args.data,
        batch_size=args.batch_size,
        samples_per_epoch=args.samples_per_epoch // 5,
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Class weights ──
    class_weights = compute_class_weights(train_loader)

    # ── Resume ──
    start_epoch = 0
    best_acc = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_acc = ckpt.get("best_acc", 0)
        print(f"[Resume] from epoch {start_epoch}, best_acc={best_acc:.3f}")

    # ── TensorBoard ──
    writer = SummaryWriter(args.log_dir)

    # ── Training Loop ──
    print(f"\n{'Epoch':>5} | {'Loss':>7} | {'A.Loss':>7} | {'D.Loss':>7} | "
          f"{'Acc':>6} | {'V.Acc':>6} | {'LR':>8} | {'Time':>5}")
    print("─" * 70)

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, device,
            depth_weight=args.depth_weight,
            class_weights=class_weights,
        )

        # Eval (every 5 epochs)
        val_metrics = None
        if (epoch + 1) % 5 == 0 or epoch == args.epochs - 1:
            val_metrics = eval_epoch(
                model, val_loader, device,
                depth_weight=args.depth_weight,
            )

        scheduler.step()
        dt = time.time() - t0
        lr = scheduler.get_last_lr()[0]

        # Print
        val_acc_str = f"{val_metrics['accuracy']:.3f}" if val_metrics else "  -  "
        print(f"{epoch:5d} | {train_metrics['loss']:7.4f} | "
              f"{train_metrics['action_loss']:7.4f} | "
              f"{train_metrics['depth_loss']:7.4f} | "
              f"{train_metrics['accuracy']:.3f} | {val_acc_str} | "
              f"{lr:.2e} | {dt:5.1f}s")

        # Per-action accuracy
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
        writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
        writer.add_scalar("lr", lr, epoch)
        if val_metrics:
            writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
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

        torch.save(ckpt, os.path.join(args.save_dir, "latest.pt"))
        if is_best:
            torch.save(ckpt, os.path.join(args.save_dir, "best.pt"))
            print(f"        ★ New best: {best_acc:.3f}")

        # Save stripped model (no depth head) every 50 epochs
        if (epoch + 1) % 50 == 0:
            deploy_model = TeacherModel(
                feat_dim=128, cue_dim=256,
                num_actions=NUM_ACTIONS, use_depth_head=True,
            )
            deploy_model.load_state_dict(model.state_dict())
            deploy_model.strip_depth_head()
            deploy_path = os.path.join(
                args.save_dir, f"deploy_epoch{epoch + 1}.pt"
            )
            torch.save(deploy_model.state_dict(), deploy_path)
            print(f"        Saved deploy model: {deploy_path}")

    writer.close()

    # ── Final export ──
    print(f"\n{'═' * 50}")
    print(f"  Training complete!")
    print(f"  Best validation accuracy: {best_acc:.3f}")
    print(f"  Checkpoints: {args.save_dir}/")
    print(f"{'═' * 50}")

    # Strip and save final deployment model
    model.strip_depth_head()
    final_path = os.path.join(args.save_dir, "deploy_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"  Deployment model: {final_path}")
    print(f"  → Next: python quantize_export.py --model {final_path}")


if __name__ == "__main__":
    main()
