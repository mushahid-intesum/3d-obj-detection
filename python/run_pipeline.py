#!/usr/bin/env python3
"""
run_pipeline.py — Phase 4.5: Run the full data preparation pipeline.

Executes the complete chain from raw exploration data to training-ready
offline RL dataset:

    1. gen_depth.py        — Generate MiDaS depth maps for all sessions
    2. merge_sessions.py   — Merge sessions into single dataset + segments
    3. hindsight_relabel.py — Build goal-conditioned offline RL transitions

Produces:
    data/offline_dataset/
      frames.npz          — shared image bank (N, 128, 128, 3)
      transitions.npz     — obs_idx, next_obs_idx, goal_idx, actions, rewards, dones
      dataset_info.json   — statistics

Usage:
    python run_pipeline.py                    # run all steps
    python run_pipeline.py --skip-depth       # skip MiDaS (if already done)
    python run_pipeline.py --skip-merge       # skip merge (if already done)
"""

import os
import sys
import time
import subprocess
import json


# ═══════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════
DATA_ROOT       = "data"
MERGED_DIR      = "data/merged"
OFFLINE_DIR     = "data/offline_dataset"
PYTHON          = sys.executable
# ═══════════════════════════════════════════


def run_step(name, script, check_output=None):
    """Run a pipeline step and check for success."""
    print(f"\n{'═' * 60}")
    print(f"  Step: {name}")
    print(f"  Script: {script}")
    print(f"{'═' * 60}\n")

    t0 = time.time()
    result = subprocess.run(
        [PYTHON, script],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )

    elapsed = time.time() - t0

    if result.returncode != 0:
        print(f"\n  ✗ FAILED (exit code {result.returncode})")
        print(f"    Fix the error and re-run.")
        return False

    # Verify output exists
    if check_output and not os.path.exists(check_output):
        print(f"\n  ✗ Output missing: {check_output}")
        return False

    print(f"\n  ✓ Completed in {elapsed:.1f}s")
    return True


def count_sessions(data_root):
    """Count sessions with trajectory.jsonl."""
    count = 0
    for entry in os.listdir(data_root):
        subdir = os.path.join(data_root, entry)
        if os.path.isdir(subdir) and os.path.isfile(
            os.path.join(subdir, "trajectory.jsonl")
        ):
            count += 1
    return count


def main():
    skip_depth = "--skip-depth" in sys.argv
    skip_merge = "--skip-merge" in sys.argv

    print("╔═══════════════════════════════════════════════════╗")
    print("║   Full Data Preparation Pipeline                 ║")
    print("╚═══════════════════════════════════════════════════╝")

    # Pre-check: any sessions exist?
    n_sessions = count_sessions(DATA_ROOT)
    if n_sessions == 0:
        print(f"\n[ERROR] No sessions found in {DATA_ROOT}/")
        print("Run the receiver + exploration first to collect data.")
        sys.exit(1)

    print(f"\n[*] Found {n_sessions} session(s) in {DATA_ROOT}/")

    t_total = time.time()
    steps_run = 0

    # Step 1: Generate depth maps
    if not skip_depth:
        ok = run_step(
            "Generate MiDaS depth maps",
            "gen_depth.py",
        )
        if not ok:
            sys.exit(1)
        steps_run += 1
    else:
        print("\n[SKIP] Depth map generation (--skip-depth)")

    # Step 2: Merge sessions
    if not skip_merge:
        ok = run_step(
            "Merge exploration sessions",
            "merge_sessions.py",
            check_output=os.path.join(MERGED_DIR, "dataset.npz"),
        )
        if not ok:
            sys.exit(1)
        steps_run += 1
    else:
        print("\n[SKIP] Session merging (--skip-merge)")

    # Step 3: Hindsight relabeling
    ok = run_step(
        "Hindsight goal relabeling",
        "hindsight_relabel.py",
        check_output=os.path.join(OFFLINE_DIR, "transitions.npz"),
    )
    if not ok:
        sys.exit(1)
    steps_run += 1

    # Summary
    elapsed_total = time.time() - t_total

    print(f"\n{'═' * 60}")
    print(f"  Pipeline Complete!")
    print(f"  Steps run:  {steps_run}")
    print(f"  Total time: {elapsed_total:.0f}s")
    print(f"{'═' * 60}")

    # Print dataset stats
    info_path = os.path.join(OFFLINE_DIR, "dataset_info.json")
    if os.path.exists(info_path):
        with open(info_path) as f:
            info = json.load(f)
        print(f"\n  Dataset statistics:")
        print(f"    Total frames:      {info.get('total_frames', '?')}")
        print(f"    Total transitions: {info.get('total_transitions', '?')}")
        print(f"    Action dist:       {info.get('action_distribution', '?')}")
        print(f"    Relabels/step:     {info.get('num_relabels', '?')}")

    print(f"\n  Output: {OFFLINE_DIR}/")
    print(f"  Next:   python train_teacher.py")
    print(f"{'═' * 60}")


if __name__ == "__main__":
    main()
