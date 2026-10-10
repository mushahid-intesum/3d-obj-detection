#!/usr/bin/env python3
"""
hindsight_relabel.py — Phase 3: Build offline RL dataset via hindsight goal relabeling.

Takes the merged exploration dataset (from merge_sessions.py) and produces
a goal-conditioned offline RL dataset by retroactively assigning goals
from future timesteps in each trajectory segment.

Each training sample:  (obs, goal, action, reward, next_obs, done)
"""

import os
import json
import time

import numpy as np

# ═══════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════
MERGED_DIR          = "./data/merged"           # output of merge_sessions.py
OUTPUT_DIR          = "./data/offline_dataset"   # where to save the RL dataset
NUM_RELABELS        = 5         # number of future goals sampled per timestep
MAX_HORIZON         = 50        # max future steps to sample goals from
TERMINAL_THRESHOLD  = 3         # steps within this horizon count as near-goal
TERMINAL_REWARD     = 5.0       # reward for reaching the goal
STEP_PENALTY        = -0.01     # per-step penalty to encourage efficiency
SEED                = 42
# ═══════════════════════════════════════════


def load_merged_dataset(merged_dir):
    """Load the merged dataset and segment index."""
    data = np.load(os.path.join(merged_dir, "dataset.npz"))
    frames = data["frames"]     # (N, 128, 128, 3) uint8
    actions = data["actions"]   # (N,) int32

    with open(os.path.join(merged_dir, "segments.json"), "r") as f:
        segments = json.load(f)

    print(f"[*] Loaded {len(frames)} frames, {len(segments)} segments")
    return frames, actions, segments


def relabel_segment(frames, actions, seg_start, seg_end, rng):
    """
    Apply hindsight goal relabeling to one trajectory segment.

    For each timestep t in [seg_start, seg_end-1]:
      - Sample NUM_RELABELS future timesteps k from [1, min(MAX_HORIZON, remaining)]
      - The frame at t+k becomes the "goal image"
      - The action at t is the action taken
      - Reward: +TERMINAL_REWARD if k <= TERMINAL_THRESHOLD, else STEP_PENALTY

    Returns list of transition dicts.
    """
    transitions = []
    seg_len = seg_end - seg_start

    for local_t in range(seg_len - 1):
        t = seg_start + local_t
        remaining = seg_end - t - 1
        max_k = min(MAX_HORIZON, remaining)

        if max_k < 1:
            continue

        # Sample future goal indices
        k_values = rng.integers(1, max_k + 1, size=NUM_RELABELS)

        for k in k_values:
            goal_idx = t + k
            is_terminal = (k <= TERMINAL_THRESHOLD)

            transitions.append({
                "obs_idx": t,
                "next_obs_idx": t + 1,
                "goal_idx": goal_idx,
                "action": int(actions[t]),
                "reward": TERMINAL_REWARD if is_terminal else STEP_PENALTY,
                "done": bool(k == 1),
            })

    return transitions


def build_dataset(frames, actions, segments):
    """
    Build the full offline RL dataset from all segments.

    Instead of duplicating image data, we store indices into the
    frames array. The actual images are loaded on-the-fly during training.
    """
    rng = np.random.default_rng(SEED)
    all_transitions = []

    for i, seg in enumerate(segments):
        seg_start = seg["start"]
        seg_end = seg["end"]
        seg_len = seg["length"]

        transitions = relabel_segment(frames, actions, seg_start, seg_end, rng)
        all_transitions.extend(transitions)

        if (i + 1) % 20 == 0 or i == len(segments) - 1:
            print(f"    Segment {i+1}/{len(segments)}: "
                  f"{seg_len} steps → {len(transitions)} transitions")

    return all_transitions


def save_dataset(frames, transitions, output_dir):
    """
    Save the offline RL dataset in an efficient format.

    We save:
      - frames.npz: all frames (shared, referenced by index)
      - transitions.npz: arrays of obs_idx, goal_idx, action, reward, next_obs_idx, done
      - dataset_info.json: metadata
    """
    os.makedirs(output_dir, exist_ok=True)

    n = len(transitions)
    obs_idx = np.array([t["obs_idx"] for t in transitions], dtype=np.int32)
    next_obs_idx = np.array([t["next_obs_idx"] for t in transitions], dtype=np.int32)
    goal_idx = np.array([t["goal_idx"] for t in transitions], dtype=np.int32)
    action_arr = np.array([t["action"] for t in transitions], dtype=np.int32)
    reward_arr = np.array([t["reward"] for t in transitions], dtype=np.float32)
    done_arr = np.array([t["done"] for t in transitions], dtype=np.bool_)

    # Save frames (shared image bank)
    np.savez_compressed(
        os.path.join(output_dir, "frames.npz"),
        frames=frames,
    )

    # Save transitions (indices + labels)
    np.savez_compressed(
        os.path.join(output_dir, "transitions.npz"),
        obs_idx=obs_idx,
        next_obs_idx=next_obs_idx,
        goal_idx=goal_idx,
        actions=action_arr,
        rewards=reward_arr,
        dones=done_arr,
    )

    # Compute stats
    reward_pos = np.sum(reward_arr > 0)
    reward_neg = np.sum(reward_arr < 0)
    action_counts = {int(a): int(c) for a, c in
                     zip(*np.unique(action_arr, return_counts=True))}

    info = {
        "total_transitions": n,
        "total_frames": len(frames),
        "num_relabels": NUM_RELABELS,
        "max_horizon": MAX_HORIZON,
        "terminal_threshold": TERMINAL_THRESHOLD,
        "action_distribution": action_counts,
        "positive_reward_count": int(reward_pos),
        "negative_reward_count": int(reward_neg),
        "seed": SEED,
    }

    with open(os.path.join(output_dir, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    return info


def main():
    print("=" * 50)
    print("  Phase 3: Hindsight Goal Relabeling")
    print("=" * 50)

    t_start = time.time()

    print(f"\n[*] Loading merged dataset from: {MERGED_DIR}")
    frames, actions, segments = load_merged_dataset(MERGED_DIR)

    print(f"\n[*] Relabeling with {NUM_RELABELS} goals/step, "
          f"horizon={MAX_HORIZON}, terminal≤{TERMINAL_THRESHOLD} steps")
    transitions = build_dataset(frames, actions, segments)

    print(f"\n[*] Total transitions: {len(transitions)}")
    print(f"[*] Saving to: {OUTPUT_DIR}")
    info = save_dataset(frames, transitions, OUTPUT_DIR)

    elapsed = time.time() - t_start
    print(f"\n{'='*50}")
    print(f"  Dataset built in {elapsed:.1f}s")
    print(f"  Transitions: {info['total_transitions']}")
    print(f"  Frames:      {info['total_frames']}")
    print(f"  Action dist: {info['action_distribution']}")
    print(f"  Terminal:    {info['positive_reward_count']} "
          f"({100*info['positive_reward_count']/info['total_transitions']:.1f}%)")
    print(f"  Output:      {OUTPUT_DIR}")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
