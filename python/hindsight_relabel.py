
import os
import json
import time

import numpy as np

MERGED_DIR          = "./data/merged"
OUTPUT_DIR          = "./data/offline_dataset"
NUM_RELABELS        = 5
MAX_HORIZON         = 50
TERMINAL_THRESHOLD  = 3
TERMINAL_REWARD     = 5.0
STEP_PENALTY        = -0.01
SEED                = 42


def load_merged_dataset(merged_dir):
    data = np.load(os.path.join(merged_dir, "dataset.npz"))
    frames = data["frames"]
    actions = data["actions"]
    headings = data["headings"]
    proximity = data["proximity"]

    with open(os.path.join(merged_dir, "segments.json"), "r") as f:
        segments = json.load(f)

    print(f"[*] Loaded {len(frames)} frames, {len(segments)} segments")
    print(f"    proximity: mean={proximity.mean():.1f}, "
          f"blocked={np.mean(proximity < 50.0)*100:.1f}%")
    return frames, actions, headings, proximity, segments


def relabel_segment(frames, actions, seg_start, seg_end, rng):
    transitions = []
    seg_len = seg_end - seg_start

    for local_t in range(seg_len - 1):
        t = seg_start + local_t
        remaining = seg_end - t - 1
        max_k = min(MAX_HORIZON, remaining)

        if max_k < 1:
            continue

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


def save_dataset(frames, headings, proximity, transitions, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    n = len(transitions)
    obs_idx = np.array([t["obs_idx"] for t in transitions], dtype=np.int32)
    next_obs_idx = np.array([t["next_obs_idx"] for t in transitions], dtype=np.int32)
    goal_idx = np.array([t["goal_idx"] for t in transitions], dtype=np.int32)
    action_arr = np.array([t["action"] for t in transitions], dtype=np.int32)
    reward_arr = np.array([t["reward"] for t in transitions], dtype=np.float32)
    done_arr = np.array([t["done"] for t in transitions], dtype=np.bool_)

    obs_proximity = proximity[obs_idx]
    goal_proximity = proximity[goal_idx]

    np.savez_compressed(
        os.path.join(output_dir, "frames.npz"),
        frames=frames,
        headings=headings,
        proximity=proximity,
    )

    np.savez_compressed(
        os.path.join(output_dir, "transitions.npz"),
        obs_idx=obs_idx,
        next_obs_idx=next_obs_idx,
        goal_idx=goal_idx,
        actions=action_arr,
        rewards=reward_arr,
        dones=done_arr,
        obs_proximity=obs_proximity,
        goal_proximity=goal_proximity,
    )

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
        "proximity_stats": {
            "obs_mean": round(float(obs_proximity.mean()), 1),
            "goal_mean": round(float(goal_proximity.mean()), 1),
        },
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
    frames, actions, headings, proximity, segments = load_merged_dataset(MERGED_DIR)

    print(f"\n[*] Relabeling with {NUM_RELABELS} goals/step, "
          f"horizon={MAX_HORIZON}, terminal≤{TERMINAL_THRESHOLD} steps")
    transitions = build_dataset(frames, actions, segments)

    print(f"\n[*] Total transitions: {len(transitions)}")
    print(f"[*] Saving to: {OUTPUT_DIR}")
    info = save_dataset(frames, headings, proximity, transitions, OUTPUT_DIR)

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
