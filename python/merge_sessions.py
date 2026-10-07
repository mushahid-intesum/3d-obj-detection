#!/usr/bin/env python3
"""
merge_sessions.py — Merge multiple exploration sessions into a single dataset.

After running explorer.py multiple times, this script combines all sessions
into one unified dataset ready for Phase 3 (hindsight goal relabeling).
"""

import os
import json

import numpy as np

# ═══════════════════════════════════════════
#  Configuration — edit these before running
# ═══════════════════════════════════════════
SESSION_DIRS = [
    "./data/session_01",
    "./data/session_02",
    # Add more session paths here
]
OUTPUT_DIR = "./data/merged"
MIN_SEGMENT_LENGTH = 5      # discard segments shorter than this
# ═══════════════════════════════════════════


def load_session(session_dir):
    """Load one session's frames and trajectory."""
    frames_path = os.path.join(session_dir, "frames_all.npz")
    if not os.path.exists(frames_path):
        print(f"[WARN] No frames_all.npz in {session_dir}, skipping")
        return None, None

    data = np.load(frames_path)
    frames = data["frames"]

    traj_path = os.path.join(session_dir, "trajectory.jsonl")
    trajectory = []
    with open(traj_path, "r") as f:
        for line in f:
            trajectory.append(json.loads(line.strip()))

    assert len(frames) == len(trajectory), \
        f"Mismatch: {len(frames)} frames vs {len(trajectory)} trajectory steps"

    return frames, trajectory


def find_trajectory_boundaries(trajectory):
    """
    Split a session into sub-trajectories at override events.
    Overrides break the causal chain, so we segment there.
    """
    boundaries = []
    seg_start = 0

    for i, step in enumerate(trajectory):
        if step.get("override", False):
            if i > seg_start + 2:
                boundaries.append((seg_start, i))
            seg_start = i + 1

    if seg_start < len(trajectory) - 2:
        boundaries.append((seg_start, len(trajectory)))

    return boundaries


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_segments = []
    total_frames = 0
    total_segments = 0

    for sess_dir in SESSION_DIRS:
        print(f"\n[*] Loading session: {sess_dir}")
        frames, trajectory = load_session(sess_dir)
        if frames is None:
            continue

        print(f"    Frames: {len(frames)}")

        boundaries = find_trajectory_boundaries(trajectory)
        print(f"    Segments: {len(boundaries)}")

        for seg_start, seg_end in boundaries:
            seg_len = seg_end - seg_start
            if seg_len < MIN_SEGMENT_LENGTH:
                continue

            seg_frames = frames[seg_start:seg_end]
            seg_actions = np.array([
                trajectory[i]["action_executed"]
                for i in range(seg_start, seg_end)
            ], dtype=np.int32)

            all_segments.append({
                "frames": seg_frames,
                "actions": seg_actions,
                "length": seg_len,
                "source": os.path.basename(sess_dir),
                "original_range": [seg_start, seg_end],
            })

            total_frames += seg_len
            total_segments += 1

    print(f"\n[*] Total: {total_segments} segments, {total_frames} frames")

    segment_index = []
    all_frames_list = []
    all_actions_list = []
    offset = 0

    for seg in all_segments:
        seg_len = seg["length"]
        all_frames_list.append(seg["frames"])
        all_actions_list.append(seg["actions"])
        segment_index.append({
            "start": offset, "end": offset + seg_len,
            "length": seg_len, "source": seg["source"],
        })
        offset += seg_len

    all_frames = np.concatenate(all_frames_list, axis=0)
    all_actions = np.concatenate(all_actions_list, axis=0)

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "dataset.npz"),
        frames=all_frames, actions=all_actions,
    )

    with open(os.path.join(OUTPUT_DIR, "segments.json"), "w") as f:
        json.dump(segment_index, f, indent=2)

    meta = {
        "total_frames": total_frames,
        "total_segments": total_segments,
        "sessions": [os.path.basename(s) for s in SESSION_DIRS],
        "img_shape": [48, 48, 3],
    }
    with open(os.path.join(OUTPUT_DIR, "dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = os.path.getsize(os.path.join(OUTPUT_DIR, "dataset.npz")) / 1e6
    print(f"\n[OK] Merged dataset saved to: {OUTPUT_DIR}")
    print(f"     Frames: {total_frames}, Segments: {total_segments}")
    print(f"     Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
