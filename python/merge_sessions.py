#!/usr/bin/env python3
"""
merge_sessions.py — Merge multiple exploration sessions into a single dataset.

After running explorer.py multiple times (different sessions, lighting, etc.),
this script combines all sessions into one unified dataset ready for Phase 3.

Usage:
    python merge_sessions.py --sessions ./data/session_01 ./data/session_02 \
                             --output ./data/merged
"""

import argparse
import os
import json

import numpy as np


def load_session(session_dir):
    """Load one session's frames and trajectory."""
    # Load frames
    frames_path = os.path.join(session_dir, "frames_all.npz")
    if not os.path.exists(frames_path):
        print(f"[WARN] No frames_all.npz in {session_dir}, skipping")
        return None, None

    data = np.load(frames_path)
    frames = data["frames"]  # (N, 48, 48, 3)

    # Load trajectory
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

    Overrides indicate the robot hit an obstacle and its action was
    forcefully changed. These break the causal chain, so we segment
    the trajectory at these points.

    Returns list of (start_idx, end_idx) tuples.
    """
    boundaries = []
    seg_start = 0

    for i, step in enumerate(trajectory):
        if step.get("override", False):
            if i > seg_start + 2:  # at least 3 steps per segment
                boundaries.append((seg_start, i))
            seg_start = i + 1

    # Final segment
    if seg_start < len(trajectory) - 2:
        boundaries.append((seg_start, len(trajectory)))

    return boundaries


def main():
    parser = argparse.ArgumentParser(description="Merge exploration sessions")
    parser.add_argument("--sessions", nargs="+", required=True,
                        help="Paths to session directories")
    parser.add_argument("--output", required=True,
                        help="Output directory for merged dataset")
    parser.add_argument("--min-segment-length", type=int, default=5,
                        help="Minimum trajectory segment length (default: 5)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    all_segments = []  # list of (frames_array, actions_array) per segment
    total_frames = 0
    total_segments = 0

    for sess_dir in args.sessions:
        print(f"\n[*] Loading session: {sess_dir}")
        frames, trajectory = load_session(sess_dir)
        if frames is None:
            continue

        print(f"    Frames: {len(frames)}")

        # Segment at override boundaries
        boundaries = find_trajectory_boundaries(trajectory)
        print(f"    Segments: {len(boundaries)}")

        for seg_start, seg_end in boundaries:
            seg_len = seg_end - seg_start
            if seg_len < args.min_segment_length:
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

    # Save merged dataset
    # Segments stored as separate arrays with an index
    segment_index = []
    all_frames_list = []
    all_actions_list = []
    offset = 0

    for seg in all_segments:
        seg_len = seg["length"]
        all_frames_list.append(seg["frames"])
        all_actions_list.append(seg["actions"])

        segment_index.append({
            "start": offset,
            "end": offset + seg_len,
            "length": seg_len,
            "source": seg["source"],
        })
        offset += seg_len

    # Concatenate all frames and actions
    all_frames = np.concatenate(all_frames_list, axis=0)   # (N, 48, 48, 3)
    all_actions = np.concatenate(all_actions_list, axis=0)  # (N,)

    # Save
    np.savez_compressed(
        os.path.join(args.output, "dataset.npz"),
        frames=all_frames,
        actions=all_actions,
    )

    with open(os.path.join(args.output, "segments.json"), "w") as f:
        json.dump(segment_index, f, indent=2)

    meta = {
        "total_frames": total_frames,
        "total_segments": total_segments,
        "sessions": [os.path.basename(s) for s in args.sessions],
        "img_shape": [48, 48, 3],
    }
    with open(os.path.join(args.output, "dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    size_mb = os.path.getsize(os.path.join(args.output, "dataset.npz")) / 1e6
    print(f"\n[OK] Merged dataset saved to: {args.output}")
    print(f"     Frames: {total_frames}, Segments: {total_segments}")
    print(f"     Size: {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
