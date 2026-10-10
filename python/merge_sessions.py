#!/usr/bin/env python3
"""
merge_sessions.py — Merge multiple exploration sessions into one dataset.

Input (per session, from receiver.py):
    data/<session>/
      trajectory.jsonl       ← one JSON line per timestep
      images/frame_NNNNNN.jpg

Output:
    data/merged/
      dataset.npz            ← frames (N, 48, 48, 3) uint8, actions (N,) int32
      segments.json           ← segment boundaries [{start, end, length, source}]
      dataset_meta.json       ← summary metadata

Pipeline position:
    receiver.py → [this] → hindsight_relabel.py → train_teacher.py
"""

import os
import json
import glob
import time

import numpy as np
from PIL import Image

# ═══════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════
DATA_ROOT           = "./data"
OUTPUT_DIR          = "./data/merged"
MIN_SEGMENT_LENGTH  = 5       # discard segments shorter than this
IMG_SIZE            = 256
# ═══════════════════════════════════════════


def discover_sessions(data_root):
    """
    Auto-discover session directories containing trajectory.jsonl.

    Returns list of session directory paths, sorted by name.
    """
    sessions = []
    for entry in sorted(os.listdir(data_root)):
        subdir = os.path.join(data_root, entry)
        traj_path = os.path.join(subdir, "trajectory.jsonl")
        if os.path.isdir(subdir) and os.path.isfile(traj_path):
            sessions.append(subdir)

    # Exclude the output directory itself
    sessions = [s for s in sessions if os.path.abspath(s) !=
                os.path.abspath(OUTPUT_DIR)]

    print(f"[Discover] Found {len(sessions)} sessions:")
    for s in sessions:
        print(f"  - {os.path.basename(s)}")
    return sessions


def load_session(session_dir):
    """
    Load one session's images and trajectory.

    Returns:
        frames: np.array (N, 48, 48, 3) uint8
        trajectory: list of dicts from trajectory.jsonl
    """
    traj_path = os.path.join(session_dir, "trajectory.jsonl")
    img_dir = os.path.join(session_dir, "images")

    # Load trajectory
    trajectory = []
    with open(traj_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                trajectory.append(json.loads(line))

    if not trajectory:
        print(f"  [WARN] Empty trajectory in {session_dir}")
        return None, None

    # Load frames in order
    frames = []
    for step in trajectory:
        img_path = os.path.join(img_dir, step["frame"])
        if not os.path.exists(img_path):
            print(f"  [WARN] Missing image: {img_path}")
            return None, None
        img = Image.open(img_path).convert("RGB")
        if img.size != (IMG_SIZE, IMG_SIZE):
            img = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)
        frames.append(np.array(img, dtype=np.uint8))

    frames = np.stack(frames, axis=0)  # (N, 48, 48, 3)
    return frames, trajectory


def find_segments(trajectory, min_length=MIN_SEGMENT_LENGTH):
    """
    Split trajectory into contiguous segments at obstacle override events.

    An override breaks the causal chain (the action the explorer
    intended was replaced), so we segment there.

    Returns list of (start, end) index pairs.
    """
    segments = []
    seg_start = 0

    for i, step in enumerate(trajectory):
        if step.get("blocked", False):
            # Override happened — end current segment
            if i - seg_start >= min_length:
                segments.append((seg_start, i))
            seg_start = i + 1

    # Final segment
    if len(trajectory) - seg_start >= min_length:
        segments.append((seg_start, len(trajectory)))

    return segments


def main():
    print("=" * 50)
    print("  Merge Exploration Sessions")
    print("=" * 50)

    t_start = time.time()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Discover sessions
    sessions = discover_sessions(DATA_ROOT)
    if not sessions:
        print("\n[ERROR] No sessions found. Run receiver.py first.")
        return

    # Process each session
    all_segments = []
    total_frames = 0

    for sess_dir in sessions:
        print(f"\n[*] Loading: {os.path.basename(sess_dir)}")
        frames, trajectory = load_session(sess_dir)
        if frames is None:
            continue

        segments = find_segments(trajectory)
        print(f"    {len(frames)} frames → {len(segments)} segments")

        for seg_start, seg_end in segments:
            seg_len = seg_end - seg_start
            seg_frames = frames[seg_start:seg_end]
            seg_actions = np.array([
                trajectory[i]["action"] for i in range(seg_start, seg_end)
            ], dtype=np.int32)

            all_segments.append({
                "frames": seg_frames,
                "actions": seg_actions,
                "length": seg_len,
                "source": os.path.basename(sess_dir),
            })
            total_frames += seg_len

    if not all_segments:
        print("\n[ERROR] No valid segments found.")
        return

    print(f"\n[*] Total: {len(all_segments)} segments, {total_frames} frames")

    # Concatenate all segments into one flat array
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

    all_frames = np.concatenate(all_frames_list, axis=0)   # (N, 48, 48, 3)
    all_actions = np.concatenate(all_actions_list, axis=0)  # (N,)

    # Save
    print(f"[*] Saving to {OUTPUT_DIR}...")

    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "dataset.npz"),
        frames=all_frames, actions=all_actions,
    )

    with open(os.path.join(OUTPUT_DIR, "segments.json"), "w") as f:
        json.dump(segment_index, f, indent=2)

    # Action distribution
    action_names = ["FWD", "RIGHT", "LEFT", "STOP"]
    action_counts = {name: int(np.sum(all_actions == i))
                     for i, name in enumerate(action_names)}

    meta = {
        "total_frames": total_frames,
        "total_segments": len(all_segments),
        "sessions": [os.path.basename(s) for s in sessions],
        "action_distribution": action_counts,
        "img_shape": [48, 48, 3],
    }
    with open(os.path.join(OUTPUT_DIR, "dataset_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t_start
    size_mb = os.path.getsize(os.path.join(OUTPUT_DIR, "dataset.npz")) / 1e6

    print(f"\n{'=' * 50}")
    print(f"  Merged dataset saved!")
    print(f"  Frames:     {total_frames}")
    print(f"  Segments:   {len(all_segments)}")
    print(f"  Actions:    {action_counts}")
    print(f"  Size:       {size_mb:.1f} MB")
    print(f"  Time:       {elapsed:.1f}s")
    print(f"  Output:     {OUTPUT_DIR}")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
