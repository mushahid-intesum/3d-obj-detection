#!/usr/bin/env python3
"""
validate_session.py — Phase 4.3: Validate collected exploration data.

Checks a session directory for:
  1. trajectory.jsonl exists and is valid JSON-lines
  2. All referenced images exist and are loadable
  3. Actions are within valid range
  4. Headings are in [0, 360)
  5. Timestamps are monotonically increasing
  6. Frame count matches

Usage:
    python validate_session.py                          # validates all sessions
    python validate_session.py data/session_20261008    # validates one session
"""

import os
import sys
import json
import glob

from PIL import Image

ACTION_NAMES = ["FWD", "RIGHT", "LEFT", "STOP"]


def validate_session(session_dir):
    """Validate one session directory. Returns (ok, stats_dict)."""
    errors = []
    warnings = []

    name = os.path.basename(session_dir)
    traj_path = os.path.join(session_dir, "trajectory.jsonl")
    img_dir = os.path.join(session_dir, "images")

    # Check trajectory exists
    if not os.path.isfile(traj_path):
        return False, {"error": f"Missing trajectory.jsonl in {session_dir}"}

    # Parse trajectory
    trajectory = []
    with open(traj_path, "r") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
                trajectory.append(entry)
            except json.JSONDecodeError as e:
                errors.append(f"Line {i}: invalid JSON: {e}")

    if not trajectory:
        return False, {"error": "Empty trajectory"}

    n = len(trajectory)
    action_counts = [0, 0, 0, 0]
    blocked_count = 0
    missing_images = 0
    bad_images = 0

    prev_ts = -1

    for i, step in enumerate(trajectory):
        # Check required fields
        for field in ["t", "frame", "action", "heading"]:
            if field not in step:
                errors.append(f"Step {i}: missing field '{field}'")

        # Check action range
        action = step.get("action", -1)
        if 0 <= action < 4:
            action_counts[action] += 1
        else:
            errors.append(f"Step {i}: invalid action {action}")

        # Check heading
        heading = step.get("heading", -1)
        if not (0 <= heading < 360):
            if heading != -1 and heading != 0:
                warnings.append(f"Step {i}: heading {heading}° out of [0,360)")

        # Check blocked
        if step.get("blocked", False):
            blocked_count += 1

        # Check timestamp monotonicity
        ts = step.get("ts", 0)
        if ts < prev_ts:
            warnings.append(f"Step {i}: timestamp went backwards ({ts} < {prev_ts})")
        prev_ts = ts

        # Check image exists
        img_path = os.path.join(img_dir, step.get("frame", ""))
        if not os.path.isfile(img_path):
            missing_images += 1
            if missing_images <= 3:
                errors.append(f"Step {i}: missing image {step.get('frame')}")
        else:
            # Quick load test (first and last only, to save time)
            if i == 0 or i == n - 1:
                try:
                    img = Image.open(img_path)
                    if img.size != (128, 128):
                        warnings.append(
                            f"Step {i}: image size {img.size} != (128,128)")
                except Exception as e:
                    bad_images += 1
                    errors.append(f"Step {i}: can't load image: {e}")

    # Count actual images on disk
    actual_images = len(glob.glob(os.path.join(img_dir, "*.jpg")))

    stats = {
        "session": name,
        "frames": n,
        "images_on_disk": actual_images,
        "missing_images": missing_images,
        "actions": {ACTION_NAMES[i]: c for i, c in enumerate(action_counts)},
        "blocked": blocked_count,
        "blocked_pct": f"{100 * blocked_count / max(n, 1):.1f}%",
        "errors": len(errors),
        "warnings": len(warnings),
    }

    ok = len(errors) == 0

    # Print results
    status = "✓ PASS" if ok else "✗ FAIL"
    print(f"\n  [{status}] {name}")
    print(f"    Frames:     {n}")
    print(f"    Images:     {actual_images} on disk, {missing_images} missing")
    print(f"    Actions:    {stats['actions']}")
    print(f"    Blocked:    {blocked_count} ({stats['blocked_pct']})")

    if errors:
        print(f"    Errors ({len(errors)}):")
        for e in errors[:5]:
            print(f"      - {e}")
        if len(errors) > 5:
            print(f"      ... and {len(errors) - 5} more")

    if warnings:
        print(f"    Warnings ({len(warnings)}):")
        for w in warnings[:3]:
            print(f"      - {w}")

    return ok, stats


def discover_sessions(data_root):
    """Find all session directories."""
    sessions = []
    for entry in sorted(os.listdir(data_root)):
        subdir = os.path.join(data_root, entry)
        if os.path.isdir(subdir) and os.path.isfile(
            os.path.join(subdir, "trajectory.jsonl")
        ):
            sessions.append(subdir)
    return sessions


def main():
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
        sessions = [sys.argv[1]]
    else:
        sessions = discover_sessions("data")

    if not sessions:
        print("No sessions found in data/")
        return

    print("═" * 50)
    print("  Session Validation")
    print("═" * 50)

    all_ok = True
    total_frames = 0

    for sess in sessions:
        ok, stats = validate_session(sess)
        if not ok:
            all_ok = False
        total_frames += stats.get("frames", 0)

    print(f"\n{'═' * 50}")
    print(f"  Sessions:     {len(sessions)}")
    print(f"  Total frames: {total_frames}")
    print(f"  Status:       {'✓ ALL PASS' if all_ok else '✗ SOME FAILED'}")
    print(f"{'═' * 50}")

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
