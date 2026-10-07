#!/usr/bin/env python3
"""
explorer.py — Autonomous pink-noise exploration controller for Phase 2.

Connects to the ESP32 robot, sends pink-noise actions, receives frames,
and saves the complete exploration dataset to disk.
"""

import os
import time
import socket
import struct
import json
import signal

import numpy as np
from pink_noise import generate_exploration_actions

# ═══════════════════════════════════════════
#  Configuration — edit these before running
# ═══════════════════════════════════════════
ESP32_HOST          = "192.168.1.100"       # TODO: set ESP32 IP
ESP32_PORT          = 8888
OUTPUT_DIR          = "./data/session_01"   # output dir for this session
DURATION_SEC        = 1800                  # exploration duration (1800 = 30 min)
RATE_HZ             = 5.0                   # action/frame rate
BETA                = 1.0                   # noise exponent (1.0=pink, 2.0=brown)
SEED                = None                  # random seed (None = random)
OBSTACLE_THRESH_CM  = 12                    # ultrasonic override threshold
# ═══════════════════════════════════════════

# Protocol constants (must match wifi_stream.h)
MAGIC = 0x494D4731
HEADER_SIZE = 11
IMG_W, IMG_H, IMG_CH = 48, 48, 3

# Actions: pink noise generates 0=forward, 1=left, 2=right
# Mapped to firmware commands: F=NORTH, B=SOUTH, L=WEST, R=EAST, S=STAY
ACTION_TO_CMD = {0: b'F', 1: b'L', 2: b'R', 3: b'S'}
ACTION_NAMES = {0: "FWD", 1: "LFT", 2: "RGT", 3: "STP"}

_shutdown = False


def signal_handler(sig, frame):
    global _shutdown
    print("\n[!] Shutdown requested...")
    _shutdown = True


def recv_exact(sock, nbytes):
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            raise ConnectionError("ESP32 disconnected")
        buf.extend(chunk)
    return bytes(buf)


def receive_frame(sock):
    """Receive one frame. Handles both JPEG and raw RGB protocols."""
    header = recv_exact(sock, HEADER_SIZE)
    magic = struct.unpack_from('<I', header, 0)[0]
    if magic != MAGIC:
        raise ValueError(f"Bad magic: 0x{magic:08X}")
    frame_id = struct.unpack_from('<I', header, 4)[0]
    ultrasonic_cm = struct.unpack_from('<H', header, 8)[0]
    last_action = header[10]

    # Read image data — try JPEG (variable len) or fixed RGB
    # The firmware sends raw RGB888 (48x48x3 = 6912 bytes) for collection
    img_size = IMG_W * IMG_H * IMG_CH
    img_data = recv_exact(sock, img_size)
    image = np.frombuffer(img_data, dtype=np.uint8).reshape(IMG_H, IMG_W, IMG_CH)
    return frame_id, image, ultrasonic_cm, last_action


class ExplorationSession:
    """Manages a single exploration session: actions, frames, metadata."""

    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.frames_dir = os.path.join(output_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.trajectory = []
        self.meta = {
            "start_time": None, "end_time": None,
            "total_frames": 0, "total_overrides": 0,
            "img_shape": [IMG_H, IMG_W, IMG_CH],
        }

    def add_step(self, step_idx, image, action_sent, action_executed,
                 ultrasonic_cm, was_override, timestamp):
        fname = f"frame_{step_idx:06d}.npy"
        np.save(os.path.join(self.frames_dir, fname), image)
        self.trajectory.append({
            "step": step_idx, "timestamp": timestamp,
            "action_sent": int(action_sent),
            "action_executed": int(action_executed),
            "ultrasonic_cm": int(ultrasonic_cm),
            "override": was_override, "frame_file": fname,
        })

    def save(self):
        self.meta["end_time"] = time.time()
        self.meta["total_frames"] = len(self.trajectory)

        with open(os.path.join(self.output_dir, "trajectory.jsonl"), "w") as f:
            for step in self.trajectory:
                f.write(json.dumps(step) + "\n")

        with open(os.path.join(self.output_dir, "session_meta.json"), "w") as f:
            json.dump(self.meta, f, indent=2)

        if self.trajectory:
            all_frames = []
            for step in self.trajectory:
                fpath = os.path.join(self.frames_dir, step["frame_file"])
                all_frames.append(np.load(fpath))
            np.savez_compressed(
                os.path.join(self.output_dir, "frames_all.npz"),
                frames=np.stack(all_frames)
            )

        print(f"\n[OK] Session saved: {self.output_dir}")
        print(f"     Frames: {len(self.trajectory)}, "
              f"Overrides: {self.meta['total_overrides']}")


def run_exploration():
    global _shutdown
    total_steps = int(DURATION_SEC * RATE_HZ)
    step_interval = 1.0 / RATE_HZ

    print(f"[*] Generating {total_steps} pink-noise actions (beta={BETA})...")
    actions = generate_exploration_actions(total_steps, beta=BETA, seed=SEED)

    print(f"[*] Connecting to {ESP32_HOST}:{ESP32_PORT}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    sock.connect((ESP32_HOST, ESP32_PORT))
    print("[OK] Connected!")

    session = ExplorationSession(OUTPUT_DIR)
    session.meta["start_time"] = time.time()
    session.meta["config"] = {
        "duration_sec": DURATION_SEC, "rate_hz": RATE_HZ,
        "beta": BETA, "seed": SEED, "obstacle_cm": OBSTACLE_THRESH_CM,
    }

    print(f"[*] Exploring for {DURATION_SEC}s at {RATE_HZ}Hz ({total_steps} steps)")
    print(f"{'Step':>7} | {'Action':>6} | {'Dist':>6} | {'Ovr':>4} | {'FPS':>5}")
    print("-" * 42)

    t_start = time.time()
    overrides = 0

    try:
        for i in range(total_steps):
            if _shutdown:
                break
            t0 = time.time()

            action = int(actions[i])
            sock.sendall(ACTION_TO_CMD[action])

            frame_id, image, dist_cm, reported = receive_frame(sock)

            was_override = (reported != action and dist_cm < OBSTACLE_THRESH_CM)
            if was_override:
                action = reported
                overrides += 1

            session.add_step(i, image, int(actions[i]), action,
                             dist_cm, was_override, time.time())

            if (i + 1) % 50 == 0:
                elapsed = time.time() - t_start
                fps = (i + 1) / elapsed if elapsed > 0 else 0
                d = str(dist_cm) if dist_cm < 65535 else "N/A"
                o = "Y" if was_override else ""
                print(f"{i+1:>7} | {ACTION_NAMES[action]:>6} | "
                      f"{d:>6} | {o:>4} | {fps:>5.1f}")

            sleep = step_interval - (time.time() - t0)
            if sleep > 0:
                time.sleep(sleep)

    except (ConnectionError, BrokenPipeError) as e:
        print(f"\n[!] Connection lost: {e}")
    finally:
        try:
            sock.sendall(b'S')
        except Exception:
            pass
        sock.close()
        session.meta["total_overrides"] = overrides
        session.save()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, signal_handler)
    run_exploration()
