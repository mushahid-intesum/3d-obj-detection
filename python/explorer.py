#!/usr/bin/env python3
"""
explorer.py — Autonomous pink-noise exploration controller for Phase 2.

Connects to the ESP32 robot, sends pink-noise actions, receives frames,
and saves the complete exploration dataset to disk.

Usage:
    python explorer.py --host <ESP32_IP> --output ./data/session_01
"""

import argparse
import os
import sys
import time
import socket
import struct
import json
import signal

import numpy as np
from pink_noise import generate_exploration_actions

MAGIC = 0x494D4731
HEADER_SIZE = 11
IMG_W, IMG_H, IMG_CH = 48, 48, 3
IMG_SIZE = IMG_W * IMG_H * IMG_CH
PACKET_SIZE = HEADER_SIZE + IMG_SIZE

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
    raw = recv_exact(sock, PACKET_SIZE)
    magic = struct.unpack_from('<I', raw, 0)[0]
    if magic != MAGIC:
        raise ValueError(f"Bad magic: 0x{magic:08X}")
    frame_id = struct.unpack_from('<I', raw, 4)[0]
    ultrasonic_cm = struct.unpack_from('<H', raw, 8)[0]
    last_action = raw[10]
    image = np.frombuffer(raw[HEADER_SIZE:], dtype=np.uint8).reshape(IMG_H, IMG_W, IMG_CH)
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

        # Compact archive of all frames
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


def run_exploration(host, port, output_dir, duration_sec, rate_hz,
                    beta, seed, obstacle_cm):
    global _shutdown
    total_steps = int(duration_sec * rate_hz)
    step_interval = 1.0 / rate_hz

    print(f"[*] Generating {total_steps} pink-noise actions (beta={beta})...")
    actions = generate_exploration_actions(total_steps, beta=beta, seed=seed)

    print(f"[*] Connecting to {host}:{port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    sock.connect((host, port))
    print("[OK] Connected!")

    session = ExplorationSession(output_dir)
    session.meta["start_time"] = time.time()
    session.meta["config"] = {
        "duration_sec": duration_sec, "rate_hz": rate_hz,
        "beta": beta, "seed": seed, "obstacle_cm": obstacle_cm,
    }

    print(f"[*] Exploring for {duration_sec}s at {rate_hz}Hz ({total_steps} steps)")
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

            was_override = (reported != action and dist_cm < obstacle_cm)
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


def main():
    signal.signal(signal.SIGINT, signal_handler)
    p = argparse.ArgumentParser(description="Phase 2: Pink-Noise Exploration")
    p.add_argument("--host", required=True, help="ESP32 IP address")
    p.add_argument("--port", type=int, default=8888)
    p.add_argument("--output", required=True, help="Output dir (e.g., ./data/session_01)")
    p.add_argument("--duration", type=int, default=1800, help="Seconds (default: 1800)")
    p.add_argument("--rate", type=float, default=5.0, help="Hz (default: 5)")
    p.add_argument("--beta", type=float, default=1.0, help="Noise exponent (default: 1.0)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--obstacle-cm", type=int, default=12, help="Override threshold (cm)")
    args = p.parse_args()

    run_exploration(args.host, args.port, args.output, args.duration,
                    args.rate, args.beta, args.seed, args.obstacle_cm)


if __name__ == "__main__":
    main()
