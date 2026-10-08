#!/usr/bin/env python3
"""
receiver.py — TCP client for free exploration data collection (IMG4 protocol).

Connects to Camera Board ESP32-S3, receives IMG4 packets containing:
  - JPEG frame (captured by camera)
  - action_taken (from Motor Board via SPI)
  - depth_blocked (from depth guard)
  - heading_deg (from IMU via Motor Board)
  - timestep (session-local index)

Saves:
  data/<session_name>/
    trajectory.jsonl      ← one JSON line per timestep
    images/frame_NNNNNN.jpg
    metadata.json         ← session-level metadata
"""

import socket
import struct
import time
import os
import json
from io import BytesIO
from datetime import datetime

import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ═══════════════════════════════════════════════
#  Configuration — edit these before running
# ═══════════════════════════════════════════════
ESP32_IP        = "192.168.1.100"   # Camera Board IP
ESP32_PORT      = 8888
SESSION_NAME    = ""                # auto-generated if empty
OUTPUT_DIR      = "data"            # base output directory
IMG_SIZE        = 48                # resize to NxN
SHOW_PREVIEW    = True              # live OpenCV preview
# ═══════════════════════════════════════════════

# IMG4 protocol (must match wifi_stream.h)
MAGIC = 0x494D4734           # "IMG4"
HEADER_SIZE = 22             # 4+4+4+1+1+4+4

ACTION_NAMES = ["FWD", "RIGHT", "LEFT", "STOP"]


def recv_exact(sock, n):
    """Receive exactly n bytes."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf.extend(chunk)
    return bytes(buf)


def main():
    # Generate session name if not set
    session_name = SESSION_NAME
    if not session_name:
        session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")

    out_dir = os.path.join(OUTPUT_DIR, session_name)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    traj_path = os.path.join(out_dir, "trajectory.jsonl")

    print(f"Session: {session_name}")
    print(f"Output:  {out_dir}")
    print(f"Connecting to {ESP32_IP}:{ESP32_PORT}...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(15)
    sock.connect((ESP32_IP, ESP32_PORT))
    print("Connected!\n")

    frame_count = 0
    override_count = 0
    t_start = time.time()

    # Open trajectory file for streaming writes
    traj_file = open(traj_path, "w")

    try:
        while True:
            # ── Receive IMG4 header (22 bytes) ──
            header = recv_exact(sock, HEADER_SIZE)
            (magic, frame_id, timestep, action_taken,
             depth_blocked, heading_deg, jpeg_len) = \
                struct.unpack("<IIIBBfI", header)

            if magic != MAGIC:
                print(f"  [WARN] Bad magic 0x{magic:08X}, resync...")
                continue

            # ── Receive JPEG data ──
            jpeg_data = recv_exact(sock, jpeg_len)

            # ── Decode & resize to 48×48 ──
            img = Image.open(BytesIO(jpeg_data)).convert("RGB")
            img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

            # ── Save image ──
            filename = f"frame_{frame_count:06d}.jpg"
            filepath = os.path.join(img_dir, filename)
            img_resized.save(filepath, quality=95)

            # ── Write trajectory line ──
            entry = {
                "t": frame_count,
                "frame": filename,
                "action": int(action_taken),
                "heading": round(float(heading_deg), 2),
                "blocked": bool(depth_blocked),
                "ts": round(time.time(), 3),
            }
            traj_file.write(json.dumps(entry) + "\n")
            traj_file.flush()

            # Track overrides
            if depth_blocked:
                override_count += 1

            frame_count += 1

            # ── Console output (every 10 frames) ──
            if frame_count % 10 == 0:
                elapsed = time.time() - t_start
                fps = frame_count / elapsed if elapsed > 0 else 0
                action_str = ACTION_NAMES[action_taken] if action_taken < 4 else "?"
                print(f"  [{frame_count:>6}] "
                      f"act={action_str:<5} "
                      f"hdg={heading_deg:6.1f}° "
                      f"blk={depth_blocked} "
                      f"({fps:.1f} fps, {override_count} overrides)")

            # ── Live preview ──
            if HAS_CV2 and SHOW_PREVIEW:
                arr = np.array(img_resized)
                display = cv2.resize(arr, (384, 384),
                                     interpolation=cv2.INTER_NEAREST)
                display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)

                action_str = ACTION_NAMES[action_taken] if action_taken < 4 else "?"
                label = (f"Step:{frame_count} Act:{action_str} "
                         f"Hdg:{heading_deg:.0f}")
                color = (0, 0, 255) if depth_blocked else (0, 255, 0)
                cv2.putText(display, label, (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                if depth_blocked:
                    cv2.putText(display, "BLOCKED", (8, 370),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Exploration", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    except ConnectionError as e:
        print(f"\nConnection lost: {e}")
    finally:
        traj_file.close()
        sock.close()
        if HAS_CV2 and SHOW_PREVIEW:
            cv2.destroyAllWindows()

        # Save session metadata
        elapsed = time.time() - t_start
        metadata = {
            "session_name": session_name,
            "total_frames": frame_count,
            "overrides": override_count,
            "duration_s": round(elapsed, 1),
            "fps": round(frame_count / elapsed, 2) if elapsed > 0 else 0,
            "img_size": IMG_SIZE,
            "esp32_ip": ESP32_IP,
            "created": datetime.now().isoformat(),
        }
        meta_path = os.path.join(out_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'═' * 45}")
        print(f"  Frames:    {frame_count}")
        print(f"  Overrides: {override_count}")
        print(f"  Duration:  {elapsed:.0f}s ({metadata['fps']:.1f} fps)")
        print(f"  Saved to:  {out_dir}")
        print(f"{'═' * 45}")


if __name__ == "__main__":
    main()
