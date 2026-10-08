#!/usr/bin/env python3
"""
receiver.py — TCP client for Phase 1/2 data collection.

Connects to ESP32-S3, receives IMG3 packets (JPEG + IMU heading),
decodes to 48×48 RGB, and saves organized data with metadata.
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
ESP32_IP        = "192.168.1.100"   # ESP32 IP (check serial monitor)
ESP32_PORT      = 8888
ROOM            = "room1"           # Room name
START_DIR       = "west"            # Starting direction
OUTPUT_DIR      = "data"            # Base output directory
IMG_SIZE        = 48                # Resize to NxN
SHOW_PREVIEW    = True              # Live OpenCV preview
PHOTOS_PER_DIR  = 3                 # Must match ESP32 COLLECT_PHOTOS_PER_DIR
# ═══════════════════════════════════════════════

# Protocol (must match wifi_stream.h v3)
MAGIC = 0x494D4733           # "IMG3"
HEADER_SIZE = 17             # 4+4+1+4+4

DIR_NAMES = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


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
    # Create output directory
    run_name = f"{ROOM}_start_{START_DIR}"
    out_dir = os.path.join(OUTPUT_DIR, run_name)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    print(f"Output: {out_dir}")
    print(f"Connecting to {ESP32_IP}:{ESP32_PORT}...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(15)
    sock.connect((ESP32_IP, ESP32_PORT))
    print("Connected!\n")

    # Metadata
    metadata = {
        "room": ROOM,
        "start_direction": START_DIR,
        "run_name": run_name,
        "img_size": IMG_SIZE,
        "created": datetime.now().isoformat(),
        "frames": [],
    }

    frame_count = 0
    position = 0
    last_dir = -1
    t_start = time.time()

    try:
        while True:
            # ── Receive header ──
            header = recv_exact(sock, HEADER_SIZE)
            magic, frame_id, dir_index, heading_deg, jpeg_len = \
                struct.unpack("<IIBfI", header)

            if magic != MAGIC:
                print(f"  [WARN] Bad magic 0x{magic:08X}, skipping...")
                continue

            # Track position changes (new sweep starts when dir wraps to 0)
            if dir_index == 0 and last_dir > 0:
                position += 1
            last_dir = dir_index

            # ── Receive JPEG ──
            jpeg_data = recv_exact(sock, jpeg_len)

            # ── Decode & resize ──
            img = Image.open(BytesIO(jpeg_data)).convert("RGB")
            img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

            # ── Save image ──
            dir_name = DIR_NAMES[dir_index]
            photo_idx = frame_count % PHOTOS_PER_DIR
            filename = f"pos{position:03d}_{dir_name}_{photo_idx}.jpg"
            filepath = os.path.join(img_dir, filename)
            img_resized.save(filepath, quality=95)

            # ── Record metadata ──
            metadata["frames"].append({
                "frame_id": int(frame_id),
                "position": position,
                "dir_index": int(dir_index),
                "dir_name": dir_name,
                "photo_idx": photo_idx,
                "heading_deg": round(float(heading_deg), 2),
                "filename": filename,
                "jpeg_len": int(jpeg_len),
            })

            frame_count += 1

            # ── Console output ──
            elapsed = time.time() - t_start
            fps = frame_count / elapsed if elapsed > 0 else 0
            if frame_count % PHOTOS_PER_DIR == 0:
                print(f"  pos{position:03d}_{dir_name}: "
                      f"heading={heading_deg:6.1f}° "
                      f"({frame_count} frames, {fps:.1f} fps)")

            # ── Live preview ──
            if HAS_CV2 and SHOW_PREVIEW:
                arr = np.array(img_resized)
                display = cv2.resize(arr, (384, 384),
                                     interpolation=cv2.INTER_NEAREST)
                display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
                label = (f"Pos:{position} Dir:{dir_name} "
                         f"Hdg:{heading_deg:.0f}")
                cv2.putText(display, label, (8, 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("Data Collection", display)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    except ConnectionError as e:
        print(f"\nConnection lost: {e}")
    finally:
        sock.close()
        if HAS_CV2 and SHOW_PREVIEW:
            cv2.destroyAllWindows()

        # Save metadata
        metadata["total_frames"] = frame_count
        metadata["total_positions"] = position + 1
        meta_path = os.path.join(out_dir, "metadata.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\n{'═' * 40}")
        print(f"  Frames:    {frame_count}")
        print(f"  Positions: {position + 1}")
        print(f"  Saved to:  {out_dir}")
        print(f"  Metadata:  {meta_path}")
        print(f"{'═' * 40}")


if __name__ == "__main__":
    main()
