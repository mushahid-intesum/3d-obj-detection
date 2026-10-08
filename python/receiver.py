#!/usr/bin/env python3
"""
receiver.py — TCP client for Phase 1/2 data collection.

Connects to ESP32-S3, receives IMG3 packets (JPEG + IMU heading),
decodes to 48×48 RGB, and saves organized data with metadata.

Usage:
    python receiver.py --ip 192.168.1.X --room room1 --start-dir west
"""

import socket
import struct
import time
import os
import json
import argparse
from io import BytesIO
from datetime import datetime

import numpy as np
from PIL import Image

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False

# ─── Protocol (must match wifi_stream.h v3) ───
MAGIC = 0x494D4733           # "IMG3"
HEADER_SIZE = 17             # 4+4+1+4+4
IMG_SIZE = 48

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
    parser = argparse.ArgumentParser(description="ESP32 data collection receiver")
    parser.add_argument("--ip", required=True, help="ESP32 IP address")
    parser.add_argument("--port", type=int, default=8888)
    parser.add_argument("--room", required=True, help="Room name (e.g. room1)")
    parser.add_argument("--start-dir", required=True,
                        help="Starting direction (e.g. west, north)")
    parser.add_argument("--output", default="data", help="Output base directory")
    parser.add_argument("--img-size", type=int, default=IMG_SIZE,
                        help="Resize images to NxN")
    parser.add_argument("--no-preview", action="store_true",
                        help="Disable live OpenCV preview")
    args = parser.parse_args()

    # Create output directory
    run_name = f"{args.room}_start_{args.start_dir}"
    out_dir = os.path.join(args.output, run_name)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    print(f"Output: {out_dir}")
    print(f"Connecting to {args.ip}:{args.port}...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(15)
    sock.connect((args.ip, args.port))
    print("Connected!\n")

    # Metadata
    metadata = {
        "room": args.room,
        "start_direction": args.start_dir,
        "run_name": run_name,
        "img_size": args.img_size,
        "created": datetime.now().isoformat(),
        "frames": [],
    }

    frame_count = 0
    position = 0
    last_dir = -1
    photos_per_dir = 3       # must match COLLECT_PHOTOS_PER_DIR on ESP32
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
            img_resized = img.resize((args.img_size, args.img_size), Image.LANCZOS)

            # ── Save image ──
            dir_name = DIR_NAMES[dir_index]
            # Photo index within this direction (0, 1, 2)
            photo_idx = frame_count % photos_per_dir
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
            if frame_count % photos_per_dir == 0:
                print(f"  pos{position:03d}_{dir_name}: "
                      f"heading={heading_deg:6.1f}° "
                      f"({frame_count} frames, {fps:.1f} fps)")

            # ── Live preview ──
            if HAS_CV2 and not args.no_preview:
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
        if HAS_CV2 and not args.no_preview:
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
