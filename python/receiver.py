#!/usr/bin/env python3
"""
receiver.py — Laptop-side TCP client for Phase 1/2 data collection.

Connects to the ESP32-S3 TCP server, receives 48x48 RGB frames,
and displays them live. Used for hardware validation in Phase 1.

Usage:
    python receiver.py --host <ESP32_IP> [--port 8888] [--save-dir ./data]
"""

import argparse
import socket
import struct
import time
import os
import sys

import numpy as np

# Optional: OpenCV for live preview
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[WARN] OpenCV not installed — live preview disabled. "
          "Install with: pip install opencv-python")


# Protocol constants (must match wifi_stream.h)
MAGIC = 0x494D4731  # "IMG1"
HEADER_SIZE = 11     # 4 + 4 + 2 + 1
IMG_W, IMG_H, IMG_CH = 48, 48, 3
IMG_SIZE = IMG_W * IMG_H * IMG_CH  # 6912 bytes
PACKET_SIZE = HEADER_SIZE + IMG_SIZE  # 6923 bytes

ACTION_NAMES = {0: "FORWARD", 1: "LEFT", 2: "RIGHT", 3: "STOP"}


def recv_exact(sock, nbytes):
    """Receive exactly nbytes from socket."""
    buf = bytearray()
    while len(buf) < nbytes:
        chunk = sock.recv(nbytes - len(buf))
        if not chunk:
            raise ConnectionError("Connection closed by ESP32")
        buf.extend(chunk)
    return bytes(buf)


def receive_frame(sock):
    """
    Receive one frame packet from ESP32.

    Returns:
        frame_id (int), image (np.ndarray 48x48x3 uint8),
        ultrasonic_cm (int), last_action (int)
    """
    raw = recv_exact(sock, PACKET_SIZE)

    # Parse header
    magic = struct.unpack_from('<I', raw, 0)[0]
    if magic != MAGIC:
        raise ValueError(f"Bad magic: 0x{magic:08X} (expected 0x{MAGIC:08X})")

    frame_id = struct.unpack_from('<I', raw, 4)[0]
    ultrasonic_cm = struct.unpack_from('<H', raw, 8)[0]
    last_action = raw[10]

    # Parse image
    img_data = raw[HEADER_SIZE:]
    image = np.frombuffer(img_data, dtype=np.uint8).reshape(IMG_H, IMG_W, IMG_CH)

    return frame_id, image, ultrasonic_cm, last_action


def main():
    parser = argparse.ArgumentParser(description="MCU ImageNav — Frame Receiver")
    parser.add_argument("--host", required=True, help="ESP32 IP address")
    parser.add_argument("--port", type=int, default=8888, help="TCP port (default: 8888)")
    parser.add_argument("--save-dir", default=None,
                        help="Directory to save frames (optional, for Phase 2)")
    args = parser.parse_args()

    # Connect to ESP32
    print(f"Connecting to {args.host}:{args.port}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    sock.connect((args.host, args.port))
    print("Connected!")

    # Prepare save directory if specified
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        metadata_path = os.path.join(args.save_dir, "metadata.csv")
        meta_file = open(metadata_path, "w")
        meta_file.write("frame_id,timestamp,ultrasonic_cm,action\n")
        print(f"Saving frames to: {args.save_dir}")
    else:
        meta_file = None

    print("\nReceiving frames... Press Ctrl+C to stop.\n")
    print(f"{'Frame':>8} | {'Dist(cm)':>8} | {'Action':>8} | {'FPS':>6}")
    print("-" * 45)

    frame_count = 0
    t_start = time.time()

    try:
        while True:
            frame_id, image, dist_cm, action = receive_frame(sock)
            frame_count += 1

            # Calculate FPS
            elapsed = time.time() - t_start
            fps = frame_count / elapsed if elapsed > 0 else 0

            # Print status every 10 frames
            if frame_count % 10 == 0:
                action_name = ACTION_NAMES.get(action, "?")
                dist_str = str(dist_cm) if dist_cm < 65535 else "N/A"
                print(f"{frame_id:>8} | {dist_str:>8} | {action_name:>8} | {fps:>6.1f}")

            # Save if requested
            if args.save_dir:
                img_path = os.path.join(args.save_dir, f"frame_{frame_id:06d}.npy")
                np.save(img_path, image)
                timestamp = time.time()
                meta_file.write(f"{frame_id},{timestamp:.3f},{dist_cm},{action}\n")
                meta_file.flush()

            # Live preview with OpenCV
            if HAS_CV2:
                # Upscale for visibility and convert RGB→BGR for OpenCV
                display = cv2.resize(image, (384, 384),
                                     interpolation=cv2.INTER_NEAREST)
                display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)

                # Overlay info text
                info = f"Frame:{frame_id} Dist:{dist_cm}cm Act:{ACTION_NAMES.get(action, '?')}"
                cv2.putText(display, info, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                cv2.imshow("MCU ImageNav - Live", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    except ConnectionError as e:
        print(f"\nConnection lost: {e}")
    finally:
        sock.close()
        if meta_file:
            meta_file.close()
        if HAS_CV2:
            cv2.destroyAllWindows()

        print(f"\nTotal frames received: {frame_count}")
        if args.save_dir:
            print(f"Data saved to: {args.save_dir}")


if __name__ == "__main__":
    main()
