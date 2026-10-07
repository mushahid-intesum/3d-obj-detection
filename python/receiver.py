#!/usr/bin/env python3
"""
receiver.py — Laptop-side TCP client for Phase 1/2 data collection.

Connects to the ESP32-S3 TCP server, receives 48x48 RGB frames,
and displays them live. Used for hardware validation in Phase 1.
"""

import socket
import struct
import time
import os

import numpy as np

try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("[WARN] OpenCV not installed — live preview disabled.")

# ─── Configuration ───
ESP32_HOST = "192.168.1.100"    # TODO: set ESP32 IP address
ESP32_PORT = 8888
SAVE_DIR   = None               # Set to e.g. "./data/phase1_test" to save frames

# Protocol constants (must match wifi_stream.h)
MAGIC = 0x494D4731
HEADER_SIZE = 11
IMG_W, IMG_H, IMG_CH = 48, 48, 3
IMG_SIZE = IMG_W * IMG_H * IMG_CH
PACKET_SIZE = HEADER_SIZE + IMG_SIZE

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
    """Receive one frame packet from ESP32."""
    raw = recv_exact(sock, PACKET_SIZE)

    magic = struct.unpack_from('<I', raw, 0)[0]
    if magic != MAGIC:
        raise ValueError(f"Bad magic: 0x{magic:08X} (expected 0x{MAGIC:08X})")

    frame_id = struct.unpack_from('<I', raw, 4)[0]
    ultrasonic_cm = struct.unpack_from('<H', raw, 8)[0]
    last_action = raw[10]

    img_data = raw[HEADER_SIZE:]
    image = np.frombuffer(img_data, dtype=np.uint8).reshape(IMG_H, IMG_W, IMG_CH)

    return frame_id, image, ultrasonic_cm, last_action


def main():
    print(f"Connecting to {ESP32_HOST}:{ESP32_PORT}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(10)
    sock.connect((ESP32_HOST, ESP32_PORT))
    print("Connected!")

    meta_file = None
    if SAVE_DIR:
        os.makedirs(SAVE_DIR, exist_ok=True)
        metadata_path = os.path.join(SAVE_DIR, "metadata.csv")
        meta_file = open(metadata_path, "w")
        meta_file.write("frame_id,timestamp,ultrasonic_cm,action\n")
        print(f"Saving frames to: {SAVE_DIR}")

    print(f"\nReceiving frames... Press Ctrl+C to stop.\n")
    print(f"{'Frame':>8} | {'Dist(cm)':>8} | {'Action':>8} | {'FPS':>6}")
    print("-" * 45)

    frame_count = 0
    t_start = time.time()

    try:
        while True:
            frame_id, image, dist_cm, action = receive_frame(sock)
            frame_count += 1

            elapsed = time.time() - t_start
            fps = frame_count / elapsed if elapsed > 0 else 0

            if frame_count % 10 == 0:
                action_name = ACTION_NAMES.get(action, "?")
                dist_str = str(dist_cm) if dist_cm < 65535 else "N/A"
                print(f"{frame_id:>8} | {dist_str:>8} | {action_name:>8} | {fps:>6.1f}")

            if SAVE_DIR:
                img_path = os.path.join(SAVE_DIR, f"frame_{frame_id:06d}.npy")
                np.save(img_path, image)
                meta_file.write(f"{frame_id},{time.time():.3f},{dist_cm},{action}\n")
                meta_file.flush()

            if HAS_CV2:
                display = cv2.resize(image, (384, 384), interpolation=cv2.INTER_NEAREST)
                display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)
                info = f"Frame:{frame_id} Dist:{dist_cm}cm Act:{ACTION_NAMES.get(action, '?')}"
                cv2.putText(display, info, (10, 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                cv2.imshow("MCU ImageNav - Live", display)
                if cv2.waitKey(1) & 0xFF == ord('q'):
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


if __name__ == "__main__":
    main()
