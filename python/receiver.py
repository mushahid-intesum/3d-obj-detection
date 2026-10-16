
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

ESP32_IP        = "192.168.68.101"
ESP32_PORT      = 8888
SESSION_NAME    = ""
OUTPUT_DIR      = "data"
IMG_SIZE        = 96
SHOW_PREVIEW    = True

MAGIC = 0x494D4735
HEADER_SIZE = 26

ACTION_NAMES = ["FWD", "RIGHT", "LEFT", "STOP"]


def recv_exact(sock, n):
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf.extend(chunk)
    return bytes(buf)


def connect_with_retry(ip, port, max_retries=10, timeout=30):
    for attempt in range(1, max_retries + 1):
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            sock.connect((ip, port))

            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)

            print(f"  Connected to {ip}:{port} (timeout={timeout}s, nodelay, keepalive)")
            return sock
        except (socket.timeout, ConnectionRefusedError, OSError) as e:
            wait = min(2 ** attempt, 30)
            print(f"  [Retry {attempt}/{max_retries}] {e} — waiting {wait}s...")
            try:
                sock.close()
            except Exception:
                pass
            time.sleep(wait)

    raise ConnectionError(f"Failed to connect after {max_retries} attempts")


def receive_loop(sock, img_dir, traj_file, start_frame=0):
    frame_count = start_frame
    override_count = 0

    while True:
        header = recv_exact(sock, HEADER_SIZE)
        (magic, frame_id, timestep, action_taken,
         prox_blocked, proximity, heading_deg, jpeg_len) = \
            struct.unpack("<IIIBBffI", header)

        if magic != MAGIC:
            print(f"  [WARN] Bad magic 0x{magic:08X}, resync...")
            continue

        jpeg_data = recv_exact(sock, jpeg_len)

        img = Image.open(BytesIO(jpeg_data)).convert("RGB")
        img_resized = img.resize((IMG_SIZE, IMG_SIZE), Image.LANCZOS)

        filename = f"frame_{frame_count:06d}.jpg"
        filepath = os.path.join(img_dir, filename)
        img_resized.save(filepath, quality=95)

        entry = {
            "t": frame_count,
            "frame": filename,
            "action": int(action_taken),
            "heading": round(float(heading_deg), 2),
            "proximity": round(float(proximity), 1),
            "blocked": bool(prox_blocked),
            "ts": round(time.time(), 3),
        }
        traj_file.write(json.dumps(entry) + "\n")
        traj_file.flush()

        if prox_blocked:
            override_count += 1

        frame_count += 1

        if frame_count % 10 == 0:
            action_str = ACTION_NAMES[action_taken] if action_taken < 4 else "?"
            print(f"  [{frame_count:>6}] "
                  f"act={action_str:<5} "
                  f"hdg={heading_deg:6.1f}° "
                  f"prox={proximity:5.0f} "
                  f"blk={prox_blocked} "
                  f"({override_count} overrides)")

        if HAS_CV2 and SHOW_PREVIEW:
            arr = np.array(img_resized)
            display = cv2.resize(arr, (384, 384),
                                 interpolation=cv2.INTER_NEAREST)
            display = cv2.cvtColor(display, cv2.COLOR_RGB2BGR)

            action_str = ACTION_NAMES[action_taken] if action_taken < 4 else "?"
            label = (f"Step:{frame_count} Act:{action_str} "
                     f"Hdg:{heading_deg:.0f} Prox:{proximity:.0f}")
            color = (0, 0, 255) if prox_blocked else (0, 255, 0)
            cv2.putText(display, label, (8, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            if prox_blocked:
                cv2.putText(display, f"BLOCKED (prox={proximity:.0f})", (8, 370),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.imshow("Exploration", display)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    return frame_count, override_count


def main():
    session_name = SESSION_NAME
    if not session_name:
        session_name = datetime.now().strftime("session_%Y%m%d_%H%M%S")

    out_dir = os.path.join(OUTPUT_DIR, session_name)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    traj_path = os.path.join(out_dir, "trajectory.jsonl")

    print(f"Session: {session_name}")
    print(f"Output:  {out_dir}")

    frame_count = 0
    override_count = 0
    t_start = time.time()

    traj_file = open(traj_path, "a")

    try:
        while True:
            print(f"\nConnecting to {ESP32_IP}:{ESP32_PORT}...")
            try:
                sock = connect_with_retry(ESP32_IP, ESP32_PORT)
            except ConnectionError as e:
                print(f"  {e}")
                break

            try:
                fc, oc = receive_loop(sock, img_dir, traj_file,
                                      start_frame=frame_count)
                frame_count = fc
                override_count += oc
                break

            except (ConnectionError, socket.timeout) as e:
                print(f"\n  Connection lost: {e}")
                print(f"  Frames so far: {frame_count}")
                print(f"  Reconnecting in 3s...")
                time.sleep(3)
            finally:
                try:
                    sock.close()
                except Exception:
                    pass

    except KeyboardInterrupt:
        print("\n\nStopped by user.")
    finally:
        traj_file.close()
        if HAS_CV2 and SHOW_PREVIEW:
            cv2.destroyAllWindows()

        elapsed = time.time() - t_start
        metadata = {
            "session_name": session_name,
            "protocol": "IMG5",
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
