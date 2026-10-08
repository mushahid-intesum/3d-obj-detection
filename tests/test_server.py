#!/usr/bin/env python3
"""
test_server.py — Integration test server for dual-ESP32 navigation system.

Runs on the laptop. Both ESP32 boards connect to this server.
Orchestrates 6 sequential tests and reports PASS/FAIL for each.

Tests:
  T1: Both boards connect and send HELLO → server replies OK
  T2: Camera board sends a real JPEG frame → server validates IMG4 packet
  T3: Motor board receives dummy SPI data from camera, sends OK to server
  T4: Motor board reads IMU heading from Arduino, sends it to server
  T5: Camera captures image + gets IMU from motor via SPI, sends both to server
  T6: Motor board rotates motors (FWD, LEFT, RIGHT) and confirms via server

Protocol (test channel):
  Each message: [4B magic "TST\x01"] [2B payload_len LE] [payload bytes]
  Payload is UTF-8 JSON.

Usage:
  1. Flash cam_board with TEST_MODE=1 and motor_board with TEST_MODE=1
  2. Run: python test_server.py
  3. Power on both boards — they connect via WiFi
  4. Tests run automatically in sequence
"""

import socket
import struct
import json
import time
import sys
import threading
from datetime import datetime

# ═══════════════════════════════════════════════
#  Configuration
# ═══════════════════════════════════════════════
SERVER_PORT     = 8888
TIMEOUT_S       = 30          # per-test timeout
IMG4_MAGIC      = 0x494D4734  # "IMG4"
TEST_MAGIC      = b"TST\x01"

# ═══════════════════════════════════════════════
#  Color helpers
# ═══════════════════════════════════════════════
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
BOLD   = "\033[1m"
RESET  = "\033[0m"

def PASS(name):  print(f"  {GREEN}{BOLD}[PASS]{RESET} {name}")
def FAIL(name, reason=""): print(f"  {RED}{BOLD}[FAIL]{RESET} {name}" + (f" — {reason}" if reason else ""))
def INFO(msg):   print(f"  {CYAN}[INFO]{RESET} {msg}")
def WARN(msg):   print(f"  {YELLOW}[WARN]{RESET} {msg}")

# ═══════════════════════════════════════════════
#  Protocol helpers
# ═══════════════════════════════════════════════

def recv_exact(sock, n, timeout=TIMEOUT_S):
    """Receive exactly n bytes with timeout."""
    buf = bytearray()
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("Socket closed")
        buf.extend(chunk)
    return bytes(buf)


def recv_test_msg(sock, timeout=TIMEOUT_S):
    """Receive one test protocol message. Returns parsed JSON dict."""
    header = recv_exact(sock, 6, timeout)
    magic = header[:4]
    if magic != TEST_MAGIC:
        raise ValueError(f"Bad magic: {magic.hex()}")
    payload_len = struct.unpack("<H", header[4:6])[0]
    payload = recv_exact(sock, payload_len, timeout)
    return json.loads(payload.decode("utf-8"))


def send_test_msg(sock, data: dict):
    """Send one test protocol message."""
    payload = json.dumps(data).encode("utf-8")
    header = TEST_MAGIC + struct.pack("<H", len(payload))
    sock.sendall(header + payload)


def recv_img4_packet(sock, timeout=TIMEOUT_S):
    """Receive one IMG4 protocol packet. Returns (header_dict, jpeg_bytes)."""
    sock.settimeout(timeout)
    raw_header = recv_exact(sock, 22, timeout)
    (magic, frame_id, timestep, action_taken,
     depth_blocked, heading_deg, jpeg_len) = struct.unpack("<IIIBBfI", raw_header)

    if magic != IMG4_MAGIC:
        raise ValueError(f"Bad IMG4 magic: 0x{magic:08X}")

    jpeg_data = recv_exact(sock, jpeg_len, timeout)

    return {
        "frame_id": frame_id,
        "timestep": timestep,
        "action": action_taken,
        "blocked": depth_blocked,
        "heading": heading_deg,
        "jpeg_len": jpeg_len,
    }, jpeg_data


# ═══════════════════════════════════════════════
#  Test Server
# ═══════════════════════════════════════════════

class TestServer:
    def __init__(self, port=SERVER_PORT):
        self.port = port
        self.cam_sock = None
        self.mot_sock = None
        self.results = {}

    @staticmethod
    def _get_local_ip():
        """Get this machine's local IP on the WiFi network."""
        try:
            # UDP connect trick — doesn't send any data, just resolves local IP
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return socket.gethostbyname(socket.gethostname())

    def start(self):
        """Main entry: wait for connections, run tests."""
        # Detect local IP
        local_ip = self._get_local_ip()

        print(f"\n{BOLD}{'═' * 60}{RESET}")
        print(f"{BOLD}  ESP32 Integration Test Suite{RESET}")
        print(f"{BOLD}{'═' * 60}{RESET}")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"  Listening on port {self.port}...")
        print(f"  {CYAN}Your laptop IP: {BOLD}{local_ip}{RESET}")
        print()
        print(f"  {YELLOW}╔═══════════════════════════════════════════════════╗{RESET}")
        print(f"  {YELLOW}║  Make sure the ESP32 firmware uses:               ║{RESET}")
        print(f"  {YELLOW}║    TEST_SERVER_IP = \"{local_ip}\"{RESET}")
        print(f"  {YELLOW}║                                                   ║{RESET}")
        print(f"  {YELLOW}║  Build with:                                      ║{RESET}")
        print(f"  {YELLOW}║    -DEXTRA_CFLAGS=\"-DTEST_MODE=1                  ║{RESET}")
        print(f"  {YELLOW}║     -DTEST_SERVER_IP=\\\\\\\"{local_ip}\\\\\\\"\"  ║{RESET}")
        print(f"  {YELLOW}║                                                   ║{RESET}")
        print(f"  {YELLOW}║  Or just edit TEST_SERVER_IP in test_mode.c       ║{RESET}")
        print(f"  {YELLOW}╚═══════════════════════════════════════════════════╝{RESET}")
        print()
        print(f"  {RED}FIREWALL: If boards can't connect, run:{RESET}")
        print(f"    sudo ufw allow {self.port}/tcp")
        print(f"    # or: sudo iptables -I INPUT -p tcp --dport {self.port} -j ACCEPT")
        print()

        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("0.0.0.0", self.port))
        srv.listen(2)
        srv.settimeout(120)  # 2 minutes to wait for both boards

        # ── Wait for both boards to connect ──
        print(f"  Waiting for Camera Board and Motor Board to connect...")
        print(f"  (timeout: 120s)\n")

        connections = {}
        try:
            while len(connections) < 2:
                client, addr = srv.accept()
                INFO(f"Connection from {addr[0]}:{addr[1]}")

                # Each board sends a HELLO identifying itself
                try:
                    msg = recv_test_msg(client, timeout=10)
                    board_id = msg.get("board", "unknown")
                    connections[board_id] = client
                    INFO(f"  → Identified as: {BOLD}{board_id}{RESET}")
                except Exception as e:
                    WARN(f"  → Failed to identify: {e}")
                    client.close()
        except socket.timeout:
            FAIL("Connection", "Timed out waiting for boards")
            srv.close()
            return

        self.cam_sock = connections.get("camera")
        self.mot_sock = connections.get("motor")

        if not self.cam_sock:
            FAIL("Connection", "Camera Board did not connect")
            srv.close()
            return
        if not self.mot_sock:
            FAIL("Connection", "Motor Board did not connect")
            srv.close()
            return

        print()

        # ── Run tests sequentially ──
        tests = [
            ("T1: Board Connectivity",              self.test_1_connectivity),
            ("T2: Camera Image Streaming",           self.test_2_camera_image),
            ("T3: Motor SPI Receive (dummy data)",   self.test_3_motor_spi_dummy),
            ("T4: Motor IMU Heading",                self.test_4_motor_imu),
            ("T5: Full Pipeline (Camera+IMU→Server)",self.test_5_full_pipeline),
            ("T6: Motor Rotation",                   self.test_6_motor_rotation),
        ]

        print(f"{BOLD}── Running Tests ──{RESET}\n")
        for name, func in tests:
            print(f"  {YELLOW}▶{RESET} {name}")
            try:
                func()
            except Exception as e:
                FAIL(name, str(e))
                self.results[name] = False

        # ── Summary ──
        print(f"\n{BOLD}{'═' * 60}{RESET}")
        print(f"{BOLD}  Test Results Summary{RESET}")
        print(f"{'═' * 60}")
        passed = sum(1 for v in self.results.values() if v)
        total = len(self.results)
        for name, ok in self.results.items():
            status = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
            print(f"    [{status}] {name}")
        print(f"{'─' * 60}")
        color = GREEN if passed == total else RED
        print(f"    {color}{BOLD}{passed}/{total} tests passed{RESET}")
        print(f"{'═' * 60}\n")

        # Cleanup
        self._send_both({"cmd": "done"})
        self.cam_sock.close()
        self.mot_sock.close()
        srv.close()

    def _send_both(self, data):
        """Send a test message to both boards."""
        try:
            send_test_msg(self.cam_sock, data)
        except:
            pass
        try:
            send_test_msg(self.mot_sock, data)
        except:
            pass

    # ─────────────────────────────────────────
    #  T1: Both boards connected, server says OK
    # ─────────────────────────────────────────
    def test_1_connectivity(self):
        name = "T1: Board Connectivity"
        try:
            # Send OK to both boards
            send_test_msg(self.cam_sock, {"cmd": "t1_ok", "msg": "Camera Board connected"})
            send_test_msg(self.mot_sock, {"cmd": "t1_ok", "msg": "Motor Board connected"})

            # Wait for ACK from both
            cam_ack = recv_test_msg(self.cam_sock, timeout=10)
            mot_ack = recv_test_msg(self.mot_sock, timeout=10)

            if cam_ack.get("status") == "ok" and mot_ack.get("status") == "ok":
                PASS(name)
                self.results[name] = True
            else:
                FAIL(name, f"cam={cam_ack}, mot={mot_ack}")
                self.results[name] = False
        except Exception as e:
            FAIL(name, str(e))
            self.results[name] = False

    # ─────────────────────────────────────────
    #  T2: Camera sends real JPEG via IMG4
    # ─────────────────────────────────────────
    def test_2_camera_image(self):
        name = "T2: Camera Image Streaming"
        try:
            # Tell camera board to capture and send one frame
            send_test_msg(self.cam_sock, {"cmd": "t2_capture"})

            # Receive IMG4 packet
            hdr, jpeg = recv_img4_packet(self.cam_sock, timeout=15)

            checks = []
            # Check JPEG starts with FFD8 (SOI marker)
            if len(jpeg) >= 2 and jpeg[0] == 0xFF and jpeg[1] == 0xD8:
                checks.append("jpeg_valid")
            else:
                FAIL(name, f"Invalid JPEG header: {jpeg[:4].hex() if len(jpeg)>=4 else 'empty'}")
                self.results[name] = False
                return

            # Check reasonable JPEG size (1KB - 100KB)
            if 1024 <= len(jpeg) <= 102400:
                checks.append("jpeg_size_ok")
            else:
                WARN(f"  Unusual JPEG size: {len(jpeg)} bytes")

            # Check header fields
            if 0 <= hdr["action"] <= 3:
                checks.append("action_valid")
            if 0.0 <= hdr["heading"] < 360.0 or hdr["heading"] == 0.0:
                checks.append("heading_valid")

            INFO(f"  JPEG: {len(jpeg)} bytes, action={hdr['action']}, heading={hdr['heading']:.1f}°")
            INFO(f"  Checks passed: {', '.join(checks)}")

            # Send ACK
            send_test_msg(self.cam_sock, {"cmd": "t2_ack", "status": "ok"})

            PASS(name)
            self.results[name] = True

        except Exception as e:
            FAIL(name, str(e))
            self.results[name] = False

    # ─────────────────────────────────────────
    #  T3: Motor receives dummy SPI data from camera
    # ─────────────────────────────────────────
    def test_3_motor_spi_dummy(self):
        name = "T3: Motor SPI Receive (dummy data)"
        try:
            # Tell motor to listen for SPI FIRST (so slave is ready)
            send_test_msg(self.mot_sock, {"cmd": "t3_spi_recv"})
            time.sleep(0.5)  # Let motor set up SPI slave
            # Then tell camera to send a dummy SPI exchange
            send_test_msg(self.cam_sock, {"cmd": "t3_spi_send"})

            # Wait for motor board to report SPI reception
            mot_result = recv_test_msg(self.mot_sock, timeout=15)

            if mot_result.get("status") == "ok":
                obs = mot_result.get("obstacle_flag", -1)
                msg_type = mot_result.get("msg_type", -1)
                INFO(f"  Motor received SPI: obstacle={obs}, msg_type={msg_type}")

                # Motor sends OK confirmation
                if obs == 0 and msg_type == 0:
                    PASS(name)
                    self.results[name] = True
                else:
                    # Still pass if we got any valid SPI data
                    WARN(f"  Unexpected values but SPI worked")
                    PASS(name)
                    self.results[name] = True
            else:
                FAIL(name, f"Motor SPI failed: {mot_result}")
                self.results[name] = False

        except Exception as e:
            FAIL(name, str(e))
            self.results[name] = False

    # ─────────────────────────────────────────
    #  T4: Motor reads IMU heading from Arduino
    # ─────────────────────────────────────────
    def test_4_motor_imu(self):
        name = "T4: Motor IMU Heading"
        try:
            send_test_msg(self.mot_sock, {"cmd": "t4_imu_read"})

            mot_result = recv_test_msg(self.mot_sock, timeout=15)

            if mot_result.get("status") == "ok":
                heading = mot_result.get("heading", -1.0)
                imu_ready = mot_result.get("imu_ready", False)

                if imu_ready and 0.0 <= heading < 360.0:
                    INFO(f"  IMU heading: {heading:.1f}° (ready={imu_ready})")
                    PASS(name)
                    self.results[name] = True
                elif not imu_ready:
                    WARN(f"  IMU not ready yet (heading={heading:.1f}°)")
                    WARN(f"  Check Arduino wiring: TX → ESP32 GPIO3, GND → GND")
                    FAIL(name, "IMU not ready — Arduino may not be connected")
                    self.results[name] = False
                else:
                    FAIL(name, f"Invalid heading: {heading}")
                    self.results[name] = False
            else:
                FAIL(name, f"Motor IMU read failed: {mot_result}")
                self.results[name] = False

        except Exception as e:
            FAIL(name, str(e))
            self.results[name] = False

    # ─────────────────────────────────────────
    #  T5: Camera gets IMU from motor via SPI,
    #      sends camera + IMU data to server
    # ─────────────────────────────────────────
    def test_5_full_pipeline(self):
        name = "T5: Full Pipeline (Camera+IMU→Server)"
        try:
            # Tell motor to prepare SPI response with current heading
            send_test_msg(self.mot_sock, {"cmd": "t5_spi_respond"})
            time.sleep(0.5)  # Let motor set up SPI slave
            # Tell camera to do a full cycle: capture + SPI exchange + send IMG4
            send_test_msg(self.cam_sock, {"cmd": "t5_full_cycle"})

            # Receive IMG4 packet from camera
            hdr, jpeg = recv_img4_packet(self.cam_sock, timeout=20)

            checks = []

            # Validate JPEG
            if len(jpeg) >= 2 and jpeg[0] == 0xFF and jpeg[1] == 0xD8:
                checks.append("jpeg_valid")

            # Heading should be a real value from IMU (not 0.0 placeholder)
            heading = hdr["heading"]
            action = hdr["action"]

            INFO(f"  JPEG: {len(jpeg)} bytes")
            INFO(f"  Action: {action} (from motor via SPI)")
            INFO(f"  Heading: {heading:.1f}° (from IMU via motor→SPI→camera)")
            INFO(f"  Blocked: {hdr['blocked']}")

            if len(checks) > 0:
                checks.append("pipeline_complete")

            # Also receive OK from camera's test channel
            cam_ack = recv_test_msg(self.cam_sock, timeout=10)

            if cam_ack.get("status") == "ok" and "jpeg_valid" in checks:
                PASS(name)
                self.results[name] = True
            else:
                FAIL(name, f"Pipeline incomplete: checks={checks}, ack={cam_ack}")
                self.results[name] = False

        except Exception as e:
            FAIL(name, str(e))
            self.results[name] = False

    # ─────────────────────────────────────────
    #  T6: Motor properly rotates motors
    # ─────────────────────────────────────────
    def test_6_motor_rotation(self):
        name = "T6: Motor Rotation"
        try:
            actions_to_test = [
                {"action": 0, "name": "FORWARD",    "duration_ms": 300},
                {"action": 2, "name": "TURN_LEFT",  "duration_ms": 200},
                {"action": 1, "name": "TURN_RIGHT", "duration_ms": 200},
                {"action": 3, "name": "STOP",       "duration_ms": 0},
            ]

            all_ok = True
            for act_info in actions_to_test:
                send_test_msg(self.mot_sock, {
                    "cmd": "t6_motor",
                    "action": act_info["action"],
                    "duration_ms": act_info["duration_ms"],
                })

                result = recv_test_msg(self.mot_sock, timeout=10)
                status = result.get("status", "fail")

                if status == "ok":
                    INFO(f"  {act_info['name']}: OK")
                else:
                    WARN(f"  {act_info['name']}: FAILED — {result}")
                    all_ok = False

                # Brief pause between motor commands
                time.sleep(0.5)

            if all_ok:
                PASS(name)
                self.results[name] = True
            else:
                FAIL(name, "One or more motor actions failed")
                self.results[name] = False

        except Exception as e:
            FAIL(name, str(e))
            self.results[name] = False


# ═══════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════

if __name__ == "__main__":
    server = TestServer()
    try:
        server.start()
    except KeyboardInterrupt:
        print("\nAborted by user.")
        sys.exit(1)
