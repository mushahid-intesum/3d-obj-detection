# Phase 4: Integration Wiring & Test Guide

## Hardware Components

| # | Board | Role |
|---|---|---|
| 1 | **ESP32-S3 #1** (Freenove WROOM, w/ camera) | Camera Board — captures frames, depth guard, SPI master, WiFi streaming |
| 2 | **ESP32-S3 #2** (any ESP32-S3) | Motor Board — motors, pink noise exploration, IMU reader, SPI slave |
| 3 | **Arduino Nano 33 BLE Rev2** | IMU heading provider (BMI270 + BMM150), sends H:\<heading\>\n at 50Hz |
| 4 | **DRV8833** | Dual H-bridge motor driver |

---

## Wiring Diagram

### 1. SPI Bus — Camera Board ↔ Motor Board (5 wires)

| Signal | Camera Board (Master) | Wire | Motor Board (Slave) |
|---|---|---|---|
| MOSI | GPIO 40 | → | GPIO 15 |
| MISO | GPIO 41 | ← | GPIO 16 |
| SCLK | GPIO 42 | → | GPIO 17 |
| CS | GPIO 2 | → | GPIO 18 |
| GND | GND | ── | GND |

> [!IMPORTANT]
> Both ESP32-S3 boards are 3.3V — no level shifting needed for SPI.
> Keep SPI wires as short as possible (< 15cm) for 10 MHz.

### 2. Arduino IMU → Motor Board (2 wires)

| Signal | Arduino Nano 33 BLE Rev2 | Wire | Motor Board |
|---|---|---|---|
| UART TX | D1 (Serial1 TX) | → | GPIO 3 (UART1 RX) |
| GND | GND | ── | GND |

> [!NOTE]
> Arduino D0/D1 are Serial1 (hardware UART), NOT USB Serial.
> Both boards are 3.3V — direct connection is safe.

### 3. DRV8833 → Motor Board (6 wires + power)

| Signal | Motor Board GPIO | DRV8833 Pin |
|---|---|---|
| Motor A dir 1 | GPIO 38 | IN1 |
| Motor A dir 2 | GPIO 39 | IN2 |
| Motor B dir 1 | GPIO 40 | IN3 |
| Motor B dir 2 | GPIO 41 | IN4 |
| Motor A PWM | GPIO 42 | ENA |
| Motor B PWM | GPIO 14 | ENB |
| Motor power | — | VCC (battery) |
| Ground | GND | GND |

> [!WARNING]
> **GPIO 40-42 on Motor Board are shared between SPI slave and DRV8833!**
> This is a pin conflict. You need to choose one of:
> - **Option A**: Use different GPIOs for SPI slave (recommended — see fix below)
> - **Option B**: Use different GPIOs for motors

### Pin Conflict Resolution

The Motor Board config.h has GPIO 40/41/42 for both motors (IN3/IN4/ENA) AND the SPI
slave uses separate pins (15/16/17/18), so the **DRV8833 and SPI do NOT conflict**.
Only the motor and camera board's SPI master share the same GPIO numbers, but they are
on **different physical ESP32 boards**, so there is no conflict.

✅ **No actual conflict** — each board has its own GPIO namespace.

### 4. Camera Board WiFi

The Camera Board connects to your WiFi AP for streaming to the laptop.
Edit `cam_board/main/wifi_stream.c`:
```c
#define CONFIG_WIFI_SSID     "YourNetworkName"
#define CONFIG_WIFI_PASSWORD "YourPassword"
```

---

## Power

| Board | Power Source |
|---|---|
| Camera Board (ESP32 #1) | USB (from laptop or power bank) |
| Motor Board (ESP32 #2) | USB (separate) |
| Arduino Nano 33 BLE Rev2 | USB (separate) or 3.3V from Motor Board |
| DRV8833 + Motors | External battery (e.g., 2S LiPo 7.4V) |

> [!CAUTION]
> Do NOT power motors from ESP32 USB — current draw will brown out the MCU.
> Use a separate battery for DRV8833 VCC.

---

## Flash & Test Procedure

### Step 1: Flash Arduino IMU
```bash
# In Arduino IDE:
# 1. Open esp32/arduino/imu_heading/imu_heading.ino
# 2. Select Board: "Arduino Nano 33 BLE"
# 3. Upload
# 4. Open Serial Monitor at 115200 — should see "Heading: xxx.xx"
```

### Step 2: Flash Motor Board
```bash
cd esp32/motor_board
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyUSBx flash monitor
# Expected: "Motor Board — Collection Mode"
# Expected: "SPI slave initialized"
# Expected: "Explorer initialized: 2048 actions (FWD=xxx R=xxx L=xxx)"
# It will then block waiting for SPI from Camera Board
```

### Step 3: Flash Camera Board
```bash
cd esp32/cam_board
idf.py set-target esp32s3
idf.py build
idf.py -p /dev/ttyUSBy flash monitor
# Expected: "Camera Board — Collection Mode"
# Expected: "Camera initialized (QVGA JPEG)"
# Expected: "SPI master initialized (CLK=10000000 Hz)"
# Expected: "Connected! IP: 192.168.x.x"
# Expected: "TCP server on port 8888 — waiting for client..."
```

### Step 4: Run Receiver on Laptop
```bash
cd python
# Edit receiver.py: set ESP32_IP to the Camera Board's IP
python receiver.py
# Expected: connects, starts receiving frames
# Check: data/<session>/trajectory.jsonl is growing
# Check: data/<session>/images/ has frame_000000.jpg etc.
```

### Step 5: Collect Data (5-min test)
- Let the robot explore for 5 minutes
- Press Ctrl+C to stop receiver.py
- Verify:
  ```bash
  wc -l data/<session>/trajectory.jsonl     # should be ~600 lines (2 Hz × 300s)
  ls data/<session>/images/ | wc -l          # same count
  head -3 data/<session>/trajectory.jsonl    # check JSON structure
  ```

### Step 6: Full Collection Session (30-60 min per room)
- Place robot in room
- Start Motor Board, Camera Board, then receiver
- Let it explore for 30-60 minutes
- Repeat for different rooms / starting positions
- Naming: sessions auto-named by timestamp

---

## Validation Checklist

| # | Check | Expected | Status |
|---|---|---|---|
| 4.1 | Arduino Serial Monitor | "Heading: xxx.xx" at ~50Hz | ☐ |
| 4.2 | Motor Board serial log | "IMU ready — heading: xxx.x°" | ☐ |
| 4.3 | Motor Board serial log | "Explorer initialized: 2048 actions" | ☐ |
| 4.4 | Camera Board serial log | "Camera initialized (QVGA JPEG)" | ☐ |
| 4.5 | Camera Board serial log | "SPI master initialized" | ☐ |
| 4.6 | Camera Board serial log | "Client connected!" | ☐ |
| 4.7 | Receiver laptop | Frames arriving, trajectory.jsonl writing | ☐ |
| 4.8 | Robot behavior | Moving with pink noise pattern, turning away from walls | ☐ |
| 4.9 | Data integrity | trajectory.jsonl has heading, action, blocked fields | ☐ |
| 4.10 | OpenCV preview | Live 48×48 preview with action/heading overlay | ☐ |
