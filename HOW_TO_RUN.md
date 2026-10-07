# MCU ImageNav — How to Run Everything

> End-to-end guide: from wiring the robot to autonomous image-goal navigation.

---

## Prerequisites

### Hardware
| Part | Specification |
|---|---|
| MCU | ESP32-S3 with PSRAM |
| Camera | OV3660 3MP 160° fisheye, 24-pin |
| Motor Driver | DRV8833 dual H-bridge |
| Motors | 2× DC gear motors (differential drive) |
| Ultrasonic | HC-SR04 (with voltage divider for 3.3V echo) |
| Power | Battery pack (motor) + USB or LiPo (ESP32) |
| Button | Momentary push button (for goal capture) |

### Software
| Tool | Version |
|---|---|
| ESP-IDF | v5.x ([install guide](https://docs.espressif.com/projects/esp-idf/en/latest/esp32s3/get-started/)) |
| Python | 3.9+ |
| PyTorch | 2.x (for training, GPU recommended) |
| TensorFlow | 2.x (for TFLite conversion only) |

---

## Step 0: Pin Wiring & Configuration

### 0.1 Wire the robot

Connect all components to ESP32-S3 GPIOs. Record which GPIO you used for each signal.

### 0.2 Edit `esp32/main/include/config.h`

Replace every `GPIO_NUM_XX` with your actual pin numbers:

```c
// Example — your pins WILL be different
#define CAM_PIN_XCLK    GPIO_NUM_15
#define CAM_PIN_SIOD    GPIO_NUM_4
#define CAM_PIN_SIOC    GPIO_NUM_5
// ... etc for all camera data/sync pins ...

#define MOTOR_IN1       GPIO_NUM_36
#define MOTOR_IN2       GPIO_NUM_37
#define MOTOR_IN3       GPIO_NUM_38
#define MOTOR_IN4       GPIO_NUM_39
#define MOTOR_ENA       GPIO_NUM_40
#define MOTOR_ENB       GPIO_NUM_41

#define US_TRIG_PIN     GPIO_NUM_10
#define US_ECHO_PIN     GPIO_NUM_11

#define BTN_GOAL_PIN    GPIO_NUM_0
```

### 0.3 Set WiFi credentials

Edit `esp32/main/Kconfig.projbuild` or run `idf.py menuconfig` after building:

```
MCU ImageNav Configuration →
  WiFi SSID: your_network_name
  WiFi Password: your_password
```

---

## Step 1: Build & Flash the Firmware

### 1.1 Set firmware mode

In `esp32/main/main.c`, set the mode:

```c
// For data collection (Phase 1/2):
static const firmware_mode_t FIRMWARE_MODE = MODE_COLLECT;

// For navigation (Phase 7):
static const firmware_mode_t FIRMWARE_MODE = MODE_NAVIGATE;
```

### 1.2 Build

```bash
cd esp32
idf.py set-target esp32s3
idf.py build
```

### 1.3 Flash & Monitor

```bash
idf.py -p /dev/ttyUSB0 flash monitor
```

You should see:
```
╔══════════════════════════════════╗
║   MCU ImageNav Firmware           ║
║   Mode: COLLECT                   ║
╚══════════════════════════════════╝
Initializing camera...
Initializing ultrasonic...
Initializing motors...
Connecting to WiFi...
Connected! IP: 192.168.1.100
TCP server listening on port 8888...
```

Note the IP address printed — you need it for the Python scripts.

---

## Step 2: Validate Hardware (Phase 1)

### 2.1 Test with receiver

On your laptop:

```bash
cd python
pip install -r requirements.txt
```

Edit `receiver.py` — set `ESP32_HOST`:

```python
ESP32_HOST = "192.168.1.100"    # ← IP from Step 1.3
```

Run:

```bash
python receiver.py
```

You should see a live OpenCV window showing 48×48 upscaled camera frames and ultrasonic distance readings in the terminal. Press `q` to quit.

### 2.2 Verify motors

The firmware runs a motor self-test on boot (removed in latest version — you can add it back or test via the TCP commands). In the receiver, the robot responds to commands from `explorer.py`.

---

## Step 3: Collect Exploration Data (Phase 2)

### 3.1 Configure explorer

Edit `python/explorer.py`:

```python
ESP32_HOST          = "192.168.1.100"       # ESP32 IP
OUTPUT_DIR          = "./data/session_01"   # unique per session
DURATION_SEC        = 1800                  # 30 min
RATE_HZ             = 5.0                   # 5 frames/sec
BETA                = 1.0                   # pink noise (1.0 recommended)
SEED                = 42                    # change per session
OBSTACLE_THRESH_CM  = 12                    # ultrasonic safety threshold
```

### 3.2 Run exploration

Place the robot in the target environment (the rooms it will navigate in).

```bash
python explorer.py
```

The robot drives autonomously using pink-noise actions. The laptop logs:
```
[*] Generating 9000 pink-noise actions (beta=1.0)...
[*] Connecting to 192.168.1.100:8888...
[OK] Connected!
[*] Exploring for 1800s at 5.0Hz (9000 steps)
   Step | Action |   Dist |  Ovr |   FPS
------------------------------------------
     50 |    FWD |     45 |      |   4.9
    100 |    LFT |     23 |      |   5.0
```

Press `Ctrl+C` to stop early. Data is saved automatically.

### 3.3 Collect multiple sessions

Run 3-5 sessions with different conditions:
- Change `SEED` each time
- Change `OUTPUT_DIR` to `session_02`, `session_03`, etc.
- Vary lighting (lights on/off, curtains)
- Start from different positions

### 3.4 Merge sessions

Edit `python/merge_sessions.py`:

```python
SESSION_DIRS = [
    "./data/session_01",
    "./data/session_02",
    "./data/session_03",
]
OUTPUT_DIR = "./data/merged"
```

Run:

```bash
python merge_sessions.py
```

Output:
```
[*] Total: 142 segments, 28500 frames
[OK] Merged dataset saved to: ./data/merged
     Size: 195.3 MB
```

---

## Step 4: Build Offline RL Dataset (Phase 3)

Edit `python/hindsight_relabel.py`:

```python
MERGED_DIR  = "./data/merged"
OUTPUT_DIR  = "./data/offline_dataset"
NUM_RELABELS = 5
```

Run:

```bash
python hindsight_relabel.py
```

Output:
```
==================================================
  Phase 3: Hindsight Goal Relabeling
==================================================
[*] Loaded 28500 frames, 142 segments
[*] Total transitions: 142500
[*] Saving to: ./data/offline_dataset
```

---

## Step 5: Train Teacher Model (Phase 4)

**Requires GPU** (CUDA recommended). This runs on your training machine, not the ESP32.

Edit `python/train_teacher.py`:

```python
DATASET_DIR     = "./data/offline_dataset"
CHECKPOINT_DIR  = "./checkpoints/teacher"
TOTAL_STEPS     = 200_000      # ~2-4 hours on a single GPU
BATCH_SIZE      = 256
```

Run:

```bash
python train_teacher.py
```

Output:
```
==================================================
  Phase 4: Teacher Training (IQL)
  Device: cuda
==================================================
[*] Teacher model: 523,456 parameters
[*] Training for 200000 steps, batch_size=256

[ 500/200000] L_total=1.2345 L_v=0.34 L_q=0.45 L_pi=0.46 | 120 steps/s
...
[200000/200000]  >> Eval action accuracy: 0.782
  >> New best! Saved to best.pt
```

Best checkpoint saved to `checkpoints/teacher/best.pt`.

---

## Step 6: Distill to Student Model (Phase 5)

Edit `python/distill.py`:

```python
DATASET_DIR     = "./data/offline_dataset"
TEACHER_CKPT    = "./checkpoints/teacher/best.pt"
CHECKPOINT_DIR  = "./checkpoints/student"
TOTAL_STEPS     = 100_000      # ~30-60 min on GPU
```

Run:

```bash
python distill.py
```

Target: **>90% agreement** with teacher. Output:
```
  >> Agreement: 0.923  [FWD:0.91 LFT:0.93 RGT:0.94 STP:0.88]
  >> New best! Saved.
```

---

## Step 7: Quantize & Export (Phase 6)

Edit `python/quantize_export.py`:

```python
STUDENT_CKPT    = "./checkpoints/student/best.pt"
TEACHER_CKPT    = "./checkpoints/teacher/best.pt"
OUTPUT_DIR      = "./export"
QAT_STEPS       = 10_000
```

Run:

```bash
python quantize_export.py
```

Output:
```
── Step 2: Export ONNX ──
[OK] Encoder ONNX: ./export/encoder.onnx
[OK] Policy ONNX: ./export/policy.onnx

── Step 3: Convert to TFLite INT8 ──
[OK] TFLite INT8: ./export/encoder.tflite (5.8 KB)
[OK] TFLite INT8: ./export/policy.tflite (3.2 KB)

── Step 4: Generate C Headers ──
[OK] C header: ./export/esp32_headers/encoder_model.h (5832 bytes)
[OK] C header: ./export/esp32_headers/policy_model.h (3241 bytes)
```

---

## Step 8: Deploy to ESP32 (Phase 7)

### 8.1 Copy model headers into firmware

```bash
cp export/esp32_headers/encoder_model.h esp32/main/include/
cp export/esp32_headers/policy_model.h  esp32/main/include/
```

### 8.2 Enable model includes in `inference.cpp`

Edit `esp32/main/inference.cpp`. Uncomment the includes and delete the placeholders:

```cpp
// Uncomment these:
#include "encoder_model.h"
#include "policy_model.h"

// Delete these placeholder lines:
// static const uint8_t encoder_model[] = {0};
// static const unsigned int encoder_model_len = 0;
// static const uint8_t policy_model[] = {0};
// static const unsigned int policy_model_len = 0;
```

### 8.3 Switch to navigation mode

In `esp32/main/main.c`:

```c
static const firmware_mode_t FIRMWARE_MODE = MODE_NAVIGATE;
```

### 8.4 Build and flash

```bash
cd esp32
idf.py build
idf.py -p /dev/ttyUSB0 flash monitor
```

### 8.5 Run navigation

1. Robot boots and prints "Press GOAL button to capture target"
2. **Carry the robot to the goal location**
3. **Press the GOAL button** — robot captures and encodes the goal image
4. **Carry the robot to a different starting position**
5. The robot navigates autonomously toward the goal

Monitor output:
```
[navigator] Goal image captured and encoded!
[navigator] Starting navigation (max 200 steps, 5 Hz)...
[navigator] Step   0: action=N dist=85cm logits=[45,-12,-8,-30] 27ms
[navigator] Step  10: action=E dist=32cm logits=[-5,-10,38,-15] 26ms
[navigator] Step  20: action=N dist=65cm logits=[50,-8,-5,-20] 28ms
...
[navigator] ═══════════════════════════════════
[navigator]   Navigation complete!
[navigator]   Steps: 47 / 200
[navigator]   Stopped: policy STAY
[navigator] ═══════════════════════════════════
```

---

## Project File Reference

```
3d-obj-detection/
├── esp32/                              # ESP-IDF firmware
│   ├── CMakeLists.txt
│   ├── sdkconfig.defaults
│   └── main/
│       ├── include/
│       │   ├── config.h                # ← EDIT THIS: all pin assignments
│       │   ├── camera.h
│       │   ├── correlation.h
│       │   ├── image_proc.h
│       │   ├── inference.h
│       │   ├── motor.h
│       │   ├── navigator.h
│       │   ├── ultrasonic.h
│       │   └── wifi_stream.h
│       ├── main.c                      # ← EDIT THIS: MODE_COLLECT / MODE_NAVIGATE
│       ├── camera.c                    # OV3660: JPEG or RGB565
│       ├── motor.c                     # DRV8833: NORTH/SOUTH/EAST/WEST
│       ├── ultrasonic.c                # HC-SR04: 10Hz median-filtered polling
│       ├── image_proc.c               # 320×240 → 48×48 area-avg downscale
│       ├── wifi_stream.c              # TCP server for laptop streaming
│       ├── correlation.c              # INT8 9×9 cross-correlation (plain C)
│       ├── navigator.c                # Goal-caching navigation loop
│       ├── inference.cpp              # ← EDIT THIS: uncomment model includes
│       ├── Kconfig.projbuild          # WiFi SSID/password
│       └── idf_component.yml          # esp-camera + tflite-micro deps
│
└── python/                             # Training pipeline (runs on laptop/GPU)
    ├── requirements.txt                # pip install -r requirements.txt
    ├── receiver.py                     # Phase 1: live frame viewer
    ├── pink_noise.py                   # Pink noise generator + visualizer
    ├── explorer.py                     # Phase 2: autonomous exploration
    ├── merge_sessions.py              # Phase 2: combine sessions
    ├── hindsight_relabel.py           # Phase 3: goal relabeling → RL dataset
    ├── dataset.py                     # PyTorch Dataset + DataLoader
    ├── models.py                      # Teacher: ResNet-9 + correlation + IQL
    ├── train_teacher.py               # Phase 4: IQL offline RL training
    ├── student_model.py               # Student: ~10K param tiny model
    ├── distill.py                     # Phase 5: teacher → student distillation
    └── quantize_export.py             # Phase 6: QAT + TFLite + C headers
```

---

## Quick Command Reference

```bash
# ── Setup ──
cd python && pip install -r requirements.txt
cd esp32  && idf.py set-target esp32s3

# ── Phase 1: Validate hardware ──
# Edit config.h pins, set MODE_COLLECT in main.c
idf.py build && idf.py -p /dev/ttyUSB0 flash monitor
python receiver.py                         # live preview

# ── Phase 2: Collect data ──
python explorer.py                         # run 3-5 times, change SEED + OUTPUT_DIR
python merge_sessions.py                   # merge all sessions

# ── Phase 3: Build RL dataset ──
python hindsight_relabel.py

# ── Phase 4: Train teacher (GPU) ──
python train_teacher.py                    # ~2-4 hours

# ── Phase 5: Distill student ──
python distill.py                          # ~30-60 min

# ── Phase 6: Quantize & export ──
python quantize_export.py
cp export/esp32_headers/*.h esp32/main/include/

# ── Phase 7: Deploy ──
# Uncomment model includes in inference.cpp, set MODE_NAVIGATE in main.c
idf.py build && idf.py -p /dev/ttyUSB0 flash monitor
# Press GOAL button → robot navigates
```

---

## Troubleshooting

| Issue | Fix |
|---|---|
| Camera init fails (0x105) | Check PSRAM enabled in sdkconfig. Verify all 16 camera pin assignments. |
| No WiFi connection | Check SSID/password in Kconfig. Ensure 2.4GHz network (ESP32 doesn't support 5GHz). |
| Motor doesn't spin | Verify DRV8833 wiring (IN1-4 + ENA/ENB). Check motor power supply is separate from ESP32. |
| Ultrasonic reads UINT16_MAX | Add voltage divider on echo pin (5V→3.3V). Check trig/echo pin assignments. |
| TFLite arena allocation fails | Reduce `TFLITE_ARENA_SIZE` or ensure PSRAM is enabled for overflow. |
| Low distillation agreement (<85%) | Try progressive distillation, increase student to ~20K params, or collect more data. |
| Robot oscillates / doesn't reach goal | Tune `FORWARD_MS` and `TURN_MS` in config.h. Collect more diverse data. |
