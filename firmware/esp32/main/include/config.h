/**
 * @file config.h
 * @brief Hardware configuration — single ESP32-S3 board.
 *
 * All-in-one: OV3660 camera, DRV8833 motors, APDS9960 proximity,
 * Arduino IMU UART, WiFi streaming.
 */
#ifndef CONFIG_H
#define CONFIG_H

/* ═══════════════════════════════════════════════════════════════════════════
 *  DRV8833 Motor Driver
 *
 *  Motor A (left):  IN1/IN2 direction, ENA speed (PWM)
 *  Motor B (right): IN3/IN4 direction, ENB speed (PWM)
 * ═══════════════════════════════════════════════════════════════════════════ */
#define MOTOR_IN1       38
#define MOTOR_IN2       39
#define MOTOR_IN3       40
#define MOTOR_IN4       41
#define MOTOR_ENA       42
#define MOTOR_ENB       14

/* ═══════════════════════════════════════════════════════════════════════════
 *  OV3660 Camera (Freenove ESP32-S3 WROOM)
 * ═══════════════════════════════════════════════════════════════════════════ */
#define CAM_PIN_PWDN    (-1)
#define CAM_PIN_RESET   (-1)
#define CAM_PIN_XCLK    15
#define CAM_PIN_SIOD    4
#define CAM_PIN_SIOC    5
#define CAM_PIN_D7      16
#define CAM_PIN_D6      17
#define CAM_PIN_D5      18
#define CAM_PIN_D4      12
#define CAM_PIN_D3      10
#define CAM_PIN_D2      8
#define CAM_PIN_D1      9
#define CAM_PIN_D0      11
#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_PCLK    13

#define CAM_XCLK_FREQ   20000000
#define CAM_FB_COUNT     1
#define CAM_JPEG_QUALITY 12

/* PWM config */
#define MOTOR_PWM_FREQ_HZ  1000
#define MOTOR_PWM_BITS     8
#define LEFT_MOTOR_SPEED       100
#define RIGHT_MOTOR_SPEED      100

/* Movement timing — step-and-stop bursts */
#define STEP_BURST_MS   500       /* forward burst duration */
#define TURN_90_MS      350       /* open-loop fallback only */
#define TURN_45_MS      175       /* open-loop fallback only */
#define TURN_ANGLE_DEG  45.0f     /* degrees per turn action */
#define TURN_TOLERANCE  3.0f      /* IMU heading tolerance */
#define TURN_TIMEOUT_MS 2000      /* max turn duration */

/* ═══════════════════════════════════════════════════════════════════════════
 *  Arduino IMU (UART1)
 *
 *  Arduino Nano BLE Sense Rev2 sends "H:<heading>,P:<proximity>\n" at 50Hz.
 *  Wiring: Arduino TX (D0) → ESP32 GPIO, GND → GND.
 *
 *  WARNING: IMU_UART_RX_PIN must NOT conflict with camera pins.
 *  GPIO 13 is currently used by CAM_PIN_PCLK above — reassign one
 *  of them before flashing!
 * ═══════════════════════════════════════════════════════════════════════════ */
#define IMU_UART_RX_PIN   44    /* FIXME: conflicts with CAM_PIN_PCLK! */
#define IMU_UART_TX_PIN   21

/* ═══════════════════════════════════════════════════════════════════════════
 *  APDS9960 Proximity (via Arduino UART)
 *
 *  The Arduino Nano BLE Sense Rev2 reads its onboard APDS9960 and sends
 *  the proximity value [0–255] alongside the heading over UART.
 *  No extra GPIO pins needed — piggybacks on IMU UART.
 *
 *  Proximity: 0 = close (obstacle), 255 = far (clear).
 * ═══════════════════════════════════════════════════════════════════════════ */
#define PROX_THRESHOLD       50     /* obstacle if proximity < this (0–255) */

/* ═══════════════════════════════════════════════════════════════════════════
 *  Exploration Parameters
 * ═══════════════════════════════════════════════════════════════════════════ */
#define EXPLORE_CYCLE_MS     1500    /* post-action delay before next capture */
#define EXPLORE_SEQ_LENGTH   2048    /* pre-generated pink noise sequence */

/* ═══════════════════════════════════════════════════════════════════════════
 *  Action Space
 * ═══════════════════════════════════════════════════════════════════════════ */
#define ACTION_FORWARD    0
#define ACTION_TURN_RIGHT 1
#define ACTION_TURN_LEFT  2
#define ACTION_STOP       3
#define NUM_ACTIONS       4

/* ═══════════════════════════════════════════════════════════════════════════
 *  Navigation Parameters
 * ═══════════════════════════════════════════════════════════════════════════ */
#define NAV_MAX_STEPS   200
#define NAV_RATE_HZ     5

/* ═══════════════════════════════════════════════════════════════════════════
 *  WiFi / Streaming
 * ═══════════════════════════════════════════════════════════════════════════ */
#define STREAM_DEFAULT_PORT  8888

#endif /* CONFIG_H */
