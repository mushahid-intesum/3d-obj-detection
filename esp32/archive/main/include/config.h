/**
 * @file config.h
 * @brief Unified hardware configuration for MCU ImageNav robot.
 *
 * Pure constants — no driver includes. Each .c file includes its own
 * driver headers. Pin numbers are plain integers.
 *
 * Camera pins from Freenove ESP32-S3 WROOM board.
 * No ultrasonic sensor — depth is learned end-to-end via
 * depth-distilled policy (MiDaS auxiliary loss during training).
 */
#ifndef CONFIG_H
#define CONFIG_H

/* ═══════════════════════════════════════════════════════════════════════════
 *  OV3660 Camera (Freenove ESP32-S3 WROOM)
 * ═══════════════════════════════════════════════════════════════════════════ */
#define CAM_PIN_PWDN    (-1)           /* Not connected            */
#define CAM_PIN_RESET   (-1)           /* Software reset           */
#define CAM_PIN_XCLK    15
#define CAM_PIN_SIOD    4              /* I2C SDA (SCCB)           */
#define CAM_PIN_SIOC    5              /* I2C SCL (SCCB)           */
#define CAM_PIN_D7      16             /* Y9                       */
#define CAM_PIN_D6      17             /* Y8                       */
#define CAM_PIN_D5      18             /* Y7                       */
#define CAM_PIN_D4      12             /* Y6                       */
#define CAM_PIN_D3      10             /* Y5                       */
#define CAM_PIN_D2      8              /* Y4                       */
#define CAM_PIN_D1      9              /* Y3                       */
#define CAM_PIN_D0      11             /* Y2                       */
#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_PCLK    13

#define CAM_XCLK_FREQ   20000000     /* 20 MHz                   */
#define CAM_FB_COUNT     1           /* single buffer for test    */
#define CAM_JPEG_QUALITY 12          /* 0-63, lower = better     */

/* ═══════════════════════════════════════════════════════════════════════════
 *  DRV8833 Motor Driver
 *
 *  Motor A (left):  IN1/IN2 direction, ENA speed (PWM)
 *  Motor B (right): IN3/IN4 direction, ENB speed (PWM)
 *
 *  GPIOs 33-37 are PSRAM — cannot use. Using 38-42 + 14.
 * ═══════════════════════════════════════════════════════════════════════════ */
#define MOTOR_IN1       38             /* Motor A direction 1      */
#define MOTOR_IN2       39             /* Motor A direction 2      */
#define MOTOR_IN3       40             /* Motor B direction 1      */
#define MOTOR_IN4       41             /* Motor B direction 2      */
#define MOTOR_ENA       42             /* Motor A PWM enable       */
#define MOTOR_ENB       14             /* Motor B PWM enable       */

/* PWM config */
#define MOTOR_PWM_FREQ_HZ  1000
#define MOTOR_PWM_BITS     8         /* 8-bit resolution = 0-255  */
#define MOTOR_SPEED        100       /* PWM duty 0-255            */

/* Movement timing — TUNE THESE MANUALLY (no IMU) */
#define FORWARD_MS      500          /* ms to move one grid cell  */
#define TURN_90_MS      350          /* ms for 90° pivot turn     */
#define TURN_45_MS      175          /* ms for 45° pivot turn     */

/* ═══════════════════════════════════════════════════════════════════════════
 *  Data Collection Parameters
 * ═══════════════════════════════════════════════════════════════════════════ */
#define COLLECT_PHOTOS_PER_DIR   3   /* photos per direction      */
#define COLLECT_SETTLE_MS        200  /* wait after turn to settle */
#define COLLECT_PHOTO_INTERVAL_MS 150 /* between consecutive shots */
#define COLLECT_MAX_POSITIONS    100  /* auto-stop after N cells   */
#define COLLECT_NUM_DIRS         8   /* N,NE,E,SE,S,SW,W,NW      */

/* ═══════════════════════════════════════════════════════════════════════════
 *  Action Space
 * ═══════════════════════════════════════════════════════════════════════════ */
#define ACTION_NORTH     0   /* forward                           */
#define ACTION_SOUTH     1   /* reverse                           */
#define ACTION_EAST      2   /* pivot right                       */
#define ACTION_WEST      3   /* pivot left                        */
#define ACTION_STAY      4   /* stop                              */
#define ACTION_INTERACT  5   /* stop (server handles game logic)  */
#define NUM_ACTIONS      6

/* ═══════════════════════════════════════════════════════════════════════════
 *  Navigation Parameters
 * ═══════════════════════════════════════════════════════════════════════════ */
#define NAV_MAX_STEPS   200
#define NAV_RATE_HZ     5

#endif /* CONFIG_H */
