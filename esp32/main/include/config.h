/**
 * @file config.h
 * @brief Unified hardware configuration for MCU ImageNav robot.
 *
 * Pin assignments based on:
 *   - Freenove ESP32-S3 WROOM board pinout
 *   - espressif/esp32-camera library reference
 *
 * Camera: OV3660 via board camera connector (PWDN not used on Freenove)
 */
#ifndef CONFIG_H
#define CONFIG_H

/* ═══════════════════════════════════════════════════════════════════════════
 *  OV3660 Camera — Freenove ESP32-S3 WROOM
 *
 *  Freenove board does NOT use PWDN/RESET — both set to -1.
 *  Pin mapping verified against Freenove schematics and esp32-camera lib.
 * ═══════════════════════════════════════════════════════════════════════════ */
#define CAM_PIN_PWDN    (-1)           /* Not used on Freenove     */
#define CAM_PIN_RESET   (-1)           /* Software reset           */
#define CAM_PIN_XCLK    15
#define CAM_PIN_SIOD    4              /* I2C SDA (SCCB)           */
#define CAM_PIN_SIOC    5              /* I2C SCL (SCCB)           */
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

#define CAM_XCLK_FREQ   20000000     /* 20 MHz                   */
#define CAM_FB_COUNT     1           /* single buffer for safety  */
#define CAM_JPEG_QUALITY 12          /* 0-63, lower = better     */

/* ═══════════════════════════════════════════════════════════════════════════
 *  DRV8833 Motor Driver
 *
 *  Motor A (left):  IN1/IN2 direction, ENA speed (PWM)
 *  Motor B (right): IN3/IN4 direction, ENB speed (PWM)
 *
 *  Avoids: camera pins (4-13, 15-18), UART (43-44), flash LED (2)
 * ═══════════════════════════════════════════════════════════════════════════ */
#define MOTOR_IN1       35             /* Motor A direction 1      */
#define MOTOR_IN2       36             /* Motor A direction 2      */
#define MOTOR_IN3       37             /* Motor B direction 1      */
#define MOTOR_IN4       38             /* Motor B direction 2      */
#define MOTOR_ENA       39             /* Motor A PWM enable       */
#define MOTOR_ENB       40             /* Motor B PWM enable       */

/* PWM config */
#define MOTOR_PWM_FREQ_HZ  1000
#define MOTOR_PWM_BITS     8         /* 8-bit resolution = 0-255  */
#define MOTOR_SPEED        100       /* PWM duty 0-255            */

/* Movement timing */
#define FORWARD_MS      500          /* ms to move one grid cell  */
#define TURN_MS         350          /* ms for 90° pivot turn     */

/* ═══════════════════════════════════════════════════════════════════════════
 *  HC-SR04 Ultrasonic Sensor
 *  Note: GPIO2 is the Freenove flash LED — use GPIO3 instead for echo.
 * ═══════════════════════════════════════════════════════════════════════════ */
#define US_TRIG_PIN     1              /* Trigger pulse            */
#define US_ECHO_PIN     3              /* Echo response            */
#define US_TIMEOUT_US   25000          /* Max echo wait (~4m)      */
#define US_POLL_MS      100            /* Polling interval (10 Hz) */
#define US_OBSTACLE_CM  12             /* Safety override threshold*/

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
