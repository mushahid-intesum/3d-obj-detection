/**
 * @file config.h
 * @brief Unified hardware configuration for MCU ImageNav robot.
 *
 * All pin assignments, timing constants, and hardware parameters.
 * Update the GPIO_NUM_XX placeholders to match your wiring.
 */
#ifndef CONFIG_H
#define CONFIG_H

#include "driver/gpio.h"
#include "driver/ledc.h"

/* ═══════════════════════════════════════════════════════════════════════════
 *  OV3660 Camera (24-pin, 160° wide-angle fisheye)
 * ═══════════════════════════════════════════════════════════════════════════ */
#define CAM_PIN_PWDN    GPIO_NUM_NC   /* -1 if not used          */
#define CAM_PIN_RESET   GPIO_NUM_NC   /* -1 if not used          */
#define CAM_PIN_XCLK    GPIO_NUM_XX   /* TODO: set pin           */
#define CAM_PIN_SIOD    GPIO_NUM_XX   /* I2C SDA (SCCB)          */
#define CAM_PIN_SIOC    GPIO_NUM_XX   /* I2C SCL (SCCB)          */
#define CAM_PIN_D7      GPIO_NUM_XX
#define CAM_PIN_D6      GPIO_NUM_XX
#define CAM_PIN_D5      GPIO_NUM_XX
#define CAM_PIN_D4      GPIO_NUM_XX
#define CAM_PIN_D3      GPIO_NUM_XX
#define CAM_PIN_D2      GPIO_NUM_XX
#define CAM_PIN_D1      GPIO_NUM_XX
#define CAM_PIN_D0      GPIO_NUM_XX
#define CAM_PIN_VSYNC   GPIO_NUM_XX
#define CAM_PIN_HREF    GPIO_NUM_XX
#define CAM_PIN_PCLK    GPIO_NUM_XX

#define CAM_XCLK_FREQ   20000000     /* 20 MHz                   */
#define CAM_FB_COUNT     2           /* double-buffer DMA        */
#define CAM_JPEG_QUALITY 15          /* 0-63, lower = better     */

/* ═══════════════════════════════════════════════════════════════════════════
 *  DRV8833 Motor Driver
 *
 *  Motor A (left):  IN1/IN2 direction, ENA speed (PWM)
 *  Motor B (right): IN3/IN4 direction, ENB speed (PWM)
 * ═══════════════════════════════════════════════════════════════════════════ */
#define MOTOR_IN1       GPIO_NUM_XX   /* Left motor dir A        */
#define MOTOR_IN2       GPIO_NUM_XX   /* Left motor dir B        */
#define MOTOR_IN3       GPIO_NUM_XX   /* Right motor dir A       */
#define MOTOR_IN4       GPIO_NUM_XX   /* Right motor dir B       */
#define MOTOR_ENA       GPIO_NUM_XX   /* Left motor PWM speed    */
#define MOTOR_ENB       GPIO_NUM_XX   /* Right motor PWM speed   */

/* PWM config */
#define LEDC_FREQ_HZ    1000
#define LEDC_RESOLUTION  LEDC_TIMER_8_BIT   /* 0-255 duty        */
#define MOTOR_SPEED     180                  /* default PWM duty  */

/* Movement timing */
#define FORWARD_MS      400          /* ms per forward step       */
#define TURN_MS         350          /* ms per pivot turn         */

/* ═══════════════════════════════════════════════════════════════════════════
 *  HC-SR04 Ultrasonic Sensor
 * ═══════════════════════════════════════════════════════════════════════════ */
#define US_TRIG_PIN     GPIO_NUM_XX   /* Trigger pulse            */
#define US_ECHO_PIN     GPIO_NUM_XX   /* Echo response            */
#define US_TIMEOUT_US   25000         /* Max echo wait (~4m)      */
#define US_POLL_MS      100           /* Polling interval (10 Hz) */
#define US_OBSTACLE_CM  12            /* Safety override threshold*/

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
 *  Status LED & Goal Button
 * ═══════════════════════════════════════════════════════════════════════════ */
#define LED_STATUS_PIN  GPIO_NUM_XX
#define BTN_GOAL_PIN    GPIO_NUM_XX

/* ═══════════════════════════════════════════════════════════════════════════
 *  Navigation Parameters
 * ═══════════════════════════════════════════════════════════════════════════ */
#define NAV_MAX_STEPS   200
#define NAV_RATE_HZ     5

#endif /* CONFIG_H */
