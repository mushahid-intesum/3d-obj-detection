/**
 * @file config.h
 * @brief Hardware configuration for Motor Board (ESP32 #2).
 *
 * DRV8833 motor driver, Arduino IMU UART, SPI slave pins.
 * No camera on this board.
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

/* PWM config */
#define MOTOR_PWM_FREQ_HZ  1000
#define MOTOR_PWM_BITS     8
#define MOTOR_SPEED        100

/* Movement timing */
#define FORWARD_MS      500
#define TURN_90_MS      350
#define TURN_45_MS      175

/* ═══════════════════════════════════════════════════════════════════════════
 *  Arduino IMU (UART1)
 *
 *  Arduino Nano 33 BLE Rev2 sends "H:<heading>\n" at 50Hz.
 *  Wiring: Arduino TX → ESP32 GPIO 3, GND → GND.
 * ═══════════════════════════════════════════════════════════════════════════ */
#define IMU_UART_RX_PIN   3
#define IMU_UART_TX_PIN   48

/* ═══════════════════════════════════════════════════════════════════════════
 *  SPI Slave — from Camera Board (ESP32 #1)
 *
 *  Must match Camera Board SPI master pin assignments.
 *  MOSI from master → this board receives obstacle/features.
 *  MISO from this board → sends heading + action back.
 * ═══════════════════════════════════════════════════════════════════════════ */
#define SPI_SLAVE_MOSI   15
#define SPI_SLAVE_MISO   16
#define SPI_SLAVE_CLK    17
#define SPI_SLAVE_CS     18

/* ═══════════════════════════════════════════════════════════════════════════
 *  Exploration Parameters
 * ═══════════════════════════════════════════════════════════════════════════ */
#define EXPLORE_CYCLE_MS     500     /* ~2 Hz action cycle */
#define EXPLORE_SEQ_LENGTH   2048    /* pre-generated pink noise sequence */

/* ═══════════════════════════════════════════════════════════════════════════
 *  Action Space (shared with Camera Board)
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

#endif /* CONFIG_H */
