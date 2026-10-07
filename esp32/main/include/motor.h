/**
 * @file motor.h
 * @brief Differential-drive motor control via PWM (LEDC).
 */
#ifndef MOTOR_H
#define MOTOR_H

#include "esp_err.h"
#include <stdint.h>

/**
 * @brief Initialize motor driver GPIOs and LEDC PWM channels.
 * @return ESP_OK on success.
 */
esp_err_t motor_init(void);

/**
 * @brief Drive both motors forward at given speed.
 * @param speed  PWM duty 0-255.
 */
void motor_forward(uint8_t speed);

/**
 * @brief Turn left in place (right motor forward, left motor backward).
 * @param speed  PWM duty 0-255.
 */
void motor_turn_left(uint8_t speed);

/**
 * @brief Turn right in place (left motor forward, right motor backward).
 * @param speed  PWM duty 0-255.
 */
void motor_turn_right(uint8_t speed);

/**
 * @brief Stop both motors immediately.
 */
void motor_stop(void);

/**
 * @brief Execute a discrete navigation action.
 *
 * Actions: 0=FORWARD, 1=TURN_LEFT, 2=TURN_RIGHT, 3=STOP
 * Forward moves for ~200ms, turns rotate for ~300ms (approx 30°).
 *
 * @param action  Action index (0-3).
 * @param speed   PWM duty 0-255 (default: 150).
 */
void motor_execute_action(int action, uint8_t speed);

/* Action indices */
#define ACTION_FORWARD      0
#define ACTION_TURN_LEFT    1
#define ACTION_TURN_RIGHT   2
#define ACTION_STOP         3

#endif /* MOTOR_H */
