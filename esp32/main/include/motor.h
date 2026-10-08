/**
 * @file motor.h
 * @brief Motor control interface for DRV8833 dual H-bridge.
 *
 * Supports both open-loop (timed) and closed-loop (IMU heading) turns.
 */
#ifndef MOTOR_H
#define MOTOR_H

#include "esp_err.h"
#include <stdbool.h>
#include <stdint.h>

esp_err_t motor_init(void);
void motor_forward(uint32_t duration_ms);
void motor_reverse(uint32_t duration_ms);
void motor_turn_left(uint32_t duration_ms);
void motor_turn_right(uint32_t duration_ms);
void motor_stop(void);
void motor_execute_action(uint8_t action_id);

/**
 * @brief Turn to exact heading using IMU feedback (closed-loop).
 *
 * Pivots until the IMU heading matches target ± tolerance.
 * Falls back to open-loop if IMU is not ready.
 *
 * @param target_heading  Target heading in degrees [0, 360).
 * @param tolerance_deg   Acceptable error (e.g., 3.0 degrees).
 * @param timeout_ms      Max time to spend turning.
 * @return true if target reached, false if timed out.
 */
bool motor_turn_to_heading(float target_heading, float tolerance_deg,
                           uint32_t timeout_ms);

/**
 * @brief Turn a relative angle using IMU feedback.
 *
 * Positive = turn right, negative = turn left.
 *
 * @param delta_deg   Relative turn angle (e.g., +45 = 45° right).
 * @param tolerance   Acceptable error in degrees.
 * @param timeout_ms  Max time to spend turning.
 * @return true if target reached.
 */
bool motor_turn_relative(float delta_deg, float tolerance, uint32_t timeout_ms);

#endif /* MOTOR_H */
