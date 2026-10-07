/**
 * @file motor.h
 * @brief Motor control interface for DRV8833 dual H-bridge.
 */
#ifndef MOTOR_H
#define MOTOR_H

#include "esp_err.h"
#include <stdint.h>

esp_err_t motor_init(void);
void motor_forward(uint32_t duration_ms);
void motor_reverse(uint32_t duration_ms);
void motor_turn_left(uint32_t duration_ms);
void motor_turn_right(uint32_t duration_ms);
void motor_stop(void);
void motor_execute_action(uint8_t action_id);

#endif /* MOTOR_H */
