
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


bool motor_turn_to_heading(float target_heading, float tolerance_deg,
                           uint32_t timeout_ms);


bool motor_turn_relative(float delta_deg, float tolerance, uint32_t timeout_ms);

#endif
