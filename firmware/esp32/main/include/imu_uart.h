
#ifndef IMU_UART_H
#define IMU_UART_H

#include "esp_err.h"
#include <stdbool.h>
#include <stdint.h>


esp_err_t imu_init(void);


float imu_get_heading(void);


uint8_t imu_get_proximity(void);


bool imu_proximity_blocked(void);


bool imu_is_ready(void);


float imu_angle_diff(float target, float current);


float imu_normalize(float angle);

#endif
