/**
 * @file imu_uart.h
 * @brief IMU heading + proximity reader via UART from Arduino Nano BLE Sense Rev2.
 *
 * The Arduino sends "H:<heading>,P:<proximity>\n" at 50Hz over UART.
 *   heading   = 0.00–359.99 degrees (from BMI270 + BMM150)
 *   proximity = 0–255 (from APDS9960, 0=far, 255=close)
 *
 * This driver parses both values in a background task and provides
 * thread-safe access to the latest readings.
 */
#ifndef IMU_UART_H
#define IMU_UART_H

#include "esp_err.h"
#include <stdbool.h>
#include <stdint.h>

/**
 * @brief Initialize UART and start the heading/proximity reader task.
 * @return ESP_OK on success.
 */
esp_err_t imu_init(void);

/**
 * @brief Get the latest heading from the IMU.
 * @return Heading in degrees [0, 360). Returns -1 if no reading yet.
 */
float imu_get_heading(void);

/**
 * @brief Get the latest APDS9960 proximity reading.
 * @return Proximity value [0, 255]. 0 = far, 255 = close.
 *         Returns 0 if no reading yet.
 */
uint8_t imu_get_proximity(void);

/**
 * @brief Check if proximity indicates an obstacle.
 * @return true if proximity > PROX_THRESHOLD.
 */
bool imu_proximity_blocked(void);

/**
 * @brief Check if the IMU is providing valid readings.
 */
bool imu_is_ready(void);

/**
 * @brief Compute the shortest signed angular difference.
 *
 * @param target  Target heading (degrees).
 * @param current Current heading (degrees).
 * @return Signed difference in [-180, 180]. Positive = turn right.
 */
float imu_angle_diff(float target, float current);

/**
 * @brief Normalize angle to [0, 360).
 */
float imu_normalize(float angle);

#endif /* IMU_UART_H */
