/**
 * @file imu_uart.h
 * @brief IMU heading reader via UART from Arduino Nano 33 BLE Rev2.
 *
 * The Arduino sends "H:<heading>\n" at 50Hz over UART.
 * This driver parses headings in a background task and provides
 * thread-safe access to the latest heading value.
 */
#ifndef IMU_UART_H
#define IMU_UART_H

#include "esp_err.h"
#include <stdbool.h>

/**
 * @brief Initialize UART and start the heading reader task.
 * @return ESP_OK on success.
 */
esp_err_t imu_init(void);

/**
 * @brief Get the latest heading from the IMU.
 * @return Heading in degrees [0, 360). Returns -1 if no reading yet.
 */
float imu_get_heading(void);

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
