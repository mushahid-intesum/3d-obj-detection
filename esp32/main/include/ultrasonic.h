/**
 * @file ultrasonic.h
 * @brief HC-SR04 ultrasonic distance sensor driver.
 */
#ifndef ULTRASONIC_H
#define ULTRASONIC_H

#include "esp_err.h"
#include <stdint.h>

/**
 * @brief Initialize HC-SR04 GPIO pins.
 * @return ESP_OK on success.
 */
esp_err_t ultrasonic_init(void);

/**
 * @brief Take a single distance measurement.
 *
 * Sends a 10µs trigger pulse and measures echo duration.
 * Applies median filter over 3 readings for stability.
 *
 * @param[out] distance_cm  Distance in centimeters (0-400).
 *                           Set to UINT16_MAX on timeout.
 * @return ESP_OK on success, ESP_ERR_TIMEOUT if no echo.
 */
esp_err_t ultrasonic_measure(uint16_t *distance_cm);

/**
 * @brief Get the last cached distance reading.
 *
 * Non-blocking — returns the most recent measurement
 * from the ultrasonic polling task.
 *
 * @return Distance in cm, or UINT16_MAX if no reading yet.
 */
uint16_t ultrasonic_get_cached_cm(void);

/**
 * @brief Start background polling task at 10 Hz.
 * @return ESP_OK on success.
 */
esp_err_t ultrasonic_start_task(void);

#endif /* ULTRASONIC_H */
