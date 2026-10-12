/**
 * @file sonar.h
 * @brief HC-SR04 ultrasonic distance sensor driver.
 *
 * Measures distance by sending a 10µs trigger pulse and timing the
 * echo response. Thread-safe — uses a spinlock for GPIO timing.
 *
 * Wiring:
 *   TRIG → ESP32 GPIO (output, 3.3V is fine — HC-SR04 triggers on ≥2.5V)
 *   ECHO → voltage divider → ESP32 GPIO (input, must be ≤3.3V)
 *   VCC  → 5V supply
 *   GND  → common ground with ESP32
 */
#ifndef SONAR_H
#define SONAR_H

#include "esp_err.h"
#include <stdbool.h>

/**
 * @brief Initialize the HC-SR04 sonar (configure TRIG/ECHO GPIOs).
 * @return ESP_OK on success.
 */
esp_err_t sonar_init(void);

/**
 * @brief Measure distance to nearest obstacle.
 *
 * Blocking call — takes up to ~30ms (SONAR_TIMEOUT_US).
 * Safe to call from any task.
 *
 * @return Distance in centimeters (0.0 – SONAR_MAX_RANGE_CM).
 *         Returns SONAR_MAX_RANGE_CM on timeout (no echo / out of range).
 */
float sonar_get_distance_cm(void);

/**
 * @brief Check if an obstacle is within the threshold distance.
 *
 * Equivalent to: sonar_get_distance_cm() < SONAR_THRESHOLD_CM
 *
 * @return true if obstacle detected within SONAR_THRESHOLD_CM.
 */
bool sonar_is_blocked(void);

#endif /* SONAR_H */
