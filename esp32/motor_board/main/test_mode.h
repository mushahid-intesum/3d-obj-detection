/**
 * @file test_mode.h
 * @brief Integration test mode for Motor Board.
 *
 * When TEST_MODE is enabled, app_main skips normal exploration/navigation
 * and instead connects to the test server to process test commands.
 */
#ifndef TEST_MODE_H
#define TEST_MODE_H

#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Run the motor board test suite.
 *
 * Connects to the test server, sends HELLO, then processes test commands.
 * This function does not return until all tests are done.
 *
 * Requires: motors initialized, IMU initialized, SPI slave initialized.
 *
 * @return ESP_OK on success.
 */
esp_err_t test_mode_run_motor(void);

#ifdef __cplusplus
}
#endif

#endif /* TEST_MODE_H */
