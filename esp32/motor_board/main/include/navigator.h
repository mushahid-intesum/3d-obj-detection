/**
 * @file navigator.h
 * @brief Autonomous image-goal navigation on Motor Board.
 *
 * In navigation mode:
 *   1. Receives encoder features from Camera Board via SPI
 *   2. Correlates with cached goal features (plain C int8)
 *   3. Runs policy MLP to select action
 *   4. Applies safety override if obstacle detected
 *   5. Executes motor action
 */
#ifndef NAVIGATOR_H
#define NAVIGATOR_H

#include "esp_err.h"
#include <stdint.h>

/** How the goal image features are obtained. */
typedef enum {
    NAV_GOAL_FROM_FIRST_FRAME,  /**< Use first SPI-received features as goal. */
    NAV_GOAL_FROM_FLASH,        /**< Load pre-stored goal features from NVS. */
} nav_goal_mode_t;

/**
 * @brief Start the navigation task.
 *
 * Replaces the exploration task. Blocks on SPI transactions from Camera Board,
 * runs correlation + policy, executes motor actions.
 *
 * @param mode  How to obtain the goal features.
 * @return ESP_OK on success.
 */
esp_err_t navigator_start(nav_goal_mode_t mode);

/**
 * @brief Store goal features to NVS for later use.
 *
 * @param features  288 bytes of int8 encoder features.
 * @return ESP_OK on success.
 */
esp_err_t navigator_save_goal(const int8_t *features);

/**
 * @brief Load goal features from NVS.
 *
 * @param features  Output buffer, 288 bytes.
 * @return ESP_OK on success.
 */
esp_err_t navigator_load_goal(int8_t *features);

#endif /* NAVIGATOR_H */
