/**
 * @file navigator.h
 * @brief Autonomous image-goal navigation (single board).
 *
 * In navigation mode:
 *   1. Captures camera frame → downsample to 96×96
 *   2. Reads sonar distance for obstacle awareness
 *   3. Runs student model: image + sonar → action logits
 *   4. Applies sonar safety override if needed
 *   5. Executes motor action
 */
#ifndef NAVIGATOR_H
#define NAVIGATOR_H

#include "esp_err.h"
#include <stdint.h>

/** How the goal image is obtained. */
typedef enum {
    NAV_GOAL_FROM_FIRST_FRAME,  /**< Use first captured frame as goal. */
    NAV_GOAL_FROM_FLASH,        /**< Load pre-stored goal image from NVS. */
} nav_goal_mode_t;

/**
 * @brief Start the navigation task.
 *
 * Replaces the exploration task. Captures frames, runs policy,
 * executes motor actions.
 *
 * @param mode  How to obtain the goal image.
 * @return ESP_OK on success.
 */
esp_err_t navigator_start(nav_goal_mode_t mode);

/**
 * @brief Store goal image to NVS for later use.
 *
 * Saves the raw 96×96×3 RGB image (27648 bytes).
 *
 * @param goal_img  96×96×3 uint8 RGB image.
 * @return ESP_OK on success.
 */
esp_err_t navigator_save_goal(const uint8_t *goal_img);

/**
 * @brief Load goal image from NVS.
 *
 * @param goal_img  Output buffer, 27648 bytes.
 * @return ESP_OK on success.
 */
esp_err_t navigator_load_goal(uint8_t *goal_img);

#endif /* NAVIGATOR_H */
