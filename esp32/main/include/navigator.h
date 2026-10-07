/**
 * @file navigator.h
 * @brief Autonomous image-goal navigation task.
 */
#ifndef NAVIGATOR_H
#define NAVIGATOR_H

#include "esp_err.h"
#include <stdint.h>

/** Navigation mode: how goal image is obtained. */
typedef enum {
    NAV_GOAL_FROM_BUTTON,   /**< Capture current frame as goal on button press. */
    NAV_GOAL_FROM_FLASH,    /**< Load pre-stored goal from flash. */
} nav_goal_mode_t;

/**
 * @brief Start the navigation task.
 *
 * Initializes inference, captures/loads goal, runs the navigation loop:
 *   1. Encode goal image once → cache features
 *   2. Loop: capture → encode obs → correlate → policy → safety check → execute
 *
 * @param mode  How to obtain the goal image.
 * @return ESP_OK on success.
 */
esp_err_t navigator_start(nav_goal_mode_t mode);

#endif /* NAVIGATOR_H */
