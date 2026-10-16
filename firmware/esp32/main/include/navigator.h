
#ifndef NAVIGATOR_H
#define NAVIGATOR_H

#include "esp_err.h"
#include <stdint.h>


typedef enum {
    NAV_GOAL_FROM_FIRST_FRAME,
    NAV_GOAL_FROM_FLASH,
} nav_goal_mode_t;


esp_err_t navigator_start(nav_goal_mode_t mode);


esp_err_t navigator_save_goal(const uint8_t *goal_img);


esp_err_t navigator_load_goal(uint8_t *goal_img);

#endif
