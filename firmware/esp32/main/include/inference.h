/**
 * @file inference.h
 * @brief TFLite Micro inference — student policy on single board.
 *
 * The student model runs end-to-end on the ESP32-S3:
 *   TinyEncoder(obs) + TinyEncoder(goal) → Correlation → Policy(cue, proximity) → 4 action logits
 *
 * During collection mode (no model flashed), inference_init() returns
 * ESP_ERR_NOT_FOUND, causing main.c to use the pink noise explorer.
 */
#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/** TFLite Micro arena — allocated in PSRAM at runtime. */
#define TFLITE_ARENA_SIZE   (512 * 1024)   /* 512 KB — full student model */

/**
 * @brief Initialize TFLite Micro interpreter for the student policy.
 * @return ESP_OK on success, ESP_ERR_NOT_FOUND if no model embedded.
 */
esp_err_t inference_init(void);

/**
 * @brief Run the full student model: image + proximity → action logits.
 *
 * In the future, this will:
 *   1. Run TinyEncoder on obs image → obs features
 *   2. Correlate obs features with cached goal features → cue
 *   3. Run TinyPolicy(cue, proximity) → action logits
 *
 * @param[in]  obs_img       96×96×3 uint8 RGB image (27648 bytes).
 * @param[in]  proximity     APDS9960 proximity value (0–255 as float).
 * @param[out] action_logits 4 int8 logits (higher = better action).
 * @return ESP_OK on success.
 */
esp_err_t inference_run_policy(const uint8_t *obs_img, float proximity,
                               int8_t *action_logits);

/**
 * @brief Cache the goal image features for navigation.
 *
 * Runs TinyEncoder on the goal image and stores the feature map.
 * Must be called once before inference_run_policy().
 *
 * @param[in]  goal_img  96×96×3 uint8 RGB image (27648 bytes).
 * @return ESP_OK on success.
 */
esp_err_t inference_set_goal(const uint8_t *goal_img);

/**
 * @brief Find the index of the maximum value in an int8 array.
 */
int inference_argmax_i8(const int8_t *arr, int len);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_H */
