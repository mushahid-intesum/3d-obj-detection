/**
 * @file inference.h
 * @brief TFLite Micro inference — policy only on Motor Board.
 *
 * The Motor Board runs the policy MLP: 83-dim cue → 4 action logits.
 * The encoder runs on the Camera Board and sends features via SPI.
 */
#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>
#include "esp_err.h"
#include "correlation.h"

#ifdef __cplusplus
extern "C" {
#endif

/** TFLite Micro arena for policy (small — only 3 FC layers). */
#define TFLITE_ARENA_SIZE   (16 * 1024)   /* 16 KB */

/**
 * @brief Initialize TFLite Micro interpreter for the policy.
 * @return ESP_OK on success, ESP_ERR_NOT_FOUND if no model embedded.
 */
esp_err_t inference_init(void);

/**
 * @brief Run the policy MLP on a correlation cue.
 *
 * Input:  83-dim int8 correlation cue
 * Output: 4 int8 action logits
 *
 * @param[in]  corr_cue      83-dim int8 correlation cue.
 * @param[out] action_logits 4 int8 logits (higher = better action).
 * @return ESP_OK on success.
 */
esp_err_t inference_run_policy(const int8_t *corr_cue, int8_t *action_logits);

/**
 * @brief Find the index of the maximum value in an int8 array.
 */
int inference_argmax_i8(const int8_t *arr, int len);

#ifdef __cplusplus
}
#endif

#endif /* INFERENCE_H */
