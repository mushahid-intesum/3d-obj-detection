/**
 * @file inference.h
 * @brief TFLite Micro inference wrapper for encoder and policy models.
 */
#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>
#include "esp_err.h"
#include "correlation.h"

/** TFLite Micro arena size for inference scratch memory. */
#define TFLITE_ARENA_SIZE   (80 * 1024)   /* 80 KB */

/**
 * @brief Initialize TFLite Micro interpreters for encoder and policy.
 *
 * Loads the INT8 models from the embedded C headers and allocates
 * the inference arena in SRAM.
 *
 * @return ESP_OK on success.
 */
esp_err_t inference_init(void);

/**
 * @brief Run the encoder on a 48x48 RGB image.
 *
 * Input:  48x48x3 uint8 image (will be quantized to int8 internally)
 * Output: 3x3x32 int8 feature map (288 bytes)
 *
 * @param[in]  img_rgb888    48x48x3 uint8 image.
 * @param[out] features      Output feature map, 288 int8 values.
 * @return ESP_OK on success.
 */
esp_err_t inference_run_encoder(const uint8_t *img_rgb888, int8_t *features);

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

#endif /* INFERENCE_H */
