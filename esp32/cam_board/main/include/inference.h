/**
 * @file inference.h
 * @brief TFLite Micro — encoder-only inference on Camera Board.
 *
 * Camera Board runs only the encoder (no correlation, no policy).
 * Policy runs on Motor Board.
 */
#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Encoder feature output size: 3x3x32 = 288 bytes. */
#define ENCODER_FEAT_SIZE   288

/** TFLite Micro arena size for encoder. */
#define TFLITE_ARENA_SIZE   (80 * 1024)   /* 80 KB */

/**
 * @brief Initialize TFLite Micro interpreter for the encoder.
 * @return ESP_OK on success, ESP_ERR_NOT_FOUND if no model embedded.
 */
esp_err_t inference_init(void);

/**
 * @brief Run the encoder on a 48x48 RGB image.
 *
 * Input:  48x48x3 uint8 image (quantized to int8 internally)
 * Output: 3x3x32 int8 feature map (288 bytes)
 *
 * @param[in]  img_rgb888    48x48x3 uint8 image.
 * @param[out] features      Output feature map, 288 int8 values.
 * @return ESP_OK on success.
 */
esp_err_t inference_run_encoder(const uint8_t *img_rgb888, int8_t *features);

/**
 * @brief Run the policy MLP (NOT available on Camera Board).
 * @return ESP_ERR_NOT_SUPPORTED always.
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
