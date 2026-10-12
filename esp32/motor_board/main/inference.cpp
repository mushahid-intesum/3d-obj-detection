/**
 * @file inference.cpp
 * @brief Inference stub — compiles without TFLite.
 *
 * Returns ESP_ERR_NOT_FOUND from inference_init(), which causes
 * main.c to select collection mode (pink noise explorer).
 *
 * When models are ready:
 *   1. Add tflite-micro component to idf_component.yml
 *   2. Replace this file with the real TFLite implementation
 *   3. Include the exported policy_model.h C header
 */
#include "inference.h"
#include "esp_log.h"

#include <string.h>

static const char *TAG = "inference";

esp_err_t inference_init(void)
{
    ESP_LOGW(TAG, "TFLite not linked — policy stub active (collection mode).");
    return ESP_ERR_NOT_FOUND;
}

esp_err_t inference_run_policy(const uint8_t *obs_img, float sonar_cm,
                               int8_t *action_logits)
{
    (void)obs_img;
    (void)sonar_cm;
    (void)action_logits;
    return ESP_ERR_INVALID_STATE;
}

esp_err_t inference_set_goal(const uint8_t *goal_img)
{
    (void)goal_img;
    return ESP_ERR_INVALID_STATE;
}

int inference_argmax_i8(const int8_t *arr, int len)
{
    int best_idx = 0;
    int8_t best_val = arr[0];
    for (int i = 1; i < len; i++) {
        if (arr[i] > best_val) {
            best_val = arr[i];
            best_idx = i;
        }
    }
    return best_idx;
}
