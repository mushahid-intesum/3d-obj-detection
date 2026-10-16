
#ifndef INFERENCE_H
#define INFERENCE_H

#include <stdint.h>
#include "esp_err.h"

#ifdef __cplusplus
extern "C" {
#endif


#define TFLITE_ARENA_SIZE   (512 * 1024)


esp_err_t inference_init(void);


esp_err_t inference_run_policy(const uint8_t *obs_img, float proximity,
                               int8_t *action_logits);


esp_err_t inference_set_goal(const uint8_t *goal_img);


int inference_argmax_i8(const int8_t *arr, int len);

#ifdef __cplusplus
}
#endif

#endif
