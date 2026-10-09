/**
 * @file inference.c
 * @brief Depth guard + encoder inference stubs for Camera Board.
 *
 * Depth guard: loads INT8 TFLite model for obstacle detection.
 *   - If model/depth_guard_model.h exists → real inference
 *   - If not → stub returns "no obstacle" (collection mode)
 *
 * Encoder: remains stubbed until navigation models are ready.
 *
 * When the depth guard model is trained and exported:
 *   1. Run: python train_depth_guard.py
 *   2. It auto-generates: esp32/cam_board/main/model/depth_guard_model.h
 *   3. Uncomment #define DEPTH_GUARD_MODEL_AVAILABLE below
 *   4. Rebuild firmware
 */
#include "inference.h"
#include "esp_log.h"

#include <string.h>

static const char *TAG = "inference";

/* ══════════════════════════════════════════════════════════════════════════
 *  Depth Guard Model
 *
 *  Uncomment the next line once depth_guard_model.h has been generated
 *  by train_depth_guard.py:
 * ══════════════════════════════════════════════════════════════════════════ */
#define DEPTH_GUARD_MODEL_AVAILABLE

#ifdef DEPTH_GUARD_MODEL_AVAILABLE

/*
 * When the model is available, we use TFLite Micro to run inference.
 * This requires the tflite-micro component in idf_component.yml.
 */
#include "model/depth_guard_model.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

static uint8_t s_dg_arena[DEPTH_GUARD_ARENA_SIZE]
    __attribute__((aligned(16)));
static tflite::MicroInterpreter *s_dg_interpreter = nullptr;
static TfLiteTensor *s_dg_input = nullptr;
static TfLiteTensor *s_dg_output = nullptr;
static bool s_dg_ready = false;

esp_err_t depth_guard_init(void)
{
    ESP_LOGI(TAG, "Loading depth guard model (%d bytes)...",
             DEPTH_GUARD_MODEL_LEN);

    const tflite::Model *model =
        tflite::GetModel(depth_guard_model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Model schema version mismatch!");
        return ESP_FAIL;
    }

    /* Register only the ops used by TinyDepthNet */
    static tflite::MicroMutableOpResolver<8> resolver;
    resolver.AddConv2D();
    resolver.AddDepthwiseConv2D();
    resolver.AddFullyConnected();
    resolver.AddRelu();
    resolver.AddMean();          /* AdaptiveAvgPool → ReduceMean */
    resolver.AddReshape();       /* Flatten */
    resolver.AddQuantize();
    resolver.AddDequantize();

    static tflite::MicroInterpreter interpreter(
        model, resolver, s_dg_arena, DEPTH_GUARD_ARENA_SIZE);
    s_dg_interpreter = &interpreter;

    if (s_dg_interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "Depth guard tensor allocation failed!");
        return ESP_FAIL;
    }

    s_dg_input  = s_dg_interpreter->input(0);
    s_dg_output = s_dg_interpreter->output(0);

    ESP_LOGI(TAG, "Depth guard ready — input: [%d,%d,%d,%d] output: [%d]",
             s_dg_input->dims->data[0], s_dg_input->dims->data[1],
             s_dg_input->dims->data[2], s_dg_input->dims->data[3],
             s_dg_output->dims->data[1]);

    s_dg_ready = true;
    return ESP_OK;
}

esp_err_t depth_guard_run(const uint8_t *img_rgb888, bool *is_blocked)
{
    if (!s_dg_ready) {
        *is_blocked = false;
        return ESP_ERR_INVALID_STATE;
    }

    /*
     * Quantize uint8 [0,255] → int8 [-128,127] for the model.
     * INT8 quantization: q = (float_val / scale) + zero_point
     * For typical image models: q = uint8_val - 128
     */
    int8_t *input_data = s_dg_input->data.int8;
    for (int i = 0; i < DEPTH_GUARD_INPUT_SIZE; i++) {
        input_data[i] = (int8_t)(img_rgb888[i] - 128);
    }

    /* Run inference */
    if (s_dg_interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Depth guard inference failed!");
        *is_blocked = false;
        return ESP_FAIL;
    }

    /* Read output logit: positive → obstacle */
    int8_t logit = s_dg_output->data.int8[0];
    *is_blocked = (logit > OBSTACLE_LOGIT_THRESHOLD);

    return ESP_OK;
}

#else  /* No model available — stubs */

esp_err_t depth_guard_init(void)
{
    ESP_LOGW(TAG, "Depth guard model not embedded — stub active.");
    ESP_LOGW(TAG, "  Run: python train_depth_guard.py");
    ESP_LOGW(TAG, "  Then: #define DEPTH_GUARD_MODEL_AVAILABLE in inference.c");
    return ESP_ERR_NOT_FOUND;
}

esp_err_t depth_guard_run(const uint8_t *img_rgb888, bool *is_blocked)
{
    (void)img_rgb888;
    *is_blocked = false;  /* Safe default: no obstacle */
    return ESP_OK;
}

#endif /* DEPTH_GUARD_MODEL_AVAILABLE */


/* ══════════════════════════════════════════════════════════════════════════
 *  Encoder — stub until navigation model ready
 * ══════════════════════════════════════════════════════════════════════════ */

esp_err_t inference_init(void)
{
    ESP_LOGW(TAG, "Encoder model not linked — stub active (collection mode).");
    return ESP_ERR_NOT_FOUND;
}

esp_err_t inference_run_encoder(const uint8_t *img_rgb888, int8_t *features)
{
    (void)img_rgb888;
    (void)features;
    return ESP_ERR_INVALID_STATE;
}

esp_err_t inference_run_policy(const int8_t *corr_cue, int8_t *action_logits)
{
    (void)corr_cue;
    (void)action_logits;
    return ESP_ERR_NOT_SUPPORTED;
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
