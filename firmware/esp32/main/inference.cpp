

#include "inference.h"
#include "config.h"
#include "image_proc.h"
#include "esp_log.h"
#include "esp_heap_caps.h"

#include <cstring>
#include <cstdlib>
#include <cmath>


#ifndef INFERENCE_HAS_MODEL
#define INFERENCE_HAS_MODEL  0
#endif

#if INFERENCE_HAS_MODEL

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "encoder_model.h"
#include "policy_model.h"
#endif

static const char *TAG = "inference";


#define CORR_POOL_SIZE   4
#define CORR_POSITIONS   (CORR_POOL_SIZE * CORR_POOL_SIZE)
#define CORR_FLAT_DIM    (CORR_POSITIONS * CORR_POSITIONS + 2)
#define CUE_DIM          (CORR_FLAT_DIM + 1)
#define PROX_NORM_MAX    255.0f


#define FEAT_DIM         1024
#define FEAT_H           6
#define FEAT_W           6
#define FEAT_SIZE        (FEAT_DIM * FEAT_H * FEAT_W)


#define ENCODER_ARENA_SIZE  (384 * 1024)
#define POLICY_ARENA_SIZE   (32  * 1024)


static bool s_initialized = false;

#if INFERENCE_HAS_MODEL
static tflite::MicroInterpreter *s_enc_interp = nullptr;
static tflite::MicroInterpreter *s_pol_interp = nullptr;
static uint8_t *s_enc_arena = nullptr;
static uint8_t *s_pol_arena = nullptr;


static float s_goal_pooled[CORR_POSITIONS * FEAT_DIM];
static bool  s_goal_cached = false;
#endif


#if INFERENCE_HAS_MODEL


static tflite::MicroMutableOpResolver<12> s_resolver;

static void register_ops(void)
{

    s_resolver.AddConv2D();
    s_resolver.AddDepthwiseConv2D();
    s_resolver.AddRelu();
    s_resolver.AddAdd();
    s_resolver.AddMean();
    s_resolver.AddReshape();


    s_resolver.AddFullyConnected();


    s_resolver.AddQuantize();
    s_resolver.AddDequantize();
}

#endif


static void l2_normalize(float *vec, int len)
{
    float sum_sq = 0.0f;
    for (int i = 0; i < len; i++) {
        sum_sq += vec[i] * vec[i];
    }
    float inv_norm = (sum_sq > 1e-12f) ? (1.0f / sqrtf(sum_sq)) : 0.0f;
    for (int i = 0; i < len; i++) {
        vec[i] *= inv_norm;
    }
}


static void adaptive_avg_pool_and_transpose(const float *features, float *pooled)
{
    const int ps = CORR_POOL_SIZE;


    for (int py = 0; py < ps; py++) {
        int y0 = (py * FEAT_H) / ps;
        int y1 = ((py + 1) * FEAT_H) / ps;
        if (y1 == y0) y1 = y0 + 1;

        for (int px = 0; px < ps; px++) {
            int x0 = (px * FEAT_W) / ps;
            int x1 = ((px + 1) * FEAT_W) / ps;
            if (x1 == x0) x1 = x0 + 1;

            int out_pos = py * ps + px;
            int count = (y1 - y0) * (x1 - x0);
            float inv_count = 1.0f / (float)count;

            for (int c = 0; c < FEAT_DIM; c++) {
                float sum = 0.0f;
                for (int y = y0; y < y1; y++) {
                    for (int x = x0; x < x1; x++) {
                        sum += features[c * FEAT_H * FEAT_W + y * FEAT_W + x];
                    }
                }
                pooled[out_pos * FEAT_DIM + c] = sum * inv_count;
            }
        }
    }
}


static void compute_correlation(const float *goal_pooled, float *obs_pooled,
                                float *cue_out)
{
    const int ps = CORR_POOL_SIZE;
    const int N = CORR_POSITIONS;


    for (int i = 0; i < N; i++) {
        l2_normalize(&obs_pooled[i * FEAT_DIM], FEAT_DIM);
    }


    int idx = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            float dot = 0.0f;
            const float *g = &goal_pooled[i * FEAT_DIM];
            const float *o = &obs_pooled[j * FEAT_DIM];
            for (int c = 0; c < FEAT_DIM; c++) {
                dot += g[c] * o[c];
            }
            cue_out[idx++] = dot;
        }
    }


    float g_left[FEAT_DIM], g_right[FEAT_DIM];
    float o_left[FEAT_DIM], o_right[FEAT_DIM];
    memset(g_left,  0, sizeof(g_left));
    memset(g_right, 0, sizeof(g_right));
    memset(o_left,  0, sizeof(o_left));
    memset(o_right, 0, sizeof(o_right));

    float inv_ps = 1.0f / (float)ps;
    for (int row = 0; row < ps; row++) {
        int li = row * ps;
        int ri = row * ps + ps - 1;
        for (int c = 0; c < FEAT_DIM; c++) {
            g_left[c]  += goal_pooled[li * FEAT_DIM + c] * inv_ps;
            g_right[c] += goal_pooled[ri * FEAT_DIM + c] * inv_ps;
            o_left[c]  += obs_pooled[li * FEAT_DIM + c]  * inv_ps;
            o_right[c] += obs_pooled[ri * FEAT_DIM + c]  * inv_ps;
        }
    }

    float left_sim = 0.0f, right_sim = 0.0f;
    for (int c = 0; c < FEAT_DIM; c++) {
        left_sim  += g_left[c]  * o_left[c];
        right_sim += g_right[c] * o_right[c];
    }
    cue_out[idx++] = left_sim;
    cue_out[idx++] = right_sim;

}


esp_err_t inference_init(void)
{
    if (s_initialized) return ESP_OK;

#if !INFERENCE_HAS_MODEL
    ESP_LOGW(TAG, "No model embedded (INFERENCE_HAS_MODEL=0) — collection mode.");
    return ESP_ERR_NOT_FOUND;
#else

    register_ops();


    s_enc_arena = (uint8_t *)heap_caps_malloc(ENCODER_ARENA_SIZE, MALLOC_CAP_SPIRAM);
    s_pol_arena = (uint8_t *)heap_caps_malloc(POLICY_ARENA_SIZE, MALLOC_CAP_SPIRAM);
    if (!s_enc_arena || !s_pol_arena) {
        ESP_LOGE(TAG, "Failed to allocate PSRAM for TFLite arenas");
        return ESP_ERR_NO_MEM;
    }


    const tflite::Model *enc_model = tflite::GetModel(encoder_model);
    if (enc_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Encoder model schema mismatch: got %lu, want %d",
                 (unsigned long)enc_model->version(), TFLITE_SCHEMA_VERSION);
        return ESP_FAIL;
    }

    s_enc_interp = new tflite::MicroInterpreter(
        enc_model, s_resolver, s_enc_arena, ENCODER_ARENA_SIZE);
    if (s_enc_interp->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "Encoder AllocateTensors() failed");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "Encoder loaded: arena %d/%d bytes used",
             (int)s_enc_interp->arena_used_bytes(), ENCODER_ARENA_SIZE);


    const tflite::Model *pol_model = tflite::GetModel(policy_model);
    if (pol_model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Policy model schema mismatch");
        return ESP_FAIL;
    }

    s_pol_interp = new tflite::MicroInterpreter(
        pol_model, s_resolver, s_pol_arena, POLICY_ARENA_SIZE);
    if (s_pol_interp->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "Policy AllocateTensors() failed");
        return ESP_FAIL;
    }
    ESP_LOGI(TAG, "Policy loaded: arena %d/%d bytes used",
             (int)s_pol_interp->arena_used_bytes(), POLICY_ARENA_SIZE);

    s_initialized = true;
    s_goal_cached = false;
    ESP_LOGI(TAG, "Inference ready (encoder + policy)");
    return ESP_OK;

#endif
}


#if INFERENCE_HAS_MODEL
static esp_err_t run_encoder(const uint8_t *rgb888_img, float *features_out)
{
    TfLiteTensor *input = s_enc_interp->input(0);


    float in_scale = input->params.scale;
    int32_t in_zp = input->params.zero_point;
    int8_t *in_data = input->data.int8;
    int total_pixels = IMG_TARGET_W * IMG_TARGET_H * IMG_TARGET_CH;

    for (int i = 0; i < total_pixels; i++) {

        float val = (float)rgb888_img[i] / 255.0f;
        int32_t q = (int32_t)roundf(val / in_scale) + in_zp;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        in_data[i] = (int8_t)q;
    }


    if (s_enc_interp->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Encoder Invoke() failed");
        return ESP_FAIL;
    }


    TfLiteTensor *output = s_enc_interp->output(0);
    float out_scale = output->params.scale;
    int32_t out_zp = output->params.zero_point;
    int8_t *out_data = output->data.int8;
    int total_feats = FEAT_DIM * FEAT_H * FEAT_W;

    for (int i = 0; i < total_feats; i++) {
        features_out[i] = ((float)out_data[i] - (float)out_zp) * out_scale;
    }

    return ESP_OK;
}
#endif


esp_err_t inference_set_goal(const uint8_t *goal_img)
{
#if !INFERENCE_HAS_MODEL
    (void)goal_img;
    return ESP_ERR_INVALID_STATE;
#else
    if (!s_initialized) return ESP_ERR_INVALID_STATE;


    float *raw_feats = (float *)heap_caps_malloc(
        FEAT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!raw_feats) {
        ESP_LOGE(TAG, "Failed to allocate temp features buffer");
        return ESP_ERR_NO_MEM;
    }


    esp_err_t err = run_encoder(goal_img, raw_feats);
    if (err != ESP_OK) {
        free(raw_feats);
        return err;
    }


    adaptive_avg_pool_and_transpose(raw_feats, s_goal_pooled);


    for (int i = 0; i < CORR_POSITIONS; i++) {
        l2_normalize(&s_goal_pooled[i * FEAT_DIM], FEAT_DIM);
    }

    free(raw_feats);
    s_goal_cached = true;
    ESP_LOGI(TAG, "Goal features cached (%d positions × %d channels)",
             CORR_POSITIONS, FEAT_DIM);
    return ESP_OK;

#endif
}


esp_err_t inference_run_policy(const uint8_t *obs_img, float proximity,
                               int8_t *action_logits)
{
#if !INFERENCE_HAS_MODEL
    (void)obs_img;
    (void)proximity;
    (void)action_logits;
    return ESP_ERR_INVALID_STATE;
#else
    if (!s_initialized) return ESP_ERR_INVALID_STATE;
    if (!s_goal_cached) {
        ESP_LOGE(TAG, "Goal not set — call inference_set_goal() first");
        return ESP_ERR_INVALID_STATE;
    }


    float *raw_feats = (float *)heap_caps_malloc(
        FEAT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!raw_feats) return ESP_ERR_NO_MEM;

    esp_err_t err = run_encoder(obs_img, raw_feats);
    if (err != ESP_OK) {
        free(raw_feats);
        return err;
    }


    float *obs_pooled = (float *)heap_caps_malloc(
        CORR_POSITIONS * FEAT_DIM * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!obs_pooled) {
        free(raw_feats);
        return ESP_ERR_NO_MEM;
    }
    adaptive_avg_pool_and_transpose(raw_feats, obs_pooled);
    free(raw_feats);


    float cue[CUE_DIM];
    compute_correlation(s_goal_pooled, obs_pooled, cue);
    free(obs_pooled);


    float prox_norm = proximity / PROX_NORM_MAX;
    if (prox_norm > 1.0f) prox_norm = 1.0f;
    if (prox_norm < 0.0f) prox_norm = 0.0f;
    cue[CORR_FLAT_DIM] = prox_norm;


    TfLiteTensor *pol_input = s_pol_interp->input(0);
    float pol_in_scale = pol_input->params.scale;
    int32_t pol_in_zp = pol_input->params.zero_point;
    int8_t *pol_in_data = pol_input->data.int8;

    for (int i = 0; i < CUE_DIM; i++) {
        int32_t q = (int32_t)roundf(cue[i] / pol_in_scale) + pol_in_zp;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        pol_in_data[i] = (int8_t)q;
    }

    if (s_pol_interp->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Policy Invoke() failed");
        return ESP_FAIL;
    }


    TfLiteTensor *pol_output = s_pol_interp->output(0);
    memcpy(action_logits, pol_output->data.int8, NUM_ACTIONS * sizeof(int8_t));

    return ESP_OK;

#endif
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
