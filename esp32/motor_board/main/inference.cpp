/**
 * @file inference.cpp
 * @brief TFLite Micro inference — student policy on single board.
 *
 * Runs the full student navigation model on ESP32-S3:
 *   1. TinyEncoder (TFLite INT8) on obs/goal images → feature maps
 *   2. SimplifiedCorrelation (C implementation) → 258-dim cue
 *   3. TinyPolicy (TFLite INT8) on cue + sonar → 4 action logits
 *
 * Two TFLite models are embedded as C arrays:
 *   - encoder_model[] from encoder_model.h  (obs/goal → features)
 *   - policy_model[]  from policy_model.h   (cue+sonar → logits)
 *
 * When no models are embedded (INFERENCE_HAS_MODEL == 0),
 * inference_init() returns ESP_ERR_NOT_FOUND → collection mode.
 */

#include "inference.h"
#include "config.h"
#include "image_proc.h"
#include "esp_log.h"
#include "esp_heap_caps.h"

#include <cstring>
#include <cstdlib>
#include <cmath>

/* ═══════════════════════════════════════════════════════════════════════════
 *  Model embedding flag
 *
 *  Set INFERENCE_HAS_MODEL=1 and include the C headers when models are
 *  ready.  When 0, all TFLite code is compiled out and inference_init()
 *  returns ESP_ERR_NOT_FOUND → collection mode.
 * ═══════════════════════════════════════════════════════════════════════════ */
#ifndef INFERENCE_HAS_MODEL
#define INFERENCE_HAS_MODEL  0
#endif

#if INFERENCE_HAS_MODEL
/* TFLite Micro headers — only needed when a model is embedded */
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "encoder_model.h"
#include "policy_model.h"
#endif

static const char *TAG = "inference";



/* ── Correlation constants (must match student_model.py) ── */
#define CORR_POOL_SIZE   4
#define CORR_POSITIONS   (CORR_POOL_SIZE * CORR_POOL_SIZE)   /* 16 */
#define CORR_FLAT_DIM    (CORR_POSITIONS * CORR_POSITIONS + 2)  /* 258 */
#define CUE_DIM          (CORR_FLAT_DIM + 1)                 /* 259 = corr + sonar */
#define SONAR_NORM_MAX   200.0f

/* ── Encoder feature map dimensions (must match TinyEncoder output) ── */
#define FEAT_DIM         1024
#define FEAT_H           6
#define FEAT_W           6
#define FEAT_SIZE        (FEAT_DIM * FEAT_H * FEAT_W)  /* floats */

/* ── Arena sizes ── */
#define ENCODER_ARENA_SIZE  (384 * 1024)   /* 384 KB for encoder */
#define POLICY_ARENA_SIZE   (32  * 1024)   /* 32 KB for policy MLP */

/* ═══════════════════════════════════════════════════════════════════════════
 *  Static state
 * ═══════════════════════════════════════════════════════════════════════════ */
static bool s_initialized = false;

#if INFERENCE_HAS_MODEL
static tflite::MicroInterpreter *s_enc_interp = nullptr;
static tflite::MicroInterpreter *s_pol_interp = nullptr;
static uint8_t *s_enc_arena = nullptr;
static uint8_t *s_pol_arena = nullptr;

/* Cached goal features after pooling to CORR_POOL_SIZE × CORR_POOL_SIZE */
static float s_goal_pooled[CORR_POSITIONS * FEAT_DIM];
static bool  s_goal_cached = false;
#endif

/* ═══════════════════════════════════════════════════════════════════════════
 *  TFLite Op Resolver
 *
 *  Register only the ops used by TinyEncoder + TinyPolicy.
 *  This keeps the binary small (~100 KB vs ~500 KB for AllOpsResolver).
 * ═══════════════════════════════════════════════════════════════════════════ */
#if INFERENCE_HAS_MODEL

/* Max ops needed: Conv2D, DWConv2D, Add, Relu, Mean/Pool, MatMul, FullyConnected, Reshape */
static tflite::MicroMutableOpResolver<12> s_resolver;

static void register_ops(void)
{
    /* Encoder ops (DepthwiseSeparable + BatchNorm folded into Conv) */
    s_resolver.AddConv2D();
    s_resolver.AddDepthwiseConv2D();
    s_resolver.AddRelu();
    s_resolver.AddAdd();                /* residual connections (if any) */
    s_resolver.AddMean();               /* AdaptiveAvgPool2d → ReduceMean */
    s_resolver.AddReshape();

    /* Policy ops */
    s_resolver.AddFullyConnected();

    /* Quantization support */
    s_resolver.AddQuantize();
    s_resolver.AddDequantize();
}

#endif /* INFERENCE_HAS_MODEL */

/* ═══════════════════════════════════════════════════════════════════════════
 *  Correlation (C implementation)
 *
 *  Mirrors SimplifiedCorrelation from student_model.py:
 *    1. Pool feature maps to CORR_POOL_SIZE × CORR_POOL_SIZE
 *    2. L2-normalize each spatial position
 *    3. NxN cross-correlation → CORR_POSITIONS² flat values
 *    4. Left/right similarity → 2 extra values
 *    5. Total output: CORR_FLAT_DIM = 258
 * ═══════════════════════════════════════════════════════════════════════════ */

/**
 * @brief L2-normalize a vector in-place.
 */
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

/**
 * @brief Adaptive average pool: (FEAT_DIM, FEAT_H, FEAT_W) → (FEAT_DIM, ps, ps).
 *
 * Input layout:  features[C][H][W] — channel-first from TFLite.
 * Output layout: pooled[ps*ps][C] — spatial-first for correlation.
 *
 * NOTE: TFLite INT8 output may be in NHWC format. The caller must handle
 * dequantization and layout conversion before calling this.
 *
 * @param[in]  features  Dequantized float features (C, H, W).
 * @param[out] pooled    Output buffer, size [CORR_POSITIONS * FEAT_DIM].
 */
static void adaptive_avg_pool_and_transpose(const float *features, float *pooled)
{
    const int ps = CORR_POOL_SIZE;

    /* For each output cell (py, px), average over the corresponding
     * input region.  With H=6, W=6, ps=4:
     *   bin height = 6/4 = 1.5 → bins have variable size (1 or 2 rows).
     *   We use standard floor-based bin boundaries. */
    for (int py = 0; py < ps; py++) {
        int y0 = (py * FEAT_H) / ps;
        int y1 = ((py + 1) * FEAT_H) / ps;
        if (y1 == y0) y1 = y0 + 1;  /* at least 1 row */

        for (int px = 0; px < ps; px++) {
            int x0 = (px * FEAT_W) / ps;
            int x1 = ((px + 1) * FEAT_W) / ps;
            if (x1 == x0) x1 = x0 + 1;

            int out_pos = py * ps + px;       /* [0..15] */
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

/**
 * @brief Compute simplified cross-correlation between goal and obs features.
 *
 * @param[in]  goal_pooled  Cached goal features, L2-normalized, [CORR_POSITIONS][FEAT_DIM].
 * @param[in]  obs_pooled   Observation features, [CORR_POSITIONS][FEAT_DIM]. Will be L2-normalized.
 * @param[out] cue_out      Output cue vector, size [CORR_FLAT_DIM] = 258.
 */
static void compute_correlation(const float *goal_pooled, float *obs_pooled,
                                float *cue_out)
{
    const int ps = CORR_POOL_SIZE;
    const int N = CORR_POSITIONS;

    /* L2-normalize obs (goal was normalized at set_goal time) */
    for (int i = 0; i < N; i++) {
        l2_normalize(&obs_pooled[i * FEAT_DIM], FEAT_DIM);
    }

    /* NxN cross-correlation: corr[i][j] = dot(goal[i], obs[j]) */
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
    /* idx == N*N == 256 */

    /* Left/Right similarity */
    /* left_idx  = [0, ps, 2*ps, 3*ps]   = column 0 */
    /* right_idx = [ps-1, 2*ps-1, ...]   = column ps-1 */
    float g_left[FEAT_DIM], g_right[FEAT_DIM];
    float o_left[FEAT_DIM], o_right[FEAT_DIM];
    memset(g_left,  0, sizeof(g_left));
    memset(g_right, 0, sizeof(g_right));
    memset(o_left,  0, sizeof(o_left));
    memset(o_right, 0, sizeof(o_right));

    float inv_ps = 1.0f / (float)ps;
    for (int row = 0; row < ps; row++) {
        int li = row * ps;          /* left column */
        int ri = row * ps + ps - 1; /* right column */
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
    /* idx == 258 == CORR_FLAT_DIM */
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Public API
 * ═══════════════════════════════════════════════════════════════════════════ */

esp_err_t inference_init(void)
{
    if (s_initialized) return ESP_OK;

#if !INFERENCE_HAS_MODEL
    ESP_LOGW(TAG, "No model embedded (INFERENCE_HAS_MODEL=0) — collection mode.");
    return ESP_ERR_NOT_FOUND;
#else

    register_ops();

    /* Allocate arenas in PSRAM */
    s_enc_arena = (uint8_t *)heap_caps_malloc(ENCODER_ARENA_SIZE, MALLOC_CAP_SPIRAM);
    s_pol_arena = (uint8_t *)heap_caps_malloc(POLICY_ARENA_SIZE, MALLOC_CAP_SPIRAM);
    if (!s_enc_arena || !s_pol_arena) {
        ESP_LOGE(TAG, "Failed to allocate PSRAM for TFLite arenas");
        return ESP_ERR_NO_MEM;
    }

    /* Load encoder model */
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

    /* Load policy model */
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

#endif /* INFERENCE_HAS_MODEL */
}


/**
 * @brief Run encoder and extract dequantized features.
 *
 * Handles INT8 input quantization and output dequantization.
 * Output is in CHW format (FEAT_DIM, FEAT_H, FEAT_W).
 */
#if INFERENCE_HAS_MODEL
static esp_err_t run_encoder(const uint8_t *rgb888_img, float *features_out)
{
    TfLiteTensor *input = s_enc_interp->input(0);

    /* Input quantization: uint8 RGB → INT8 quantized */
    float in_scale = input->params.scale;
    int32_t in_zp = input->params.zero_point;
    int8_t *in_data = input->data.int8;
    int total_pixels = IMG_TARGET_W * IMG_TARGET_H * IMG_TARGET_CH;

    for (int i = 0; i < total_pixels; i++) {
        /* Normalize [0,255] → [0,1], then quantize */
        float val = (float)rgb888_img[i] / 255.0f;
        int32_t q = (int32_t)roundf(val / in_scale) + in_zp;
        if (q < -128) q = -128;
        if (q > 127) q = 127;
        in_data[i] = (int8_t)q;
    }

    /* Run encoder */
    if (s_enc_interp->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Encoder Invoke() failed");
        return ESP_FAIL;
    }

    /* Dequantize output: INT8 → float */
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
#endif /* INFERENCE_HAS_MODEL */


esp_err_t inference_set_goal(const uint8_t *goal_img)
{
#if !INFERENCE_HAS_MODEL
    (void)goal_img;
    return ESP_ERR_INVALID_STATE;
#else
    if (!s_initialized) return ESP_ERR_INVALID_STATE;

    /* Allocate temp buffer for raw features in PSRAM */
    float *raw_feats = (float *)heap_caps_malloc(
        FEAT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!raw_feats) {
        ESP_LOGE(TAG, "Failed to allocate temp features buffer");
        return ESP_ERR_NO_MEM;
    }

    /* Run encoder on goal image */
    esp_err_t err = run_encoder(goal_img, raw_feats);
    if (err != ESP_OK) {
        free(raw_feats);
        return err;
    }

    /* Pool to CORR_POOL_SIZE × CORR_POOL_SIZE and transpose */
    adaptive_avg_pool_and_transpose(raw_feats, s_goal_pooled);

    /* L2-normalize each spatial position */
    for (int i = 0; i < CORR_POSITIONS; i++) {
        l2_normalize(&s_goal_pooled[i * FEAT_DIM], FEAT_DIM);
    }

    free(raw_feats);
    s_goal_cached = true;
    ESP_LOGI(TAG, "Goal features cached (%d positions × %d channels)",
             CORR_POSITIONS, FEAT_DIM);
    return ESP_OK;

#endif /* INFERENCE_HAS_MODEL */
}


esp_err_t inference_run_policy(const uint8_t *obs_img, float sonar_cm,
                               int8_t *action_logits)
{
#if !INFERENCE_HAS_MODEL
    (void)obs_img;
    (void)sonar_cm;
    (void)action_logits;
    return ESP_ERR_INVALID_STATE;
#else
    if (!s_initialized) return ESP_ERR_INVALID_STATE;
    if (!s_goal_cached) {
        ESP_LOGE(TAG, "Goal not set — call inference_set_goal() first");
        return ESP_ERR_INVALID_STATE;
    }

    /* 1. Run encoder on observation image */
    float *raw_feats = (float *)heap_caps_malloc(
        FEAT_SIZE * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!raw_feats) return ESP_ERR_NO_MEM;

    esp_err_t err = run_encoder(obs_img, raw_feats);
    if (err != ESP_OK) {
        free(raw_feats);
        return err;
    }

    /* 2. Pool and transpose obs features */
    float *obs_pooled = (float *)heap_caps_malloc(
        CORR_POSITIONS * FEAT_DIM * sizeof(float), MALLOC_CAP_SPIRAM);
    if (!obs_pooled) {
        free(raw_feats);
        return ESP_ERR_NO_MEM;
    }
    adaptive_avg_pool_and_transpose(raw_feats, obs_pooled);
    free(raw_feats);

    /* 3. Compute correlation cue (258 dims) */
    float cue[CUE_DIM];
    compute_correlation(s_goal_pooled, obs_pooled, cue);
    free(obs_pooled);

    /* 4. Append normalized sonar → cue[258] */
    float sonar_norm = sonar_cm / SONAR_NORM_MAX;
    if (sonar_norm > 1.0f) sonar_norm = 1.0f;
    if (sonar_norm < 0.0f) sonar_norm = 0.0f;
    cue[CORR_FLAT_DIM] = sonar_norm;

    /* 5. Run policy MLP: cue[259] → logits[4] */
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

    /* 6. Copy INT8 output logits */
    TfLiteTensor *pol_output = s_pol_interp->output(0);
    memcpy(action_logits, pol_output->data.int8, NUM_ACTIONS * sizeof(int8_t));

    return ESP_OK;

#endif /* INFERENCE_HAS_MODEL */
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
