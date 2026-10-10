/**
 * @file correlation.h
 * @brief INT8 cross-correlation computation in plain C.
 *
 * Computes the 258-dim correlation cue between goal and observation
 * feature maps on the ESP32-S3 without using TFLite.
 *
 * Features arrive as 8×8×1024 from the encoder, pooled to 4×4 before
 * computing cross-correlation (matching student model's CORR_POOL_SIZE).
 */
#ifndef CORRELATION_H
#define CORRELATION_H

#include <stdint.h>

/** Feature map dimensions from TinyEncoder (after pooling to 4×4). */
#define CORR_FEAT_H     4
#define CORR_FEAT_W     4
#define CORR_FEAT_CH    1024
#define CORR_FEAT_SIZE  (CORR_FEAT_H * CORR_FEAT_W * CORR_FEAT_CH)  /* 16384 */

/** Raw encoder output dimensions (received via SPI). */
#define ENCODER_FEAT_H   8
#define ENCODER_FEAT_W   8
#define ENCODER_FEAT_CH  1024
#define ENCODER_FEAT_RAW_SIZE  (ENCODER_FEAT_H * ENCODER_FEAT_W * ENCODER_FEAT_CH)  /* 65536 */

/** Correlation cue dimensions. */
#define CORR_CROSS_SIZE 256  /* 16 x 16 (4×4 × 4×4) */
#define CORR_LR_SIZE    2   /* left sim + right sim */
#define CORR_CUE_SIZE   258 /* 256 + 2 */

/**
 * @brief Compute the full 258-dim correlation cue from two pooled feature maps.
 *
 * Computes:
 *   - 16×16 cross-correlation (cosine similarity between all 4×4 spatial positions)
 *   - Left/Right similarity (2 values)
 *
 * Features must be pre-pooled from 8×8 to 4×4 before calling.
 * All arithmetic is INT8/INT32 — no floating point.
 *
 * @param[in]  goal_feat  Goal feature map, INT8, shape [16 * 1024] = 16384 bytes.
 *                         Layout: position-major, i.e. [pos0_ch0, ..., pos15_ch1023]
 * @param[in]  obs_feat   Observation feature map, same layout.
 * @param[out] cue        Output correlation cue, INT8, 258 bytes.
 */
void correlation_compute(const int8_t *goal_feat, const int8_t *obs_feat,
                         int8_t *cue);

/**
 * @brief Pool 8×8 feature map to 4×4 using average pooling.
 *
 * @param[in]  src  Source 8×8×1024 INT8 feature map (65536 bytes).
 * @param[out] dst  Destination 4×4×1024 INT8 feature map (16384 bytes).
 */
void correlation_pool_features(const int8_t *src, int8_t *dst);

#endif /* CORRELATION_H */
