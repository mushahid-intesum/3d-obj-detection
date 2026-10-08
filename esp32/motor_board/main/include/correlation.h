/**
 * @file correlation.h
 * @brief INT8 cross-correlation computation in plain C.
 *
 * Computes the 83-dim correlation cue between goal and observation
 * feature maps on the ESP32-S3 without using TFLite.
 */
#ifndef CORRELATION_H
#define CORRELATION_H

#include <stdint.h>

/** Feature map dimensions from TinyEncoder. */
#define CORR_FEAT_H     3
#define CORR_FEAT_W     3
#define CORR_FEAT_CH    32
#define CORR_FEAT_SIZE  (CORR_FEAT_H * CORR_FEAT_W * CORR_FEAT_CH)  /* 288 */

/** Correlation cue dimensions. */
#define CORR_CROSS_SIZE 81  /* 9 x 9 */
#define CORR_LR_SIZE    2   /* left sim + right sim */
#define CORR_CUE_SIZE   83  /* 81 + 2 */

/**
 * @brief Compute the full 83-dim correlation cue from two feature maps.
 *
 * Computes:
 *   - 9x9 cross-correlation (cosine similarity between all spatial positions)
 *   - Left/Right similarity (2 values)
 *
 * All arithmetic is INT8/INT32 — no floating point.
 *
 * @param[in]  goal_feat  Goal feature map, INT8, shape [9 * 32] = 288 bytes.
 *                         Layout: position-major, i.e. [pos0_ch0, pos0_ch1, ..., pos8_ch31]
 * @param[in]  obs_feat   Observation feature map, same layout.
 * @param[out] cue        Output correlation cue, INT8, 83 bytes.
 */
void correlation_compute(const int8_t *goal_feat, const int8_t *obs_feat,
                         int8_t *cue);

#endif /* CORRELATION_H */
