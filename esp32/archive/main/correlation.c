/**
 * @file correlation.c
 * @brief INT8 cross-correlation — runs on ESP32-S3, no TFLite needed.
 */
#include "correlation.h"
#include <string.h>

/**
 * @brief Dot product of two INT8 vectors, accumulated in INT32.
 */
static int32_t dot_i8(const int8_t *a, const int8_t *b, int len)
{
    int32_t acc = 0;
    for (int i = 0; i < len; i++) {
        acc += (int32_t)a[i] * (int32_t)b[i];
    }
    return acc;
}

/**
 * @brief Compute L2 norm squared of an INT8 vector (in INT32).
 */
static int32_t norm_sq_i8(const int8_t *v, int len)
{
    int32_t acc = 0;
    for (int i = 0; i < len; i++) {
        acc += (int32_t)v[i] * (int32_t)v[i];
    }
    return acc;
}

/**
 * @brief Approximate cosine similarity scaled to INT8 range [-127, 127].
 *
 * cosine_sim = dot(a,b) / (|a| * |b|)
 *
 * We approximate the normalization by computing:
 *   result = dot(a,b) * 127 / sqrt(norm_a * norm_b)
 *
 * Using integer square root approximation for MCU.
 */
static int8_t cosine_sim_i8(const int8_t *a, const int8_t *b, int len)
{
    int32_t dot = dot_i8(a, b, len);
    int32_t na = norm_sq_i8(a, len);
    int32_t nb = norm_sq_i8(b, len);

    if (na == 0 || nb == 0) return 0;

    /* Integer sqrt approximation (Newton's method, 4 iterations) */
    int32_t product = na / 256 * (nb / 256);  /* scale down to avoid overflow */
    if (product <= 0) return 0;

    uint32_t x = (uint32_t)product;
    uint32_t s = x;
    /* Initial guess */
    if (s > 0) {
        s = (s + 1) >> 1;
        for (int i = 0; i < 8; i++) {
            uint32_t t = (s + x / s) >> 1;
            if (t >= s) break;
            s = t;
        }
    }

    /* Scale: dot was computed at full precision, sqrt(na*nb) at /256 scale
     * So we need: result = dot * 127 / (s * 256)  */
    int32_t denom = (int32_t)s * 256;
    if (denom == 0) return 0;

    int32_t result = (dot * 127) / denom;

    /* Clamp to INT8 range */
    if (result > 127) result = 127;
    if (result < -127) result = -127;

    return (int8_t)result;
}

void correlation_compute(const int8_t *goal_feat, const int8_t *obs_feat,
                         int8_t *cue)
{
    /*
     * Feature map layout: [position][channel]
     * Position index for 3x3 grid:
     *   0  1  2
     *   3  4  5
     *   6  7  8
     * Each position has CORR_FEAT_CH (32) channels.
     */

    /* ── 9x9 Cross-correlation ── */
    for (int g = 0; g < 9; g++) {
        const int8_t *g_vec = &goal_feat[g * CORR_FEAT_CH];
        for (int o = 0; o < 9; o++) {
            const int8_t *o_vec = &obs_feat[o * CORR_FEAT_CH];
            cue[g * 9 + o] = cosine_sim_i8(g_vec, o_vec, CORR_FEAT_CH);
        }
    }

    /* ── Left/Right similarity ── */
    /* Left positions:  col 0 → indices 0, 3, 6 */
    /* Right positions: col 2 → indices 2, 5, 8 */
    static const int left_idx[3]  = {0, 3, 6};
    static const int right_idx[3] = {2, 5, 8};

    /* Average the 3 left positions' feature vectors */
    int32_t g_left_avg[CORR_FEAT_CH];
    int32_t o_left_avg[CORR_FEAT_CH];
    int32_t g_right_avg[CORR_FEAT_CH];
    int32_t o_right_avg[CORR_FEAT_CH];
    memset(g_left_avg, 0, sizeof(g_left_avg));
    memset(o_left_avg, 0, sizeof(o_left_avg));
    memset(g_right_avg, 0, sizeof(g_right_avg));
    memset(o_right_avg, 0, sizeof(o_right_avg));

    for (int k = 0; k < 3; k++) {
        int li = left_idx[k];
        int ri = right_idx[k];
        for (int c = 0; c < CORR_FEAT_CH; c++) {
            g_left_avg[c]  += goal_feat[li * CORR_FEAT_CH + c];
            o_left_avg[c]  += obs_feat[li * CORR_FEAT_CH + c];
            g_right_avg[c] += goal_feat[ri * CORR_FEAT_CH + c];
            o_right_avg[c] += obs_feat[ri * CORR_FEAT_CH + c];
        }
    }

    /* Left similarity: dot(g_left, o_left) / (|g_left| * |o_left|) */
    int32_t left_dot = 0, left_gn = 0, left_on = 0;
    int32_t right_dot = 0, right_gn = 0, right_on = 0;
    for (int c = 0; c < CORR_FEAT_CH; c++) {
        left_dot  += g_left_avg[c] * o_left_avg[c];
        left_gn   += g_left_avg[c] * g_left_avg[c];
        left_on   += o_left_avg[c] * o_left_avg[c];
        right_dot += g_right_avg[c] * o_right_avg[c];
        right_gn  += g_right_avg[c] * g_right_avg[c];
        right_on  += o_right_avg[c] * o_right_avg[c];
    }

    /* Scale down and compute similarity */
    int32_t left_denom = 1, right_denom = 1;
    if (left_gn > 0 && left_on > 0) {
        /* Approximate: use product>>16 to fit in range */
        uint32_t lp = (uint32_t)(left_gn >> 8) * (uint32_t)(left_on >> 8);
        uint32_t ls = lp;
        if (ls > 0) {
            ls = (ls + 1) >> 1;
            for (int i = 0; i < 8; i++) { uint32_t t = (ls + lp/ls) >> 1; if (t>=ls) break; ls=t; }
        }
        left_denom = (int32_t)ls > 0 ? (int32_t)ls : 1;
    }
    if (right_gn > 0 && right_on > 0) {
        uint32_t rp = (uint32_t)(right_gn >> 8) * (uint32_t)(right_on >> 8);
        uint32_t rs = rp;
        if (rs > 0) {
            rs = (rs + 1) >> 1;
            for (int i = 0; i < 8; i++) { uint32_t t = (rs + rp/rs) >> 1; if (t>=rs) break; rs=t; }
        }
        right_denom = (int32_t)rs > 0 ? (int32_t)rs : 1;
    }

    int32_t left_sim = (left_dot * 127) / (left_denom * 256);
    int32_t right_sim = (right_dot * 127) / (right_denom * 256);

    if (left_sim > 127) left_sim = 127;
    if (left_sim < -127) left_sim = -127;
    if (right_sim > 127) right_sim = 127;
    if (right_sim < -127) right_sim = -127;

    cue[CORR_CROSS_SIZE]     = (int8_t)left_sim;
    cue[CORR_CROSS_SIZE + 1] = (int8_t)right_sim;
}
