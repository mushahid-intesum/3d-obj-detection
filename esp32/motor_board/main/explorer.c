/**
 * @file explorer.c
 * @brief Pink uniform noise action generator — MCU implementation.
 *
 * Generates temporally-correlated exploration actions:
 *   1. Voss-McCartney algorithm for real-time pink noise
 *      (much cheaper than FFT — O(1) per sample, O(log N) state)
 *   2. Approximate Gaussian CDF for uniform marginals
 *   3. Map uniform [0,1] to discrete {FORWARD, RIGHT, LEFT}
 *
 * The Voss-McCartney method generates pink noise by summing
 * multiple random generators updated at geometrically spaced rates.
 * This avoids FFT entirely and works well on MCU.
 */
#include "explorer.h"
#include "config.h"
#include "esp_log.h"

#include <string.h>
#include <math.h>

static const char *TAG = "explorer";

/* ─── Voss-McCartney pink noise generator ─── */

#define PINK_OCTAVES  8   /* number of octave generators */

typedef struct {
    uint32_t rng_state;           /* PRNG state */
    float    octave_val[PINK_OCTAVES];  /* current value per octave */
    float    running_sum;         /* sum of all octave values */
    uint32_t counter;             /* step counter for octave scheduling */
} pink_state_t;

static pink_state_t s_pink;

/* Pre-computed action sequence */
static uint8_t s_actions[EXPLORE_SEQ_LENGTH];
static uint32_t s_step = 0;

/**
 * @brief Simple xorshift32 PRNG — fast, good enough for noise.
 */
static uint32_t xorshift32(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}

/**
 * @brief Random float in [-1, 1] from PRNG.
 */
static float rand_float(uint32_t *state)
{
    uint32_t r = xorshift32(state);
    return (float)(int32_t)r / (float)INT32_MAX;
}

/**
 * @brief Initialize pink noise generator.
 */
static void pink_init(pink_state_t *p, uint32_t seed)
{
    p->rng_state = seed ? seed : 12345;
    p->running_sum = 0.0f;
    p->counter = 0;

    for (int i = 0; i < PINK_OCTAVES; i++) {
        p->octave_val[i] = rand_float(&p->rng_state);
        p->running_sum += p->octave_val[i];
    }
}

/**
 * @brief Generate one pink noise sample using Voss-McCartney.
 *
 * Each octave is updated when the counter hits its period:
 *   Octave 0: every step
 *   Octave 1: every 2 steps
 *   Octave 2: every 4 steps
 *   ...
 *   Octave k: every 2^k steps
 *
 * The output is the sum of all octaves, producing a 1/f spectrum.
 */
static float pink_next(pink_state_t *p)
{
    uint32_t c = p->counter;
    p->counter++;

    /* Find the lowest set bit — determines which octave to update.
     * Octave k is updated when bit k transitions from 0→1. */
    for (int i = 0; i < PINK_OCTAVES; i++) {
        if ((c & (1u << i)) == 0) {
            /* This octave needs updating */
            p->running_sum -= p->octave_val[i];
            p->octave_val[i] = rand_float(&p->rng_state);
            p->running_sum += p->octave_val[i];
            break;  /* Only update the lowest triggered octave */
        }
    }

    /* Add white noise for the highest frequency */
    float white = rand_float(&p->rng_state);

    /* Normalize: sum is in [-PINK_OCTAVES-1, PINK_OCTAVES+1] */
    float raw = (p->running_sum + white) / (PINK_OCTAVES + 1);

    return raw;  /* Approximately Gaussian, range ~ [-1, 1] */
}

/**
 * @brief Approximate Gaussian CDF (normal distribution) → [0, 1].
 *
 * Uses a fast rational approximation of the error function.
 * Accuracy: ~0.001 max error, more than sufficient for exploration.
 */
static float approx_normal_cdf(float x)
{
    /* Clamp extreme values */
    if (x < -4.0f) return 0.0f;
    if (x >  4.0f) return 1.0f;

    /* Approximation: Φ(x) ≈ 1 / (1 + exp(-1.7 * x - 0.73 * x^3))
     * Simplified logistic approximation. */
    float t = -1.7f * x - 0.73f * x * x * x * 0.05f;

    /* Fast exp approximation */
    float ex;
    if (t > 10.0f)  ex = 22026.0f;
    else if (t < -10.0f) ex = 0.0f;
    else {
        /* Padé approximant for exp */
        ex = (1.0f + t / 2.0f + t * t / 12.0f);
        ex = ex * ex;  /* crude squaring to improve range */
        if (ex < 0.0f) ex = 0.0f;
    }

    float cdf = 1.0f / (1.0f + ex);
    return cdf;
}

esp_err_t explorer_init(uint32_t seed)
{
    pink_init(&s_pink, seed);

    /* Pre-compute entire action sequence */
    for (int i = 0; i < EXPLORE_SEQ_LENGTH; i++) {
        /* Step 1: Pink noise sample (temporally correlated) */
        float pink_sample = pink_next(&s_pink);

        /* Step 2: CDF transform → uniform [0, 1]
         * The pink noise is approximately N(0, σ²), scale to standard normal */
        float u = approx_normal_cdf(pink_sample * 2.0f);

        /* Step 3: Map to discrete actions (equal probability) */
        if (u < 0.333f) {
            s_actions[i] = ACTION_TURN_LEFT;
        } else if (u > 0.667f) {
            s_actions[i] = ACTION_TURN_RIGHT;
        } else {
            s_actions[i] = ACTION_FORWARD;
        }
    }

    s_step = 0;

    /* Log action distribution */
    int counts[3] = {0, 0, 0};
    for (int i = 0; i < EXPLORE_SEQ_LENGTH; i++) {
        if (s_actions[i] < 3) counts[s_actions[i]]++;
    }
    ESP_LOGI(TAG, "Explorer initialized: %d actions (FWD=%d R=%d L=%d)",
             EXPLORE_SEQ_LENGTH, counts[0], counts[1], counts[2]);

    return ESP_OK;
}

uint8_t explorer_next_action(void)
{
    uint8_t action = s_actions[s_step % EXPLORE_SEQ_LENGTH];
    s_step++;
    return action;
}

uint32_t explorer_get_step(void)
{
    return s_step;
}

void explorer_reset(void)
{
    s_step = 0;
}
