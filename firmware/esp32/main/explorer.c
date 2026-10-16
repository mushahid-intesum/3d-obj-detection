
#include "explorer.h"
#include "config.h"
#include "esp_log.h"

#include <string.h>
#include <math.h>

static const char *TAG = "explorer";


#define PINK_OCTAVES  8

typedef struct {
    uint32_t rng_state;
    float    octave_val[PINK_OCTAVES];
    float    running_sum;
    uint32_t counter;
} pink_state_t;

static pink_state_t s_pink;


static uint8_t s_actions[EXPLORE_SEQ_LENGTH];
static uint32_t s_step = 0;


static uint32_t xorshift32(uint32_t *state)
{
    uint32_t x = *state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    *state = x;
    return x;
}


static float rand_float(uint32_t *state)
{
    uint32_t r = xorshift32(state);
    return (float)(int32_t)r / (float)INT32_MAX;
}


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


static float pink_next(pink_state_t *p)
{
    uint32_t c = p->counter;
    p->counter++;


    for (int i = 0; i < PINK_OCTAVES; i++) {
        if ((c & (1u << i)) == 0) {

            p->running_sum -= p->octave_val[i];
            p->octave_val[i] = rand_float(&p->rng_state);
            p->running_sum += p->octave_val[i];
            break;
        }
    }


    float white = rand_float(&p->rng_state);


    float raw = (p->running_sum + white) / (PINK_OCTAVES + 1);

    return raw;
}


static float approx_normal_cdf(float x)
{

    if (x < -4.0f) return 0.0f;
    if (x >  4.0f) return 1.0f;


    float t = -1.7f * x - 0.73f * x * x * x * 0.05f;


    float ex;
    if (t > 10.0f)  ex = 22026.0f;
    else if (t < -10.0f) ex = 0.0f;
    else {

        ex = (1.0f + t / 2.0f + t * t / 12.0f);
        ex = ex * ex;
        if (ex < 0.0f) ex = 0.0f;
    }

    float cdf = 1.0f / (1.0f + ex);
    return cdf;
}

esp_err_t explorer_init(uint32_t seed)
{
    pink_init(&s_pink, seed);


    for (int i = 0; i < EXPLORE_SEQ_LENGTH; i++) {

        float pink_sample = pink_next(&s_pink);


        float u = approx_normal_cdf(pink_sample * 2.0f);


        if (u < 0.333f) {
            s_actions[i] = ACTION_TURN_LEFT;
        } else if (u > 0.667f) {
            s_actions[i] = ACTION_TURN_RIGHT;
        } else {
            s_actions[i] = ACTION_FORWARD;
        }
    }

    s_step = 0;


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
