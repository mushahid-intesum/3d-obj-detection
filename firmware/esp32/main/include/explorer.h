/**
 * @file explorer.h
 * @brief Pink uniform noise action generator for autonomous exploration.
 *
 * Generates temporally-correlated action sequences at startup using:
 *   1. FFT-based pink noise (1/f spectrum) for temporal correlation
 *   2. Gaussian CDF transform for uniform marginals
 *   3. Discrete mapping to {FORWARD, TURN_RIGHT, TURN_LEFT}
 *
 * The sequence is pre-computed at boot and consumed step-by-step.
 */
#ifndef EXPLORER_H
#define EXPLORER_H

#include "esp_err.h"
#include <stdint.h>

/**
 * @brief Initialize the explorer and pre-compute the action sequence.
 *
 * Generates EXPLORE_SEQ_LENGTH actions using pink uniform noise.
 * Uses a simple LFSR-based PRNG for phase randomization.
 *
 * @param seed  Random seed for reproducibility.
 * @return ESP_OK on success.
 */
esp_err_t explorer_init(uint32_t seed);

/**
 * @brief Get the next exploration action.
 *
 * Returns the next action from the pre-computed sequence.
 * Wraps around when the sequence is exhausted.
 *
 * @return Action ID: ACTION_FORWARD(0), ACTION_TURN_RIGHT(1), or ACTION_TURN_LEFT(2).
 *         Never returns ACTION_STOP — exploration doesn't stop on its own.
 */
uint8_t explorer_next_action(void);

/**
 * @brief Get the current step index in the sequence.
 */
uint32_t explorer_get_step(void);

/**
 * @brief Reset the sequence index to the beginning.
 */
void explorer_reset(void);

#endif /* EXPLORER_H */
