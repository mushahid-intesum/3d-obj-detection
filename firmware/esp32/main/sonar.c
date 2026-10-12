/**
 * @file sonar.c
 * @brief HC-SR04 ultrasonic distance sensor driver for ESP32-S3.
 *
 * Protocol:
 *   1. Pull TRIG HIGH for ≥10µs
 *   2. HC-SR04 sends 8× 40kHz burst
 *   3. ECHO goes HIGH for the round-trip time
 *   4. distance_cm = echo_duration_us / 58.0
 *
 * Timing is done with esp_timer_get_time() (µs resolution).
 * A portMUX spinlock protects the GPIO timing critical section
 * from being interrupted, which would corrupt the measurement.
 *
 * If no echo is received within SONAR_TIMEOUT_US (~30ms),
 * the function returns SONAR_MAX_RANGE_CM (no obstacle / out of range).
 */
#include "sonar.h"
#include "config.h"

#include "driver/gpio.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "rom/ets_sys.h"       /* ets_delay_us — busy-wait microsecond delay */

static const char *TAG = "sonar";

static portMUX_TYPE s_sonar_mux = portMUX_INITIALIZER_UNLOCKED;

esp_err_t sonar_init(void)
{
    /* Validate pins are assigned */
    if (SONAR_TRIG_PIN == 0 || SONAR_ECHO_PIN == 0) {
        ESP_LOGW(TAG, "Sonar pins not configured (TRIG=%d, ECHO=%d) — "
                 "sonar will return max range", SONAR_TRIG_PIN, SONAR_ECHO_PIN);
        return ESP_OK;  /* Non-fatal — collection mode works without sonar */
    }

    /* Configure TRIG as output (push-pull, no pull) */
    gpio_config_t trig_cfg = {
        .pin_bit_mask = (1ULL << SONAR_TRIG_PIN),
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&trig_cfg));
    gpio_set_level(SONAR_TRIG_PIN, 0);

    /* Configure ECHO as input (pull-down — idle LOW) */
    gpio_config_t echo_cfg = {
        .pin_bit_mask = (1ULL << SONAR_ECHO_PIN),
        .mode         = GPIO_MODE_INPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_ENABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&echo_cfg));

    ESP_LOGI(TAG, "HC-SR04 initialized (TRIG=%d, ECHO=%d, threshold=%.0f cm)",
             SONAR_TRIG_PIN, SONAR_ECHO_PIN, SONAR_THRESHOLD_CM);

    return ESP_OK;
}

float sonar_get_distance_cm(void)
{
    /* If pins not configured, return max range (no obstacle) */
    if (SONAR_TRIG_PIN == 0 || SONAR_ECHO_PIN == 0) {
        return SONAR_MAX_RANGE_CM;
    }

    /*
     * Critical section — disable scheduler interrupts during the
     * tight timing loop. The entire measurement takes ≤30ms.
     */
    portENTER_CRITICAL(&s_sonar_mux);

    /* ── 1. Send 10µs trigger pulse ── */
    gpio_set_level(SONAR_TRIG_PIN, 1);
    ets_delay_us(10);
    gpio_set_level(SONAR_TRIG_PIN, 0);

    /* ── 2. Wait for ECHO to go HIGH (start of return pulse) ── */
    int64_t t_wait_start = esp_timer_get_time();
    while (gpio_get_level(SONAR_ECHO_PIN) == 0) {
        if ((esp_timer_get_time() - t_wait_start) > SONAR_TIMEOUT_US) {
            portEXIT_CRITICAL(&s_sonar_mux);
            return SONAR_MAX_RANGE_CM;  /* No echo — nothing in range */
        }
    }

    /* ── 3. Measure how long ECHO stays HIGH ── */
    int64_t t_echo_start = esp_timer_get_time();
    while (gpio_get_level(SONAR_ECHO_PIN) == 1) {
        if ((esp_timer_get_time() - t_echo_start) > SONAR_TIMEOUT_US) {
            portEXIT_CRITICAL(&s_sonar_mux);
            return SONAR_MAX_RANGE_CM;  /* Echo stuck HIGH — out of range */
        }
    }
    int64_t t_echo_end = esp_timer_get_time();

    portEXIT_CRITICAL(&s_sonar_mux);

    /* ── 4. Calculate distance ── */
    int64_t echo_us = t_echo_end - t_echo_start;
    float distance_cm = (float)echo_us / 58.0f;

    /* Clamp to valid range */
    if (distance_cm < 0.0f) distance_cm = 0.0f;
    if (distance_cm > SONAR_MAX_RANGE_CM) distance_cm = SONAR_MAX_RANGE_CM;

    return distance_cm;
}

bool sonar_is_blocked(void)
{
    return sonar_get_distance_cm() < SONAR_THRESHOLD_CM;
}
