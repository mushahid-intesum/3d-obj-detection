/**
 * @file ultrasonic.c
 * @brief HC-SR04 driver using esp_timer for µs-precision echo measurement.
 *
 * The HC-SR04 works at 5V logic but the echo pin can be used with a
 * voltage divider (5V → 3.3V) for ESP32-S3 compatibility.
 */
#include "ultrasonic.h"
#include "pin_config.h"
#include "driver/gpio.h"
#include "esp_timer.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <string.h>

static const char *TAG = "ultrasonic";

/** Timeout for echo pulse in microseconds (~4m range). */
#define US_TIMEOUT_US   25000

/** Number of readings for median filter. */
#define MEDIAN_COUNT    3

/** Cached distance, updated by background task. */
static volatile uint16_t s_cached_distance_cm = UINT16_MAX;

esp_err_t ultrasonic_init(void)
{
    /* Trigger pin: output */
    gpio_config_t trig_conf = {
        .pin_bit_mask = (1ULL << US_TRIG_PIN),
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    gpio_config(&trig_conf);
    gpio_set_level(US_TRIG_PIN, 0);

    /* Echo pin: input */
    gpio_config_t echo_conf = {
        .pin_bit_mask = (1ULL << US_ECHO_PIN),
        .mode         = GPIO_MODE_INPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    gpio_config(&echo_conf);

    ESP_LOGI(TAG, "HC-SR04 initialized (trig=%d, echo=%d)",
             US_TRIG_PIN, US_ECHO_PIN);
    return ESP_OK;
}

/**
 * @brief Single raw measurement (blocking).
 */
static esp_err_t measure_once(uint16_t *cm)
{
    /* Send 10µs trigger pulse */
    gpio_set_level(US_TRIG_PIN, 1);
    esp_rom_delay_us(10);
    gpio_set_level(US_TRIG_PIN, 0);

    /* Wait for echo to go HIGH */
    int64_t start_wait = esp_timer_get_time();
    while (gpio_get_level(US_ECHO_PIN) == 0) {
        if ((esp_timer_get_time() - start_wait) > US_TIMEOUT_US) {
            *cm = UINT16_MAX;
            return ESP_ERR_TIMEOUT;
        }
    }

    /* Measure echo HIGH duration */
    int64_t echo_start = esp_timer_get_time();
    while (gpio_get_level(US_ECHO_PIN) == 1) {
        if ((esp_timer_get_time() - echo_start) > US_TIMEOUT_US) {
            *cm = UINT16_MAX;
            return ESP_ERR_TIMEOUT;
        }
    }
    int64_t echo_end = esp_timer_get_time();

    /* Convert to cm: speed of sound ≈ 343 m/s → round trip */
    int64_t pulse_us = echo_end - echo_start;
    *cm = (uint16_t)(pulse_us / 58);

    return ESP_OK;
}

/**
 * @brief Comparator for qsort (uint16_t).
 */
static int cmp_u16(const void *a, const void *b)
{
    uint16_t va = *(const uint16_t *)a;
    uint16_t vb = *(const uint16_t *)b;
    return (va > vb) - (va < vb);
}

esp_err_t ultrasonic_measure(uint16_t *distance_cm)
{
    uint16_t readings[MEDIAN_COUNT];

    for (int i = 0; i < MEDIAN_COUNT; i++) {
        measure_once(&readings[i]);
        esp_rom_delay_us(2000); /* 2ms between pings */
    }

    /* Median filter: sort and take middle value */
    qsort(readings, MEDIAN_COUNT, sizeof(uint16_t), cmp_u16);
    *distance_cm = readings[MEDIAN_COUNT / 2];

    return ESP_OK;
}

uint16_t ultrasonic_get_cached_cm(void)
{
    return s_cached_distance_cm;
}

/**
 * @brief Background polling task (10 Hz).
 */
static void ultrasonic_task(void *pvParam)
{
    uint16_t cm;
    while (1) {
        if (ultrasonic_measure(&cm) == ESP_OK) {
            s_cached_distance_cm = cm;
        }
        vTaskDelay(pdMS_TO_TICKS(100)); /* 10 Hz */
    }
}

esp_err_t ultrasonic_start_task(void)
{
    BaseType_t ret = xTaskCreatePinnedToCore(
        ultrasonic_task, "ultrasonic",
        2048,           /* stack size */
        NULL,           /* param */
        2,              /* priority: low */
        NULL,           /* handle */
        0               /* core 0 */
    );

    if (ret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create ultrasonic task");
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Ultrasonic polling task started (10 Hz, core 0)");
    return ESP_OK;
}
