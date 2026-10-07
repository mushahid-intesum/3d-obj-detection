/**
 * @file navigator.c
 * @brief Autonomous image-goal navigation loop on ESP32-S3.
 *
 * Core loop:
 *   1. Encode goal → cache 288-byte feature map (once)
 *   2. Per step: capture → downsample → encode → correlate → policy → act
 *   3. Ultrasonic safety override on forward actions
 */
#include "navigator.h"
#include "inference.h"
#include "correlation.h"
#include "camera.h"
#include "image_proc.h"
#include "ultrasonic.h"
#include "motor.h"
#include "pin_config.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "driver/gpio.h"

#include <string.h>

static const char *TAG = "navigator";

/** Cached goal features — computed once per episode. */
static int8_t s_goal_features[CORR_FEAT_SIZE];

/** Frame buffer for downscaled 48x48 image. */
static uint8_t s_img_buf[IMG_TARGET_SIZE];

/**
 * @brief Wait for the goal button to be pressed, then capture goal image.
 */
static esp_err_t capture_goal_on_button(void)
{
    /* Configure button pin with pull-up */
    gpio_config_t btn_conf = {
        .pin_bit_mask = (1ULL << BTN_GOAL_PIN),
        .mode         = GPIO_MODE_INPUT,
        .pull_up_en   = GPIO_PULLUP_ENABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    gpio_config(&btn_conf);

    ESP_LOGI(TAG, "Press GOAL button to capture target image...");

    /* Wait for button press (active LOW) */
    while (gpio_get_level(BTN_GOAL_PIN) == 1) {
        vTaskDelay(pdMS_TO_TICKS(50));
    }
    /* Debounce */
    vTaskDelay(pdMS_TO_TICKS(200));

    /* Capture and downsample */
    uint8_t *raw = NULL;
    int w, h;
    size_t len;
    esp_err_t err = camera_capture(&raw, &w, &h, &len);
    if (err != ESP_OK) return err;

    image_downsample(raw, w, h, s_img_buf);
    camera_fb_release();

    /* Encode goal */
    err = inference_run_encoder(s_img_buf, s_goal_features);
    if (err != ESP_OK) return err;

    ESP_LOGI(TAG, "Goal image captured and encoded!");
    return ESP_OK;
}

/**
 * @brief Navigation task — runs on core 1.
 */
static void navigation_task(void *pvParam)
{
    nav_goal_mode_t mode = (nav_goal_mode_t)(uintptr_t)pvParam;

    /* ── Initialize inference engine ── */
    esp_err_t err = inference_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Inference init failed: 0x%x", err);
        ESP_LOGE(TAG, "Make sure model .h files are included in inference.cpp");
        vTaskDelete(NULL);
        return;
    }

    /* ── Obtain goal ── */
    switch (mode) {
    case NAV_GOAL_FROM_BUTTON:
        err = capture_goal_on_button();
        break;
    case NAV_GOAL_FROM_FLASH:
        /* TODO: load goal image from flash partition */
        ESP_LOGE(TAG, "Flash goal not implemented yet");
        err = ESP_ERR_NOT_SUPPORTED;
        break;
    }

    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Failed to obtain goal image");
        vTaskDelete(NULL);
        return;
    }

    /* ── Navigation loop ── */
    ESP_LOGI(TAG, "Starting navigation (max %d steps, %d Hz)...",
             NAV_MAX_STEPS, NAV_RATE_HZ);

    const TickType_t step_delay = pdMS_TO_TICKS(1000 / NAV_RATE_HZ);
    int8_t obs_features[CORR_FEAT_SIZE];
    int8_t corr_cue[CORR_CUE_SIZE];
    int8_t action_logits[4];

    int step = 0;
    bool done = false;

    while (!done && step < NAV_MAX_STEPS) {
        int64_t t_start = esp_timer_get_time();

        /* 1. Capture and downsample */
        uint8_t *raw = NULL;
        int w, h;
        size_t len;
        if (camera_capture(&raw, &w, &h, &len) != ESP_OK) {
            ESP_LOGW(TAG, "Frame capture failed, retrying...");
            vTaskDelay(step_delay);
            continue;
        }
        image_downsample(raw, w, h, s_img_buf);
        camera_fb_release();

        /* 2. Encode observation */
        if (inference_run_encoder(s_img_buf, obs_features) != ESP_OK) {
            ESP_LOGE(TAG, "Encoder inference failed");
            break;
        }

        /* 3. Compute correlation (plain C, no TFLite) */
        correlation_compute(s_goal_features, obs_features, corr_cue);

        /* 4. Run policy */
        if (inference_run_policy(corr_cue, action_logits) != ESP_OK) {
            ESP_LOGE(TAG, "Policy inference failed");
            break;
        }
        int action = inference_argmax_i8(action_logits, 4);

        /* 5. Safety override */
        uint16_t dist = ultrasonic_get_cached_cm();
        if (action == ACTION_FORWARD && dist < NAV_OBSTACLE_CM) {
            ESP_LOGW(TAG, "Obstacle at %u cm — overriding to TURN_RIGHT", dist);
            action = ACTION_TURN_RIGHT;
        }

        /* 6. Execute */
        motor_execute_action(action, 150);

        /* 7. Check termination */
        if (action == ACTION_STOP) {
            done = true;
        }

        /* Timing */
        int64_t t_elapsed_us = esp_timer_get_time() - t_start;

        if (step % 10 == 0) {
            const char *action_names[] = {"FWD", "LFT", "RGT", "STP"};
            ESP_LOGI(TAG, "Step %3d: action=%s dist=%ucm logits=[%d,%d,%d,%d] %lldms",
                     step, action_names[action], dist,
                     action_logits[0], action_logits[1],
                     action_logits[2], action_logits[3],
                     t_elapsed_us / 1000);
        }

        step++;

        /* Rate limiting */
        int64_t remaining_us = (1000000 / NAV_RATE_HZ) - t_elapsed_us;
        if (remaining_us > 0) {
            vTaskDelay(pdMS_TO_TICKS(remaining_us / 1000));
        }
    }

    /* ── Episode complete ── */
    motor_stop();
    ESP_LOGI(TAG, "═══════════════════════════════════");
    ESP_LOGI(TAG, "  Navigation complete!");
    ESP_LOGI(TAG, "  Steps: %d / %d", step, NAV_MAX_STEPS);
    ESP_LOGI(TAG, "  Stopped: %s", done ? "policy STOP action" : "max steps reached");
    ESP_LOGI(TAG, "═══════════════════════════════════");

    vTaskDelete(NULL);
}

esp_err_t navigator_start(nav_goal_mode_t mode)
{
    BaseType_t ret = xTaskCreatePinnedToCore(
        navigation_task, "navigator",
        16384,                              /* 16KB stack (TFLite needs headroom) */
        (void *)(uintptr_t)mode,
        5,                                  /* priority: medium-high */
        NULL,
        1                                   /* core 1 */
    );

    if (ret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create navigation task");
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Navigation task created on core 1");
    return ESP_OK;
}
