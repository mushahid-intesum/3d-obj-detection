/**
 * @file navigator.c
 * @brief Autonomous image-goal navigation (single board).
 *
 * Core loop:
 *   1. Cache goal features (from first frame or NVS)
 *   2. Per step:
 *      a. Capture camera frame → downsample to 96×96
 *      b. Read sonar distance
 *      c. Run student model: image + sonar → action logits
 *      d. Safety override if sonar blocked
 *      e. Execute motor action
 *      f. Stream frame to laptop (optional monitoring)
 */
#include "navigator.h"
#include "inference.h"
#include "motor.h"
#include "imu_uart.h"
#include "config.h"

#include "camera.h"
#include "image_proc.h"
#include "wifi_stream.h"
#include "sonar.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "nvs.h"
#include "nvs_flash.h"

#include <string.h>
#include <stdlib.h>
#include "esp_heap_caps.h"

static const char *TAG = "navigator";

/** Downscaled image buffer — allocated in PSRAM. */
static uint8_t *s_img_buf = NULL;

/** Goal image buffer — allocated in PSRAM. */
static uint8_t *s_goal_img = NULL;

static const char *ACTION_NAMES[] = {"FWD", "RIGHT", "LEFT", "STOP"};

/**
 * @brief Get a random turn direction for obstacle override.
 */
static uint8_t random_turn(void)
{
    return (rand() % 2 == 0) ? ACTION_TURN_RIGHT : ACTION_TURN_LEFT;
}

/**
 * @brief Navigation task — runs on core 1.
 */
static void navigation_task(void *pvParam)
{
    nav_goal_mode_t mode = (nav_goal_mode_t)(uintptr_t)pvParam;
    esp_err_t err;

    /* Allocate image buffers in PSRAM */
    if (!s_img_buf) {
        s_img_buf = (uint8_t *)heap_caps_malloc(IMG_TARGET_SIZE, MALLOC_CAP_SPIRAM);
    }
    if (!s_goal_img) {
        s_goal_img = (uint8_t *)heap_caps_malloc(IMG_TARGET_SIZE, MALLOC_CAP_SPIRAM);
    }
    if (!s_img_buf || !s_goal_img) {
        ESP_LOGE(TAG, "Failed to allocate image buffers in PSRAM!");
        vTaskDelete(NULL);
        return;
    }

    /* Obtain goal */
    bool have_goal = false;

    if (mode == NAV_GOAL_FROM_FLASH) {
        err = navigator_load_goal(s_goal_img);
        if (err == ESP_OK) {
            /* Encode and cache goal features in the inference engine */
            err = inference_set_goal(s_goal_img);
            if (err == ESP_OK) {
                have_goal = true;
                ESP_LOGI(TAG, "Goal loaded from NVS and encoded");
            }
        } else {
            ESP_LOGW(TAG, "No goal in NVS — will use first frame");
        }
    }

    /* Navigation loop */
    ESP_LOGI(TAG, "Starting navigation (max %d steps, %d Hz)...",
             NAV_MAX_STEPS, NAV_RATE_HZ);

    int step = 0;
    uint8_t last_action = ACTION_STOP;
    uint32_t frame_id = 0;
    int8_t action_logits[4];

    while (step < NAV_MAX_STEPS) {
        int64_t t_start = esp_timer_get_time();

        /* 1. Capture camera frame */
        camera_fb_t *fb = camera_capture_frame();
        if (!fb) {
            ESP_LOGW(TAG, "Frame capture failed, skipping");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        /* 2. Downsample to 96×96 RGB888
         *
         * NOTE: Camera is initialized in JPEG mode for streaming.
         * For navigation, we need RGB565 input for image_downsample().
         * TODO: Either switch camera to RGB565 mode before navigation,
         *       or decode JPEG to RGB on-chip (more expensive).
         *       For now, this assumes camera has been re-initialized
         *       in RGB mode before navigator_start() is called.
         */
        image_downsample(fb->buf, fb->width, fb->height, s_img_buf);

        /* 3. Read sonar + IMU */
        float sonar_cm = sonar_get_distance_cm();
        bool sonar_blocked = (sonar_cm < SONAR_THRESHOLD_CM);
        float heading = imu_get_heading();
        if (heading < 0.0f) heading = 0.0f;

        /* 4. If no goal yet, use first frame as goal */
        if (!have_goal) {
            memcpy(s_goal_img, s_img_buf, IMG_TARGET_SIZE);
            inference_set_goal(s_goal_img);
            have_goal = true;
            ESP_LOGI(TAG, "Goal set from first frame");
            navigator_save_goal(s_goal_img);

            /* Skip this step — need a different observation */
            last_action = ACTION_FORWARD;
            motor_execute_action(ACTION_FORWARD);
            camera_release_frame(fb);
            step++;
            continue;
        }

        /* 5. Run policy inference: image + sonar → action logits */
        err = inference_run_policy(s_img_buf, sonar_cm, action_logits);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "Policy inference failed");
            camera_release_frame(fb);
            break;
        }

        int policy_idx = inference_argmax_i8(action_logits, 4);
        uint8_t action = (uint8_t)policy_idx;

        /* 6. Safety override (always active) */
        if (sonar_blocked && action == ACTION_FORWARD) {
            ESP_LOGW(TAG, "Sonar override! dist=%.1f cm, FWD → turn", sonar_cm);
            action = random_turn();
        }

        /* 7. Stream frame to laptop (monitoring, best-effort) */
        if (stream_is_connected()) {
            stream_send_frame(
                frame_id, (uint32_t)step, action,
                sonar_blocked ? 1 : 0,
                sonar_cm, heading, fb->buf, fb->len
            );
        }
        camera_release_frame(fb);
        frame_id++;

        /* 8. Execute action */
        motor_execute_action(action);
        last_action = action;

        /* 9. Termination check */
        if (action == ACTION_STOP) {
            ESP_LOGI(TAG, "Policy chose STOP — goal reached!");
            break;
        }

        /* 10. Logging */
        int64_t t_elapsed_us = esp_timer_get_time() - t_start;
        if (step % 5 == 0) {
            const char *aname = (action < 4) ? ACTION_NAMES[action] : "?";
            ESP_LOGI(TAG, "Step %3d: action=%s hdg=%.1f sonar=%.1f%s "
                     "logits=[%d,%d,%d,%d] %lldms",
                     step, aname, heading, sonar_cm,
                     sonar_blocked ? " BLOCKED" : "",
                     action_logits[0], action_logits[1],
                     action_logits[2], action_logits[3],
                     t_elapsed_us / 1000);
        }

        step++;
    }

    motor_stop();
    ESP_LOGI(TAG, "═══════════════════════════════════");
    ESP_LOGI(TAG, "  Navigation complete!");
    ESP_LOGI(TAG, "  Steps: %d / %d", step, NAV_MAX_STEPS);
    ESP_LOGI(TAG, "  Final: %s",
             (last_action == ACTION_STOP) ? "GOAL REACHED" : "MAX STEPS");
    ESP_LOGI(TAG, "═══════════════════════════════════");

    vTaskDelete(NULL);
}

esp_err_t navigator_start(nav_goal_mode_t mode)
{
    BaseType_t ret = xTaskCreatePinnedToCore(
        navigation_task, "navigator",
        16384,
        (void *)(uintptr_t)mode,
        5, NULL, 1
    );

    if (ret != pdPASS) {
        ESP_LOGE(TAG, "Failed to create navigation task");
        return ESP_FAIL;
    }

    ESP_LOGI(TAG, "Navigation task created on core 1");
    return ESP_OK;
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Goal image persistence (NVS)
 *
 *  Stores the raw 96×96×3 goal image (27648 bytes) in NVS.
 *  NVS max blob size is 508000 bytes, so this fits easily.
 * ═══════════════════════════════════════════════════════════════════════════ */

esp_err_t navigator_save_goal(const uint8_t *goal_img)
{
    nvs_handle_t handle;
    esp_err_t err = nvs_open("nav", NVS_READWRITE, &handle);
    if (err != ESP_OK) return err;

    err = nvs_set_blob(handle, "goal_img", goal_img, IMG_TARGET_SIZE);
    if (err == ESP_OK) {
        err = nvs_commit(handle);
    }
    nvs_close(handle);

    if (err == ESP_OK) {
        ESP_LOGI(TAG, "Goal image saved to NVS (%d bytes)", IMG_TARGET_SIZE);
    }
    return err;
}

esp_err_t navigator_load_goal(uint8_t *goal_img)
{
    nvs_handle_t handle;
    esp_err_t err = nvs_open("nav", NVS_READONLY, &handle);
    if (err != ESP_OK) return err;

    size_t required = IMG_TARGET_SIZE;
    err = nvs_get_blob(handle, "goal_img", goal_img, &required);
    nvs_close(handle);

    if (err == ESP_OK && required == IMG_TARGET_SIZE) {
        ESP_LOGI(TAG, "Goal image loaded from NVS (%zu bytes)", required);
    }
    return err;
}
