/**
 * @file navigator.c
 * @brief Autonomous navigation loop on Motor Board.
 *
 * Core loop (SPI-slave-driven):
 *   1. Cache goal features (from first frame or NVS)
 *   2. Per step:
 *      a. Set SPI response with previous action + heading
 *      b. Wait for SPI transaction → receive obs features + obstacle flag
 *      c. Correlate obs with goal → 83-dim cue
 *      d. Run policy → action logits → argmax
 *      e. Safety override if obstacle
 *      f. Execute motor action
 */
#include "navigator.h"
#include "inference.h"
#include "correlation.h"
#include "spi_slave.h"
#include "motor.h"
#include "imu_uart.h"
#include "config.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "esp_timer.h"
#include "nvs.h"
#include "nvs_flash.h"

#include <string.h>

static const char *TAG = "navigator";

/** Cached goal features — set once per episode. */
static int8_t s_goal_features[CORR_FEAT_SIZE];

static const char *ACTION_NAMES[] = {"FWD", "RIGHT", "LEFT", "STOP"};

/**
 * @brief Navigation task — runs on core 1.
 */
static void navigation_task(void *pvParam)
{
    nav_goal_mode_t mode = (nav_goal_mode_t)(uintptr_t)pvParam;

    /* Initialize policy inference */
    esp_err_t err = inference_init();
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Policy init failed (0x%x) — no model embedded?", err);
        vTaskDelete(NULL);
        return;
    }

    /* Obtain goal features */
    bool have_goal = false;

    if (mode == NAV_GOAL_FROM_FLASH) {
        err = navigator_load_goal(s_goal_features);
        if (err == ESP_OK) {
            have_goal = true;
            ESP_LOGI(TAG, "Goal loaded from NVS");
        } else {
            ESP_LOGW(TAG, "No goal in NVS — will use first frame");
        }
    }

    /* Navigation loop */
    ESP_LOGI(TAG, "Starting navigation (max %d steps, %d Hz)...",
             NAV_MAX_STEPS, NAV_RATE_HZ);

    int8_t obs_features[CORR_FEAT_SIZE];
    int8_t corr_cue[CORR_CUE_SIZE];
    int8_t action_logits[4];

    int step = 0;
    uint8_t last_action = ACTION_STOP;

    while (step < NAV_MAX_STEPS) {
        int64_t t_start = esp_timer_get_time();

        /* 1. Prepare SPI response (from last step) */
        float heading = imu_get_heading();
        if (heading < 0.0f) heading = 0.0f;
        spi_slave_set_response(last_action, heading);

        /* 2. Wait for SPI transaction from Camera Board
         *    Receives: obstacle_flag + 288 bytes of encoder features */
        uint8_t obstacle_flag = 0;
        uint8_t msg_type = 0;

        err = spi_slave_receive(
            &obstacle_flag, &msg_type, obs_features, 2000
        );

        if (err != ESP_OK) {
            ESP_LOGW(TAG, "SPI timeout — Camera Board offline?");
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }

        /* Verify we got nav features (msg_type == 1) */
        if (msg_type != 1) {
            ESP_LOGW(TAG, "Expected nav features (msg_type=1), got %d", msg_type);
            vTaskDelay(pdMS_TO_TICKS(200));
            continue;
        }

        /* 3. If no goal yet, use first frame as goal */
        if (!have_goal) {
            memcpy(s_goal_features, obs_features, CORR_FEAT_SIZE);
            have_goal = true;
            ESP_LOGI(TAG, "Goal set from first frame");
            /* Save for next boot */
            navigator_save_goal(s_goal_features);
            /* Skip this step — need a different observation */
            last_action = ACTION_FORWARD;
            motor_execute_action(ACTION_FORWARD);
            step++;
            continue;
        }

        /* 4. Correlate obs with goal → 83-dim cue */
        correlation_compute(s_goal_features, obs_features, corr_cue);

        /* 5. Run policy MLP */
        err = inference_run_policy(corr_cue, action_logits);
        if (err != ESP_OK) {
            ESP_LOGE(TAG, "Policy inference failed");
            break;
        }

        int policy_idx = inference_argmax_i8(action_logits, 4);
        uint8_t action = (uint8_t)policy_idx;

        /* 6. Safety override */
        if (obstacle_flag && action == ACTION_FORWARD) {
            ESP_LOGW(TAG, "Obstacle override! FWD → RIGHT");
            action = ACTION_TURN_RIGHT;
        }

        /* 7. Execute action */
        motor_execute_action(action);
        last_action = action;

        /* 8. Termination check */
        if (action == ACTION_STOP) {
            ESP_LOGI(TAG, "Policy chose STOP — goal reached!");
            break;
        }

        /* 9. Logging */
        int64_t t_elapsed_us = esp_timer_get_time() - t_start;
        if (step % 5 == 0) {
            const char *aname = (action < 4) ? ACTION_NAMES[action] : "?";
            ESP_LOGI(TAG, "Step %3d: action=%s hdg=%.1f obs=%d "
                     "logits=[%d,%d,%d,%d] %lldms",
                     step, aname, heading, obstacle_flag,
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
 *  Goal feature persistence (NVS)
 * ═══════════════════════════════════════════════════════════════════════════ */

esp_err_t navigator_save_goal(const int8_t *features)
{
    nvs_handle_t handle;
    esp_err_t err = nvs_open("nav", NVS_READWRITE, &handle);
    if (err != ESP_OK) return err;

    err = nvs_set_blob(handle, "goal_feat", features, CORR_FEAT_SIZE);
    if (err == ESP_OK) {
        err = nvs_commit(handle);
    }
    nvs_close(handle);

    if (err == ESP_OK) {
        ESP_LOGI(TAG, "Goal features saved to NVS (%d bytes)", CORR_FEAT_SIZE);
    }
    return err;
}

esp_err_t navigator_load_goal(int8_t *features)
{
    nvs_handle_t handle;
    esp_err_t err = nvs_open("nav", NVS_READONLY, &handle);
    if (err != ESP_OK) return err;

    size_t required = CORR_FEAT_SIZE;
    err = nvs_get_blob(handle, "goal_feat", features, &required);
    nvs_close(handle);

    if (err == ESP_OK && required == CORR_FEAT_SIZE) {
        ESP_LOGI(TAG, "Goal features loaded from NVS (%zu bytes)", required);
    }
    return err;
}
