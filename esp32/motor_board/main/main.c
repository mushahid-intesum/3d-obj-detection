/**
 * @file main.c
 * @brief Motor Board main — dual-mode: collection + navigation.
 *
 * Collection mode (default — no policy model):
 *   Pink noise explorer → SPI slave → obstacle override → motors
 *
 * Navigation mode (when policy model is flashed):
 *   SPI slave receives features → correlate → policy → motors
 *
 * Mode is auto-detected based on whether policy model is embedded.
 */
#include "config.h"
#include "motor.h"
#include "imu_uart.h"
#include "explorer.h"
#include "navigator.h"

#include "camera.h"
#include "wifi_stream.h"
#include "image_proc.h"

#include "inference.h"

#if defined(TEST_MODE) && TEST_MODE
#include "test_mode.h"
#endif

#include "esp_log.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <stdlib.h> // Required for rand() and srand()
#include <time.h> 

static const char *TAG = "mot_main";

/* ═══════════════════════════════════════════════════════════════════════════
 *  Collection Mode — pink noise exploration
 * ═══════════════════════════════════════════════════════════════════════════ */

static const char *ACTION_NAMES[] = {"FWD", "RIGHT", "LEFT", "STOP"};

static int get_random_number() {
    srand(time(NULL));

    int min = 1;
    int max = 100;
    
    return (rand() % (max - min + 1)) + min;
}

static void exploration_task(void *pvParam)
{
    ESP_LOGI(TAG, "Step-and-stop exploration started (burst=%d ms)", STEP_BURST_MS);
    ESP_LOGI(TAG, "IMU status: %s (heading=%.1f°)",
             imu_is_ready() ? "READY" : "NOT READY", imu_get_heading());

    uint32_t frame_id = 0;
    uint32_t timestep = 0;

    ESP_LOGI(TAG, "Collection loop started (step-and-stop, cycle=%d ms)",
             EXPLORE_CYCLE_MS);

    while (1) {
        /* ── 1. Get next pink-noise action ── */
        uint8_t planned_action = explorer_next_action();

        float heading_before = imu_get_heading();
        if (heading_before < 0.0f) heading_before = 0.0f;

        /* 2. Depth guard — run obstacle detector on downsampled frame */
        uint8_t obstacle_flag = 0;
        // TODO: Obstacle code via supersonic

        /* ── 3. Apply obstacle override ── */
        uint8_t final_action = planned_action;
        if (obstacle_flag && final_action == ACTION_FORWARD) {

            int coinFlip = get_random_number();

            if (coinFlip > 50) {
                final_action = ACTION_TURN_RIGHT;
            } else {
                final_action = ACTION_TURN_LEFT;
            }

            ESP_LOGW(TAG, "Obstacle override!");
        }

        /* ── 4. Execute action (blocking — 500ms burst or IMU turn) ── */
        ESP_LOGI(TAG, "[pre]  heading=%.1f° action=%s",
                 heading_before,
                 final_action < 4 ? ACTION_NAMES[final_action] : "?");

        motor_execute_action(final_action);

        /* ── 5. Post-action settle: motors are now stopped.
         *       Cam board captures the "result" frame during this window.
         *       Update SPI response with final action + post-action heading. ── */
        vTaskDelay(pdMS_TO_TICKS(50));   /* brief settle for vibration */
        float heading_after = imu_get_heading();

        /* ── 6. Log heading change ── */
        float delta = heading_after - heading_before;
        if (delta > 180.0f) delta -= 360.0f;
        if (delta < -180.0f) delta += 360.0f;

        uint32_t step = explorer_get_step();
        ESP_LOGI(TAG, "[post] step=%lu heading=%.1f° Δ=%.1f° action=%s obstacle=%d",
                 (unsigned long)step, heading_after, delta,
                 final_action < 4 ? ACTION_NAMES[final_action] : "?",
                 obstacle_flag);

        camera_fb_t *fb = camera_capture_frame();
        if (!fb) {
            ESP_LOGW(TAG, "Frame capture failed, skipping");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        uint8_t action_taken = ACTION_STOP;
        float heading_deg = 0.0f;

        /* 4. Stream IMG4 to laptop (best-effort — skip if no client) */
        if (stream_is_connected()) {
            stream_send_frame(
                frame_id, timestep, action_taken, obstacle_flag,
                heading_deg, fb->buf, fb->len
            );
        }

        camera_release_frame(fb);

        frame_id++;
        timestep++;

        /* 5. Wait for motor action to complete (~500ms burst + settle)
         *    before capturing the next frame */
        vTaskDelay(pdMS_TO_TICKS(EXPLORE_CYCLE_MS));

        /* No extra delay — action burst + SPI exchange already fills the cycle */
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  App Main
 * ═══════════════════════════════════════════════════════════════════════════ */

void app_main(void)
{
    ESP_LOGI(TAG, "╔═══════════════════════════════════════╗");
    ESP_LOGI(TAG, "║   Motor Board — Dual Mode             ║");
    ESP_LOGI(TAG, "╚═══════════════════════════════════════╝");

    /* NVS */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    ESP_LOGI(TAG, "[1/5] Initializing camera...");
    ESP_ERROR_CHECK(camera_init_jpeg());

    /* Initialize motor driver */
    ESP_LOGI(TAG, "[1/4] Initializing motors...");
    ESP_ERROR_CHECK(motor_init());

    /* Initialize IMU */
    ESP_LOGI(TAG, "[2/4] Initializing IMU UART...");
    ESP_ERROR_CHECK(imu_init());

    /* Initialize Supersonic */
    ESP_LOGI(TAG, "[3/4] Initializing Supersonic...");
    ESP_ERROR_CHECK(supersonic_init());

    ESP_LOGI(TAG, "[4/5] Connecting to WiFi...");
    ESP_ERROR_CHECK(wifi_init_sta());

#if defined(TEST_MODE) && TEST_MODE
    /* ── TEST MODE: connect to test server instead of normal operation ── */
    ESP_LOGI(TAG, "*** TEST MODE ENABLED ***");
    test_mode_run_motor();
    ESP_LOGI(TAG, "Test mode complete — halting.");
    return;
#else
    /* Try to initialize policy — determines mode */
    bool nav_mode = false;
    ret = inference_init();
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Policy model loaded — NAVIGATION mode");
        nav_mode = true;
    } else {
        ESP_LOGI(TAG, "No policy model — COLLECTION mode");
    }

    ESP_LOGI(TAG, "[5/5] Starting TCP server...");
    ESP_ERROR_CHECK(stream_server_start(STREAM_DEFAULT_PORT));

    stream_accept_start();

    if (nav_mode) {
        /* Navigation mode — skip explorer, use navigator */
        ESP_LOGI(TAG, "[4/4] Starting navigator...");
        ESP_ERROR_CHECK(navigator_start(NAV_GOAL_FROM_FLASH));
    } else {
        /* Collection mode — use pink noise explorer */
        ESP_LOGI(TAG, "[4/4] Initializing explorer...");
        ESP_ERROR_CHECK(explorer_init(42));

        xTaskCreatePinnedToCore(
            exploration_task, "explore",
            8192, NULL, 5, NULL, 1
        );
    }

    ESP_LOGI(TAG, "Motor Board running (%s mode).",
             nav_mode ? "navigation" : "collection");
#endif
}
