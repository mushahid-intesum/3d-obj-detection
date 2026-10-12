/**
 * @file main.c
 * @brief Single-board main — dual-mode: collection + navigation.
 *
 * Collection mode (default — no policy model):
 *   Pink noise explorer → sonar obstacle override → motors → stream to laptop
 *
 * Navigation mode (when policy model is flashed):
 *   Camera → student model inference → sonar safety override → motors
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

#include "esp_log.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <stdlib.h>
#include <time.h>

static const char *TAG = "main";

/* Forward declaration — sonar driver (Phase 1B) */
extern esp_err_t sonar_init(void);
extern float     sonar_get_distance_cm(void);
extern bool      sonar_is_blocked(void);

/* ═══════════════════════════════════════════════════════════════════════════
 *  Collection Mode — pink noise exploration
 * ═══════════════════════════════════════════════════════════════════════════ */

static const char *ACTION_NAMES[] = {"FWD", "RIGHT", "LEFT", "STOP"};

static void exploration_task(void *pvParam)
{
    ESP_LOGI(TAG, "Step-and-stop exploration started (burst=%d ms)", STEP_BURST_MS);
    ESP_LOGI(TAG, "IMU status: %s (heading=%.1f°)",
             imu_is_ready() ? "READY" : "NOT READY", imu_get_heading());

    uint32_t frame_id = 0;
    uint32_t timestep = 0;

    srand((unsigned int)time(NULL));

    ESP_LOGI(TAG, "Collection loop started (step-and-stop, cycle=%d ms)",
             EXPLORE_CYCLE_MS);

    while (1) {
        /* ── 1. Get next pink-noise action ── */
        uint8_t planned_action = explorer_next_action();

        float heading_before = imu_get_heading();
        if (heading_before < 0.0f) heading_before = 0.0f;

        /* ── 2. Read sonar distance ── */
        float sonar_cm = sonar_get_distance_cm();
        uint8_t sonar_blocked = (sonar_cm < SONAR_THRESHOLD_CM) ? 1 : 0;

        /* ── 3. Apply obstacle override ── */
        uint8_t final_action = planned_action;
        if (sonar_blocked && final_action == ACTION_FORWARD) {
            final_action = (rand() % 2 == 0) ? ACTION_TURN_RIGHT : ACTION_TURN_LEFT;
            ESP_LOGW(TAG, "Sonar override! dist=%.1f cm", sonar_cm);
        }

        /* ── 4. Execute action (blocking — 500ms burst or IMU turn) ── */
        ESP_LOGI(TAG, "[pre]  heading=%.1f° action=%s sonar=%.1f%s",
                 heading_before,
                 final_action < 4 ? ACTION_NAMES[final_action] : "?",
                 sonar_cm,
                 sonar_blocked ? " BLOCKED" : "");

        motor_execute_action(final_action);

        /* ── 5. Post-action settle ── */
        vTaskDelay(pdMS_TO_TICKS(50));
        float heading_after = imu_get_heading();
        if (heading_after < 0.0f) heading_after = 0.0f;

        /* ── 6. Log heading change ── */
        float delta = heading_after - heading_before;
        if (delta > 180.0f) delta -= 360.0f;
        if (delta < -180.0f) delta += 360.0f;

        uint32_t step = explorer_get_step();
        ESP_LOGI(TAG, "[post] step=%lu heading=%.1f° Δ=%.1f° action=%s sonar=%.1f",
                 (unsigned long)step, heading_after, delta,
                 final_action < 4 ? ACTION_NAMES[final_action] : "?",
                 sonar_cm);

        /* ── 7. Capture post-action frame ── */
        camera_fb_t *fb = camera_capture_frame();
        if (!fb) {
            ESP_LOGW(TAG, "Frame capture failed, skipping");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        /* ── 8. Stream to laptop (best-effort — skip if no client) ── */
        if (stream_is_connected()) {
            stream_send_frame(
                frame_id, timestep, final_action, sonar_blocked,
                heading_after, fb->buf, fb->len
            );
        }

        camera_release_frame(fb);

        frame_id++;
        timestep++;

        /* ── 9. Cycle delay ── */
        vTaskDelay(pdMS_TO_TICKS(EXPLORE_CYCLE_MS));
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  App Main
 * ═══════════════════════════════════════════════════════════════════════════ */

void app_main(void)
{
    ESP_LOGI(TAG, "╔═══════════════════════════════════════╗");
    ESP_LOGI(TAG, "║   ImageNav — Single Board             ║");
    ESP_LOGI(TAG, "╚═══════════════════════════════════════╝");

    /* NVS */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    /* ── 1. Camera (must be before motors — LEDC timer conflict) ── */
    ESP_LOGI(TAG, "[1/5] Initializing camera...");
    ESP_ERROR_CHECK(camera_init_jpeg());

    /* ── 2. Motors ── */
    ESP_LOGI(TAG, "[2/5] Initializing motors...");
    ESP_ERROR_CHECK(motor_init());

    /* ── 3. IMU ── */
    ESP_LOGI(TAG, "[3/5] Initializing IMU UART...");
    ESP_ERROR_CHECK(imu_init());

    /* ── 4. Sonar ── */
    ESP_LOGI(TAG, "[4/5] Initializing HC-SR04 sonar...");
    ESP_ERROR_CHECK(sonar_init());

    /* ── 5. WiFi ── */
    ESP_LOGI(TAG, "[5/5] Connecting to WiFi...");
    ESP_ERROR_CHECK(wifi_init_sta());

    /* Try to initialize policy — determines mode */
    bool nav_mode = false;
    ret = inference_init();
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Policy model loaded — NAVIGATION mode");
        nav_mode = true;
    } else {
        ESP_LOGI(TAG, "No policy model — COLLECTION mode");
    }

    /* Start TCP server for streaming */
    ESP_ERROR_CHECK(stream_server_start(STREAM_DEFAULT_PORT));
    stream_accept_start();

    if (nav_mode) {
        /* Navigation mode — use student model */
        ESP_LOGI(TAG, "Starting navigator...");
        ESP_ERROR_CHECK(navigator_start(NAV_GOAL_FROM_FLASH));
    } else {
        /* Collection mode — use pink noise explorer */
        ESP_LOGI(TAG, "Starting explorer...");
        ESP_ERROR_CHECK(explorer_init(42));

        xTaskCreatePinnedToCore(
            exploration_task, "explore",
            8192, NULL, 5, NULL, 1
        );
    }

    ESP_LOGI(TAG, "Board running (%s mode).",
             nav_mode ? "navigation" : "collection");
}
