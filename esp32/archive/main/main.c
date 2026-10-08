/**
 * @file main.c
 * @brief Entry point for MCU ImageNav firmware.
 *
 * Supports two modes:
 *   MODE_COLLECT:  Data collection — stream JPEG frames to laptop (Phase 1/2)
 *   MODE_NAVIGATE: Autonomous navigation — run inference on-board (Phase 7)
 *
 * Uses Arduino Nano 33 BLE Rev2 as IMU for closed-loop heading control.
 * No ultrasonic sensor — depth is learned end-to-end.
 */
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "esp_event.h"

#include "config.h"
#include "camera.h"
#include "motor.h"
#include "imu_uart.h"
#include "wifi_stream.h"
#include "navigator.h"

static const char *TAG = "main";

/* ═══════════════════════════════════════════
 *  Mode selection — change this to switch
 * ═══════════════════════════════════════════ */
typedef enum {
    MODE_COLLECT,       /* Phase 1/2: WiFi JPEG streaming to laptop */
    MODE_NAVIGATE,      /* Phase 7:   Autonomous navigation          */
} firmware_mode_t;

static const firmware_mode_t FIRMWARE_MODE = MODE_COLLECT;
/* ═══════════════════════════════════════════ */

/**
 * @brief Direction labels for logging.
 */
static const char *DIR_NAMES[COLLECT_NUM_DIRS] = {
    "N", "NE", "E", "SE", "S", "SW", "W", "NW"
};

/**
 * @brief Capture and send a burst of JPEG photos at the current heading.
 *
 * Each frame includes the true IMU heading for accurate labeling.
 */
static uint32_t capture_burst(uint32_t *frame_id, uint8_t dir_idx)
{
    uint32_t sent = 0;
    for (int p = 0; p < COLLECT_PHOTOS_PER_DIR; p++) {
        camera_fb_t *fb = camera_capture_frame();
        if (fb) {
            float heading = imu_get_heading();
            stream_send_jpeg(*frame_id, fb->buf, fb->len, dir_idx, heading);
            camera_release_frame(fb);
            (*frame_id)++;
            sent++;
        }
        if (p < COLLECT_PHOTOS_PER_DIR - 1) {
            vTaskDelay(pdMS_TO_TICKS(COLLECT_PHOTO_INTERVAL_MS));
        }
    }
    return sent;
}

/**
 * @brief Data collection task (Phase 1/2) — autonomous with IMU.
 *
 * At each position:
 *   1. Record initial heading from IMU
 *   2. Sweep 8 directions using closed-loop 45° turns
 *   3. Return to original heading (closed-loop)
 *   4. Move forward one cell
 *
 * Each photo is tagged with the TRUE IMU heading — not a guess.
 * Stops after COLLECT_MAX_POSITIONS or TCP disconnect.
 */
static void data_collection_task(void *pvParam)
{
    ESP_LOGI(TAG, "═══ Autonomous Data Collection (IMU) ═══");
    ESP_LOGI(TAG, "  %d directions × %d photos = %d photos/position",
             COLLECT_NUM_DIRS, COLLECT_PHOTOS_PER_DIR,
             COLLECT_NUM_DIRS * COLLECT_PHOTOS_PER_DIR);
    ESP_LOGI(TAG, "  Max positions: %d", COLLECT_MAX_POSITIONS);
    ESP_LOGI(TAG, "Waiting for TCP client on port %d...", STREAM_DEFAULT_PORT);

    if (stream_server_start(STREAM_DEFAULT_PORT) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start stream server");
        vTaskDelete(NULL);
        return;
    }

    uint32_t frame_id = 0;

    for (uint32_t pos = 1;
         pos <= COLLECT_MAX_POSITIONS && stream_is_connected();
         pos++) {

        /* Record starting heading */
        float start_heading = imu_get_heading();
        ESP_LOGI(TAG, "── Position %lu / %d (heading: %.1f°) ──",
                 (unsigned long)pos, COLLECT_MAX_POSITIONS, start_heading);

        /* ── 8-direction photo sweep using IMU ── */
        for (int d = 0; d < COLLECT_NUM_DIRS; d++) {
            if (d > 0) {
                /* Closed-loop 45° turn using IMU */
                float target = imu_normalize(start_heading + d * 45.0f);
                motor_turn_to_heading(target, 3.0f, 3000);
                vTaskDelay(pdMS_TO_TICKS(COLLECT_SETTLE_MS));
            }

            ESP_LOGI(TAG, "  Dir %s (%.1f°): capturing %d photos...",
                     DIR_NAMES[d], imu_get_heading(), COLLECT_PHOTOS_PER_DIR);

            capture_burst(&frame_id, (uint8_t)d);

            if (!stream_is_connected()) goto done;
        }

        /* ── Return to original heading (closed-loop) ── */
        motor_turn_to_heading(start_heading, 3.0f, 3000);
        vTaskDelay(pdMS_TO_TICKS(COLLECT_SETTLE_MS));

        ESP_LOGI(TAG, "  Sweep complete (heading: %.1f°). Total frames: %lu",
                 imu_get_heading(), (unsigned long)frame_id);

        /* ── Move forward one cell ── */
        ESP_LOGI(TAG, "  Moving forward...");
        motor_forward(FORWARD_MS);
        vTaskDelay(pdMS_TO_TICKS(COLLECT_SETTLE_MS));
    }

done:
    motor_stop();
    ESP_LOGI(TAG, "═══ Collection ended ═══");
    ESP_LOGI(TAG, "  Total frames sent: %lu", (unsigned long)frame_id);
    vTaskDelete(NULL);
}

void app_main(void)
{
    ESP_LOGI(TAG, "╔══════════════════════════════════╗");
    ESP_LOGI(TAG, "║   MCU ImageNav Firmware           ║");
    ESP_LOGI(TAG, "║   Mode: %s              ║",
             FIRMWARE_MODE == MODE_COLLECT ? "COLLECT " : "NAVIGATE");
    ESP_LOGI(TAG, "╚══════════════════════════════════╝");

    /* Initialize NVS (required for WiFi) */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    /* Network/event subsystem */
    ESP_LOGI(TAG, "Initializing network stack...");
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    /* Initialize IMU (Arduino Nano 33 BLE via UART) */
    ESP_LOGI(TAG, "Initializing IMU (UART)...");
    ESP_ERROR_CHECK(imu_init());

    /* Wait for IMU to provide valid readings */
    ESP_LOGI(TAG, "Waiting for IMU...");
    for (int i = 0; i < 50 && !imu_is_ready(); i++) {
        vTaskDelay(pdMS_TO_TICKS(100));
    }
    if (imu_is_ready()) {
        ESP_LOGI(TAG, "IMU ready — initial heading: %.1f°", imu_get_heading());
    } else {
        ESP_LOGW(TAG, "IMU not ready — will use open-loop fallback");
    }

    /* Mode-specific startup */
    switch (FIRMWARE_MODE) {
    case MODE_COLLECT:
        /* Camera FIRST — claims LEDC_TIMER_0/CH_0 for XCLK */
        ESP_LOGI(TAG, "Initializing camera (JPEG)...");
        ESP_ERROR_CHECK(camera_init_jpeg());

        /* Motors AFTER camera — uses LEDC_TIMER_1/CH_2,3 */
        ESP_LOGI(TAG, "Initializing motors...");
        ESP_ERROR_CHECK(motor_init());

        /* WiFi + TCP streaming to laptop */
        ESP_LOGI(TAG, "Connecting to WiFi...");
        ESP_ERROR_CHECK(wifi_init_sta());

        xTaskCreatePinnedToCore(
            data_collection_task, "collect",
            8192, NULL, 5, NULL, 1
        );
        break;

    case MODE_NAVIGATE:
        ESP_LOGI(TAG, "Initializing camera (RGB565)...");
        ESP_ERROR_CHECK(camera_init_rgb());

        ESP_LOGI(TAG, "Initializing motors...");
        ESP_ERROR_CHECK(motor_init());

        ESP_LOGI(TAG, "Starting autonomous navigation...");
        navigator_start(NAV_GOAL_AUTO_CAPTURE);
        break;
    }
}
