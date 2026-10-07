/**
 * @file main.c
 * @brief Entry point for MCU ImageNav firmware.
 *
 * Supports two modes:
 *   MODE_COLLECT:  Data collection — stream JPEG frames to laptop (Phase 1/2)
 *   MODE_NAVIGATE: Autonomous navigation — run inference on-board (Phase 7)
 */
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "nvs_flash.h"

#include "config.h"
#include "camera.h"
#include "ultrasonic.h"
#include "motor.h"
#include "image_proc.h"
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

static const firmware_mode_t FIRMWARE_MODE = MODE_NAVIGATE;
/* ═══════════════════════════════════════════ */

/**
 * @brief Data collection task (Phase 1/2).
 *
 * Streams JPEG frames to the laptop. The laptop decodes JPEG
 * and saves as 48×48 RGB for training.
 */
static void data_collection_task(void *pvParam)
{
    ESP_LOGI(TAG, "Data collection task started");
    ESP_LOGI(TAG, "Waiting for TCP client on port %d...", STREAM_DEFAULT_PORT);

    if (stream_server_start(STREAM_DEFAULT_PORT) != ESP_OK) {
        ESP_LOGE(TAG, "Failed to start stream server");
        vTaskDelete(NULL);
        return;
    }

    uint32_t frame_id = 0;
    uint8_t last_action = ACTION_STAY;

    while (stream_is_connected()) {
        char cmd = 0;
        esp_err_t err = stream_recv_command(&cmd);

        if (err == ESP_ERR_TIMEOUT) {
            /* No command — just capture and send */
        } else if (err == ESP_OK) {
            uint16_t dist = ultrasonic_get_cached_cm();
            uint8_t action = ACTION_STAY;

            switch (cmd) {
            case 'F': action = ACTION_NORTH; break;
            case 'B': action = ACTION_SOUTH; break;
            case 'L': action = ACTION_WEST;  break;
            case 'R': action = ACTION_EAST;  break;
            case 'S': action = ACTION_STAY;  break;
            case 'Q':
                ESP_LOGI(TAG, "Quit command received");
                motor_stop();
                goto done;
            default:
                ESP_LOGW(TAG, "Unknown command: 0x%02X", cmd);
                continue;
            }

            /* Ultrasonic safety override */
            if (action == ACTION_NORTH && dist < US_OBSTACLE_CM) {
                ESP_LOGW(TAG, "Ultrasonic override! dist=%u cm", dist);
                action = ACTION_EAST;
            }

            motor_execute_action(action);
            last_action = action;
        } else {
            break;
        }

        /* Capture JPEG frame and stream */
        camera_fb_t *fb = camera_capture_frame();
        if (fb) {
            uint16_t dist = ultrasonic_get_cached_cm();
            stream_send_frame(frame_id, fb->buf, dist, last_action);
            camera_release_frame(fb);
            frame_id++;

            if (frame_id % 100 == 0) {
                ESP_LOGI(TAG, "Frames sent: %lu, dist: %u cm",
                         (unsigned long)frame_id, dist);
            }
        }
    }

done:
    motor_stop();
    ESP_LOGI(TAG, "Data collection ended. Total frames: %lu",
             (unsigned long)frame_id);
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

    /* Initialize ultrasonic + motors (always needed) */
    ESP_LOGI(TAG, "Initializing ultrasonic...");
    ESP_ERROR_CHECK(ultrasonic_init());
    ESP_ERROR_CHECK(ultrasonic_start_task());

    ESP_LOGI(TAG, "Initializing motors...");
    ESP_ERROR_CHECK(motor_init());

    /* Mode-specific startup */
    switch (FIRMWARE_MODE) {
    case MODE_COLLECT:
        /* JPEG camera for streaming */
        ESP_LOGI(TAG, "Initializing camera (JPEG)...");
        ESP_ERROR_CHECK(camera_init_jpeg());

        /* WiFi + TCP streaming to laptop */
        ESP_LOGI(TAG, "Connecting to WiFi...");
        ESP_ERROR_CHECK(wifi_init_sta());

        xTaskCreatePinnedToCore(
            data_collection_task, "collect",
            8192, NULL, 5, NULL, 1
        );
        break;

    case MODE_NAVIGATE:
        /* RGB camera for on-board inference */
        ESP_LOGI(TAG, "Initializing camera (RGB565)...");
        ESP_ERROR_CHECK(camera_init_rgb());

        ESP_LOGI(TAG, "Starting autonomous navigation...");
        ESP_LOGI(TAG, "Press GOAL button to capture target, then robot navigates.");
        navigator_start(NAV_GOAL_FROM_BUTTON);
        break;
    }
}
