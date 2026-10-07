/**
 * @file main.c
 * @brief Entry point for MCU ImageNav firmware.
 *
 * Supports two modes:
 *   MODE_COLLECT:  Data collection — stream frames to laptop (Phase 1/2)
 *   MODE_NAVIGATE: Autonomous navigation — run inference on-board (Phase 7)
 */
#include <stdio.h>
#include <string.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_log.h"
#include "nvs_flash.h"

#include "pin_config.h"
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
    MODE_COLLECT,       /* Phase 1/2: WiFi streaming to laptop */
    MODE_NAVIGATE,      /* Phase 7:   Autonomous navigation    */
} firmware_mode_t;

static const firmware_mode_t FIRMWARE_MODE = MODE_NAVIGATE;
/* ═══════════════════════════════════════════ */

/** Buffer for downscaled 48x48 RGB888 image (collection mode). */
static uint8_t s_img_48x48[IMG_TARGET_SIZE];

/**
 * @brief Data collection task (Phase 1/2).
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
    uint8_t last_action = ACTION_STOP;

    while (stream_is_connected()) {
        char cmd = 0;
        esp_err_t err = stream_recv_command(&cmd);

        if (err == ESP_ERR_TIMEOUT) {
            /* No command — just capture and send */
        } else if (err == ESP_OK) {
            uint16_t dist = ultrasonic_get_cached_cm();
            int action = ACTION_STOP;

            switch (cmd) {
            case 'F': action = ACTION_FORWARD;    break;
            case 'L': action = ACTION_TURN_LEFT;  break;
            case 'R': action = ACTION_TURN_RIGHT; break;
            case 'S': action = ACTION_STOP;       break;
            case 'Q':
                ESP_LOGI(TAG, "Quit command received");
                motor_stop();
                goto done;
            default:
                ESP_LOGW(TAG, "Unknown command: 0x%02X", cmd);
                continue;
            }

            if (action == ACTION_FORWARD && dist < 12) {
                ESP_LOGW(TAG, "Ultrasonic override! dist=%u cm", dist);
                action = ACTION_TURN_RIGHT;
            }

            motor_execute_action(action, 150);
            last_action = (uint8_t)action;
        } else {
            break;
        }

        uint8_t *raw_buf = NULL;
        int w = 0, h = 0;
        size_t len = 0;

        if (camera_capture(&raw_buf, &w, &h, &len) == ESP_OK) {
            image_downsample(raw_buf, w, h, s_img_48x48);
            camera_fb_release();

            uint16_t dist = ultrasonic_get_cached_cm();
            stream_send_frame(frame_id, s_img_48x48, dist, last_action);
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

    /* Initialize peripherals (always needed) */
    ESP_LOGI(TAG, "Initializing camera...");
    ESP_ERROR_CHECK(camera_init());

    ESP_LOGI(TAG, "Initializing ultrasonic...");
    ESP_ERROR_CHECK(ultrasonic_init());
    ESP_ERROR_CHECK(ultrasonic_start_task());

    ESP_LOGI(TAG, "Initializing motors...");
    ESP_ERROR_CHECK(motor_init());

    /* Mode-specific startup */
    switch (FIRMWARE_MODE) {
    case MODE_COLLECT:
        /* WiFi + TCP streaming to laptop */
        ESP_LOGI(TAG, "Connecting to WiFi...");
        ESP_ERROR_CHECK(wifi_init_sta());

        xTaskCreatePinnedToCore(
            data_collection_task, "collect",
            8192, NULL, 5, NULL, 1
        );
        break;

    case MODE_NAVIGATE:
        /* Autonomous navigation — no WiFi needed */
        ESP_LOGI(TAG, "Starting autonomous navigation...");
        ESP_LOGI(TAG, "Press GOAL button to capture target, then robot navigates.");
        navigator_start(NAV_GOAL_FROM_BUTTON);
        break;
    }
}
