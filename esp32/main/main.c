/**
 * @file main.c
 * @brief Entry point for MCU ImageNav firmware.
 *
 * Phase 1: Hardware validation mode.
 *   - Initializes camera, ultrasonic, motors, WiFi.
 *   - Accepts TCP commands from laptop and streams frames back.
 *   - Serves as the data collection firmware for Phase 2.
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

static const char *TAG = "main";

/** Buffer for downscaled 48x48 RGB888 image. */
static uint8_t s_img_48x48[IMG_TARGET_SIZE];

/**
 * @brief Data collection task.
 *
 * Waits for commands from the laptop, executes actions,
 * captures + downscales a frame, and streams it back.
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
        /* 1. Receive command from laptop */
        char cmd = 0;
        esp_err_t err = stream_recv_command(&cmd);

        if (err == ESP_ERR_TIMEOUT) {
            /* No command received within timeout — just capture and send frame */
        } else if (err == ESP_OK) {
            /* 2. Safety check: ultrasonic override */
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

            /* Safety override */
            if (action == ACTION_FORWARD && dist < 12) {
                ESP_LOGW(TAG, "Ultrasonic override! dist=%u cm", dist);
                action = ACTION_TURN_RIGHT;
            }

            /* 3. Execute action */
            motor_execute_action(action, 150);
            last_action = (uint8_t)action;
        } else {
            /* Client disconnected */
            break;
        }

        /* 4. Capture frame */
        uint8_t *raw_buf = NULL;
        int w = 0, h = 0;
        size_t len = 0;

        if (camera_capture(&raw_buf, &w, &h, &len) == ESP_OK) {
            /* 5. Downsample 320x240 → 48x48 */
            image_downsample(raw_buf, w, h, s_img_48x48);
            camera_fb_release();

            /* 6. Stream to laptop */
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
    ESP_LOGI(TAG, "║   MCU ImageNav — Phase 1         ║");
    ESP_LOGI(TAG, "║   Hardware Validation + Collect   ║");
    ESP_LOGI(TAG, "╚══════════════════════════════════╝");

    /* Initialize NVS (required for WiFi) */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    /* Initialize peripherals */
    ESP_LOGI(TAG, "Initializing camera...");
    ESP_ERROR_CHECK(camera_init());

    ESP_LOGI(TAG, "Initializing ultrasonic...");
    ESP_ERROR_CHECK(ultrasonic_init());
    ESP_ERROR_CHECK(ultrasonic_start_task());

    ESP_LOGI(TAG, "Initializing motors...");
    ESP_ERROR_CHECK(motor_init());

    /* Quick motor test: forward, left, right, stop */
    ESP_LOGI(TAG, "Motor self-test...");
    motor_execute_action(ACTION_FORWARD, 100);
    vTaskDelay(pdMS_TO_TICKS(500));
    motor_execute_action(ACTION_TURN_LEFT, 100);
    vTaskDelay(pdMS_TO_TICKS(500));
    motor_execute_action(ACTION_TURN_RIGHT, 100);
    vTaskDelay(pdMS_TO_TICKS(500));
    motor_stop();
    ESP_LOGI(TAG, "Motor self-test complete");

    /* Connect to WiFi */
    ESP_LOGI(TAG, "Connecting to WiFi...");
    ESP_ERROR_CHECK(wifi_init_sta());

    /* Start data collection on core 1 */
    xTaskCreatePinnedToCore(
        data_collection_task, "collect",
        8192,       /* stack: 8KB */
        NULL,
        5,          /* priority */
        NULL,
        1           /* core 1 */
    );
}
