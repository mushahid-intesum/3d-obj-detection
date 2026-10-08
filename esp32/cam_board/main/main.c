/**
 * @file main.c
 * @brief Camera Board main — collection mode.
 *
 * Loop:
 *   1. Capture JPEG frame from OV3660
 *   2. (Future: run depth guard on captured frame)
 *   3. SPI exchange: send obstacle_flag → receive heading + action from Motor Board
 *   4. Stream IMG4 packet (frame + metadata) to laptop over WiFi
 *
 * The Motor Board drives the motors and generates exploration actions.
 * This board handles imagery and communication.
 */
#include "config.h"
#include "camera.h"
#include "wifi_stream.h"
#include "spi_master.h"

#include "esp_log.h"
#include "esp_event.h"
#include "esp_netif.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "cam_main";

/**
 * @brief Main collection loop.
 *
 * Runs until the laptop disconnects. Each cycle:
 *   - Captures a JPEG frame
 *   - TODO: Runs depth guard inference for obstacle detection
 *   - Exchanges data with Motor Board via SPI
 *   - Streams the frame + metadata to the laptop
 */
static void collection_task(void *pvParam)
{
    uint32_t frame_id = 0;
    uint32_t timestep = 0;

    ESP_LOGI(TAG, "Collection loop started (cycle=%d ms)", EXPLORE_CYCLE_MS);

    while (stream_is_connected()) {
        /* 1. Capture JPEG frame */
        camera_fb_t *fb = camera_capture_frame();
        if (!fb) {
            ESP_LOGW(TAG, "Frame capture failed, skipping");
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        /* 2. Depth guard — placeholder until model is deployed.
         *    For now, always report "clear". Once depth_guard.tflite
         *    is flashed, this will run the tiny CNN on the captured frame.
         */
        uint8_t obstacle_flag = 0;

        /* TODO Phase 2A.4: uncomment when depth guard model is ready
         * obstacle_flag = depth_guard_infer(fb) ? 1 : 0;
         */

        /* 3. SPI exchange with Motor Board */
        uint8_t action_taken = ACTION_STOP;
        float heading_deg = 0.0f;

        esp_err_t spi_ret = spi_exchange_collection(
            obstacle_flag, &action_taken, &heading_deg
        );

        if (spi_ret != ESP_OK) {
            ESP_LOGW(TAG, "SPI exchange failed, using defaults");
            action_taken = ACTION_STOP;
            heading_deg = 0.0f;
        }

        /* 4. Stream IMG4 packet to laptop */
        esp_err_t stream_ret = stream_send_frame(
            frame_id, timestep,
            action_taken, obstacle_flag,
            heading_deg,
            fb->buf, fb->len
        );

        camera_release_frame(fb);

        if (stream_ret != ESP_OK) {
            ESP_LOGW(TAG, "Stream send failed — client disconnected?");
            break;
        }

        frame_id++;
        timestep++;
        vTaskDelay(pdMS_TO_TICKS(EXPLORE_CYCLE_MS));
    }

    ESP_LOGI(TAG, "Collection ended. Frames sent: %lu", (unsigned long)frame_id);
    vTaskDelete(NULL);
}

void app_main(void)
{
    ESP_LOGI(TAG, "╔═══════════════════════════════════════╗");
    ESP_LOGI(TAG, "║   Camera Board — Collection Mode      ║");
    ESP_LOGI(TAG, "╚═══════════════════════════════════════╝");

    /* NVS — required by WiFi */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    /* Network stack */
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());

    /* 1. Initialize camera */
    ESP_LOGI(TAG, "[1/4] Initializing camera...");
    ESP_ERROR_CHECK(camera_init_jpeg());

    /* 2. Initialize SPI master */
    ESP_LOGI(TAG, "[2/4] Initializing SPI master...");
    ESP_ERROR_CHECK(spi_master_init());

    /* 3. Connect to WiFi */
    ESP_LOGI(TAG, "[3/4] Connecting to WiFi...");
    ESP_ERROR_CHECK(wifi_init_sta());

    /* 4. Start TCP server and wait for laptop */
    ESP_LOGI(TAG, "[4/4] Starting TCP server...");
    ESP_ERROR_CHECK(stream_server_start(STREAM_DEFAULT_PORT));

    /* Launch collection loop */
    xTaskCreatePinnedToCore(
        collection_task, "collection",
        8192,           /* stack */
        NULL,           /* param */
        5,              /* priority */
        NULL,           /* handle */
        0               /* core 0 — camera DMA is on core 0 */
    );

    ESP_LOGI(TAG, "Camera Board running. Waiting for data collection...");
}
