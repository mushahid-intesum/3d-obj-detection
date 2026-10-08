/**
 * @file main.c
 * @brief Camera Board main — dual-mode: collection + navigation.
 *
 * Collection mode (default):
 *   capture → depth guard → SPI exchange → stream to laptop
 *
 * Navigation mode (when encoder model is flashed):
 *   capture → downsample → encode → send features via SPI → stream telemetry
 *
 * Mode is selected based on whether the encoder model is embedded.
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

#include "inference.h"
#include "image_proc.h"

static const char *TAG = "cam_main";

/* ═══════════════════════════════════════════════════════════════════════════
 *  Collection Mode — depth guard only, no encoder
 * ═══════════════════════════════════════════════════════════════════════════ */

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

        /* 2. Depth guard — placeholder until model deployed */
        uint8_t obstacle_flag = 0;

        /* 3. SPI exchange with Motor Board */
        uint8_t action_taken = ACTION_STOP;
        float heading_deg = 0.0f;

        esp_err_t spi_ret = spi_exchange_collection(
            obstacle_flag, &action_taken, &heading_deg
        );

        if (spi_ret != ESP_OK) {
            action_taken = ACTION_STOP;
            heading_deg = 0.0f;
        }

        /* 4. Stream IMG4 to laptop */
        esp_err_t stream_ret = stream_send_frame(
            frame_id, timestep, action_taken, obstacle_flag,
            heading_deg, fb->buf, fb->len
        );

        camera_release_frame(fb);

        if (stream_ret != ESP_OK) break;

        frame_id++;
        timestep++;
        vTaskDelay(pdMS_TO_TICKS(EXPLORE_CYCLE_MS));
    }

    ESP_LOGI(TAG, "Collection ended. Frames: %lu", (unsigned long)frame_id);
    vTaskDelete(NULL);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Navigation Mode — encoder + SPI features
 * ═══════════════════════════════════════════════════════════════════════════ */

/** Frame buffer for downscaled 48x48 image. */
static uint8_t s_img_buf[IMG_TARGET_SIZE];

static void navigation_task(void *pvParam)
{
    uint32_t frame_id = 0;
    int nav_steps = 0;

    ESP_LOGI(TAG, "Navigation loop started");

    while (1) {
        /* 1. Capture frame */
        camera_fb_t *fb = camera_capture_frame();
        if (!fb) {
            vTaskDelay(pdMS_TO_TICKS(100));
            continue;
        }

        /* 2. Downsample JPEG→RGB565→48x48 RGB888 */
        image_downsample(fb->buf, fb->width, fb->height, s_img_buf);

        /* 3. Run encoder: 48x48 RGB → 3x3x32 features */
        int8_t features[288];
        esp_err_t enc_ret = inference_run_encoder(s_img_buf, features);
        if (enc_ret != ESP_OK) {
            ESP_LOGE(TAG, "Encoder failed");
            camera_release_frame(fb);
            break;
        }

        /* 4. Depth guard obstacle check (placeholder for now) */
        uint8_t obstacle_flag = 0;

        /* 5. Send features + obstacle to Motor Board via SPI */
        uint8_t action_taken = ACTION_STOP;
        float heading_deg = 0.0f;

        esp_err_t spi_ret = spi_exchange_nav_features(
            obstacle_flag, features, &action_taken, &heading_deg
        );

        if (spi_ret != ESP_OK) {
            ESP_LOGW(TAG, "SPI nav exchange failed");
        }

        /* 6. Stream telemetry to laptop (if connected) */
        if (stream_is_connected()) {
            stream_send_frame(frame_id, nav_steps, action_taken,
                              obstacle_flag, heading_deg,
                              fb->buf, fb->len);
        }

        camera_release_frame(fb);

        /* 7. Log periodically */
        if (nav_steps % 10 == 0) {
            ESP_LOGI(TAG, "Nav step %d: action=%d heading=%.1f obstacle=%d",
                     nav_steps, action_taken, heading_deg, obstacle_flag);
        }

        /* 8. Check termination */
        if (action_taken == ACTION_STOP) {
            ESP_LOGI(TAG, "Policy chose STOP — navigation complete!");
            break;
        }

        frame_id++;
        nav_steps++;

        if (nav_steps >= 200) {
            ESP_LOGW(TAG, "Max nav steps reached");
            break;
        }

        vTaskDelay(pdMS_TO_TICKS(1000 / 5));  /* 5 Hz */
    }

    ESP_LOGI(TAG, "Navigation ended after %d steps", nav_steps);
    vTaskDelete(NULL);
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  App Main
 * ═══════════════════════════════════════════════════════════════════════════ */

void app_main(void)
{
    ESP_LOGI(TAG, "╔═══════════════════════════════════════╗");
    ESP_LOGI(TAG, "║   Camera Board — Dual Mode            ║");
    ESP_LOGI(TAG, "╚═══════════════════════════════════════╝");

    /* NVS */
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

    /* Initialize camera */
    ESP_LOGI(TAG, "[1/4] Initializing camera...");
    ESP_ERROR_CHECK(camera_init_jpeg());

    /* Initialize SPI master */
    ESP_LOGI(TAG, "[2/4] Initializing SPI master...");
    ESP_ERROR_CHECK(spi_master_init());

    /* Connect to WiFi */
    ESP_LOGI(TAG, "[3/4] Connecting to WiFi...");
    ESP_ERROR_CHECK(wifi_init_sta());

    /* Start TCP server */
    ESP_LOGI(TAG, "[4/4] Starting TCP server...");
    ESP_ERROR_CHECK(stream_server_start(STREAM_DEFAULT_PORT));

    /* Try to initialize encoder — determines mode */
    bool nav_mode = false;
    ret = inference_init();
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Encoder model loaded — NAVIGATION mode");
        nav_mode = true;
    } else {
        ESP_LOGI(TAG, "No encoder model — COLLECTION mode");
    }

    if (nav_mode) {
        xTaskCreatePinnedToCore(
            navigation_task, "navigation",
            16384, NULL, 5, NULL, 0
        );
    } else {
        xTaskCreatePinnedToCore(
            collection_task, "collection",
            8192, NULL, 5, NULL, 0
        );
    }

    ESP_LOGI(TAG, "Camera Board running (%s mode).",
             nav_mode ? "navigation" : "collection");
}
