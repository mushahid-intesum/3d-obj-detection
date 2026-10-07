/**
 * @file camera.c
 * @brief OV3660 camera driver using espressif/esp32-camera library.
 *
 * Follows the official esp32-camera example:
 *   https://github.com/espressif/esp32-camera/tree/master
 *
 * Uses LEDC_TIMER_0 / LEDC_CHANNEL_0 for XCLK generation as per
 * the library's recommended defaults. Motor PWM uses TIMER_1.
 */
#include "camera.h"
#include "config.h"
#include "esp_camera.h"
#include "esp_log.h"
#include "driver/gpio.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "camera";

/**
 * @brief Camera config following the esp32-camera library example exactly.
 *
 * Key points from the library README:
 *   - xclk_freq_hz = 20MHz (standard for OV sensors)
 *   - LEDC_TIMER_0, LEDC_CHANNEL_0 (library default)
 *   - fb_location = CAMERA_FB_IN_PSRAM (required for >CIF)
 *   - grab_mode = CAMERA_GRAB_WHEN_EMPTY (safer for single buffer)
 *   - fb_count = 1 for non-JPEG, can be 2 for JPEG continuous mode
 */
static camera_config_t s_camera_config = {
    .pin_pwdn     = CAM_PIN_PWDN,
    .pin_reset    = CAM_PIN_RESET,
    .pin_xclk     = CAM_PIN_XCLK,
    .pin_sccb_sda = CAM_PIN_SIOD,
    .pin_sccb_scl = CAM_PIN_SIOC,

    .pin_d7       = CAM_PIN_D7,
    .pin_d6       = CAM_PIN_D6,
    .pin_d5       = CAM_PIN_D5,
    .pin_d4       = CAM_PIN_D4,
    .pin_d3       = CAM_PIN_D3,
    .pin_d2       = CAM_PIN_D2,
    .pin_d1       = CAM_PIN_D1,
    .pin_d0       = CAM_PIN_D0,
    .pin_vsync    = CAM_PIN_VSYNC,
    .pin_href     = CAM_PIN_HREF,
    .pin_pclk     = CAM_PIN_PCLK,

    /* XCLK 20MHz — standard for OV series sensors */
    .xclk_freq_hz  = CAM_XCLK_FREQ,
    .ledc_timer    = LEDC_TIMER_0,
    .ledc_channel  = LEDC_CHANNEL_0,

    .pixel_format  = PIXFORMAT_JPEG,     /* overridden per mode */
    .frame_size    = FRAMESIZE_QVGA,     /* 320×240 */

    .jpeg_quality  = CAM_JPEG_QUALITY,
    .fb_count      = CAM_FB_COUNT,
    .fb_location   = CAMERA_FB_IN_PSRAM,
    .grab_mode     = CAMERA_GRAB_WHEN_EMPTY,
    .jpeg_buffer_size = 0,               /* use default size */
};

/**
 * @brief Initialize camera with specified pixel format.
 *
 * Follows the esp32-camera README pattern:
 *   1. If PWDN pin is defined, drive it LOW to wake the camera
 *   2. Call esp_camera_init()
 *   3. Optionally tune sensor settings
 */
static esp_err_t camera_init_common(pixformat_t format)
{
#if (CAM_PIN_PWDN >= 0)
    {
        gpio_config_t pwdn_conf = {
            .pin_bit_mask = (1ULL << CAM_PIN_PWDN),
            .mode = GPIO_MODE_OUTPUT,
            .pull_up_en = GPIO_PULLUP_DISABLE,
            .pull_down_en = GPIO_PULLDOWN_DISABLE,
            .intr_type = GPIO_INTR_DISABLE,
        };
        gpio_config(&pwdn_conf);
        gpio_set_level(CAM_PIN_PWDN, 0);  /* LOW = power on */
        vTaskDelay(pdMS_TO_TICKS(10));     /* let camera wake up */
    }
#endif

    /* Set format for this init call */
    s_camera_config.pixel_format = format;

    /* For non-JPEG formats, use single buffer and GRAB_LATEST */
    if (format != PIXFORMAT_JPEG) {
        s_camera_config.fb_count = 1;
        s_camera_config.grab_mode = CAMERA_GRAB_WHEN_EMPTY;
    }

    /* Initialize camera — library handles XCLK, I2C probe, DMA */
    esp_err_t err = esp_camera_init(&s_camera_config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed: 0x%x", err);
        ESP_LOGE(TAG, "Check: camera connected? PSRAM enabled? Pins correct?");
        return err;
    }

    /* Log detected sensor */
    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        ESP_LOGI(TAG, "Detected sensor PID=0x%04x", s->id.PID);
    }

    const char *fmt_str = (format == PIXFORMAT_JPEG) ? "JPEG" : "RGB565";
    ESP_LOGI(TAG, "Camera initialized: QVGA %s, XCLK=%dMHz",
             fmt_str, CAM_XCLK_FREQ / 1000000);
    return ESP_OK;
}

esp_err_t camera_init_jpeg(void)
{
    return camera_init_common(PIXFORMAT_JPEG);
}

esp_err_t camera_init_rgb(void)
{
    return camera_init_common(PIXFORMAT_RGB565);
}

camera_fb_t *camera_capture_frame(void)
{
    return esp_camera_fb_get();
}

void camera_release_frame(camera_fb_t *fb)
{
    if (fb) {
        esp_camera_fb_return(fb);
    }
}
