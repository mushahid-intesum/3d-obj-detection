/**
 * @file camera.c
 * @brief OV3660 camera driver implementation using esp32-camera component.
 */
#include "camera.h"
#include "pin_config.h"
#include "esp_camera.h"
#include "esp_log.h"

static const char *TAG = "camera";
static camera_fb_t *s_current_fb = NULL;

esp_err_t camera_init(void)
{
    camera_config_t config = {
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

        .xclk_freq_hz = 20000000,          /* 20 MHz XCLK */
        .ledc_timer   = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_2,    /* avoid collision with motor PWM */

        .pixel_format = PIXFORMAT_RGB565,
        .frame_size   = FRAMESIZE_QVGA,    /* 320x240 */
        .jpeg_quality = 12,                /* unused for RGB565 */
        .fb_count     = 2,                 /* double-buffer DMA */
        .fb_location  = CAMERA_FB_IN_PSRAM,
        .grab_mode    = CAMERA_GRAB_LATEST,
    };

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Camera init failed: 0x%x", err);
        return err;
    }

    /* OV3660-specific: adjust for 160° fisheye lens */
    sensor_t *s = esp_camera_sensor_get();
    if (s) {
        s->set_brightness(s, 0);
        s->set_contrast(s, 0);
        s->set_saturation(s, 0);
        s->set_whitebal(s, 1);      /* auto white balance */
        s->set_awb_gain(s, 1);
        s->set_exposure_ctrl(s, 1); /* auto exposure */
        s->set_aec2(s, 1);         /* AEC DSP */
        s->set_gain_ctrl(s, 1);    /* auto gain */
        ESP_LOGI(TAG, "OV3660 sensor configured");
    }

    ESP_LOGI(TAG, "Camera initialized: 320x240 RGB565, 160° FOV");
    return ESP_OK;
}

esp_err_t camera_capture(uint8_t **buf, int *width, int *height, size_t *len)
{
    /* Release previous frame if held */
    if (s_current_fb) {
        esp_camera_fb_return(s_current_fb);
        s_current_fb = NULL;
    }

    s_current_fb = esp_camera_fb_get();
    if (!s_current_fb) {
        ESP_LOGE(TAG, "Frame capture failed");
        return ESP_FAIL;
    }

    *buf    = s_current_fb->buf;
    *width  = s_current_fb->width;
    *height = s_current_fb->height;
    *len    = s_current_fb->len;

    return ESP_OK;
}

void camera_fb_release(void)
{
    if (s_current_fb) {
        esp_camera_fb_return(s_current_fb);
        s_current_fb = NULL;
    }
}
