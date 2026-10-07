/**
 * @file camera.h
 * @brief OV3660 camera initialization and frame capture.
 */
#ifndef CAMERA_H
#define CAMERA_H

#include "esp_err.h"
#include <stdint.h>

/**
 * @brief Initialize the OV3660 camera module.
 *
 * Configures the camera for QVGA (320x240) RGB565 output
 * with double-buffered DMA.
 *
 * @return ESP_OK on success.
 */
esp_err_t camera_init(void);

/**
 * @brief Capture a single frame and return raw RGB565 buffer.
 *
 * Caller must call camera_fb_release() after processing.
 *
 * @param[out] buf      Pointer to the frame buffer data.
 * @param[out] width    Image width (320).
 * @param[out] height   Image height (240).
 * @param[out] len      Buffer length in bytes.
 * @return ESP_OK on success.
 */
esp_err_t camera_capture(uint8_t **buf, int *width, int *height, size_t *len);

/**
 * @brief Release the frame buffer back to the DMA pool.
 */
void camera_fb_release(void);

#endif /* CAMERA_H */
