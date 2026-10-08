/**
 * @file wifi_stream.h
 * @brief WiFi station mode and TCP streaming to laptop for data collection.
 *
 * Streams raw JPEG frames with a simple header. The server handles
 * resizing and depth map generation (MiDaS) for training.
 */
#ifndef WIFI_STREAM_H
#define WIFI_STREAM_H

#include "esp_err.h"
#include <stdint.h>
#include <stdbool.h>

/**
 * @brief Initialize WiFi in station mode and connect to AP.
 * @return ESP_OK on success.
 */
esp_err_t wifi_init_sta(void);

/**
 * @brief Start a TCP server on the given port.
 * Blocks until a client connects.
 * @param port  TCP port to listen on.
 * @return ESP_OK on success.
 */
esp_err_t stream_server_start(uint16_t port);

/**
 * @brief Send a JPEG frame + metadata to the connected client.
 *
 * Packet format:
 *   [0..3]   magic:     0x494D4732 ("IMG2")
 *   [4..7]   frame_id   (uint32, LE)
 *   [8]      dir_index  (uint8, 0-7 for N,NE,E,SE,S,SW,W,NW)
 *   [9..12]  jpeg_len   (uint32, LE)
 *   [13..N]  jpeg_data  (variable length)
 *
 * @param frame_id   Monotonic frame counter.
 * @param jpeg_buf   Raw JPEG buffer from camera.
 * @param jpeg_len   Length of JPEG data in bytes.
 * @param dir_index  Direction index (0-7).
 * @return ESP_OK on success, ESP_FAIL if client disconnected.
 */
esp_err_t stream_send_jpeg(uint32_t frame_id, const uint8_t *jpeg_buf,
                           uint32_t jpeg_len, uint8_t dir_index);

/**
 * @brief Check if a client is currently connected.
 */
bool stream_is_connected(void);

/** Default TCP port for data collection streaming. */
#define STREAM_DEFAULT_PORT  8888

/** Packet magic number "IMG2" (v2 — JPEG, no ultrasonic) */
#define STREAM_MAGIC  0x494D4732

#endif /* WIFI_STREAM_H */
