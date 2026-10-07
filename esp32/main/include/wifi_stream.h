/**
 * @file wifi_stream.h
 * @brief WiFi station mode and TCP streaming to laptop for data collection.
 */
#ifndef WIFI_STREAM_H
#define WIFI_STREAM_H

#include "esp_err.h"
#include <stdint.h>
#include <stdbool.h>

/**
 * @brief Initialize WiFi in station mode and connect to AP.
 *
 * WiFi SSID and password are set via menuconfig / Kconfig.
 *
 * @return ESP_OK on success.
 */
esp_err_t wifi_init_sta(void);

/**
 * @brief Start a TCP server on the given port.
 *
 * Blocks until a client connects, then stores the client socket
 * for subsequent send/receive calls.
 *
 * @param port  TCP port to listen on (default: 8888).
 * @return ESP_OK on success.
 */
esp_err_t stream_server_start(uint16_t port);

/**
 * @brief Send a 48x48 RGB888 frame + metadata to the connected client.
 *
 * Packet format:
 *   [4 bytes] magic: 0x494D4731 ("IMG1")
 *   [4 bytes] frame_id (uint32, little-endian)
 *   [2 bytes] ultrasonic_cm (uint16, little-endian)
 *   [1 byte]  last_action (uint8)
 *   [6912 bytes] image data (48*48*3 RGB888)
 *   Total: 6923 bytes per packet
 *
 * @param frame_id      Monotonic frame counter.
 * @param img_rgb888    48x48x3 image buffer.
 * @param ultrasonic_cm Current ultrasonic reading.
 * @param last_action   Last action executed (0-3).
 * @return ESP_OK on success, ESP_FAIL if client disconnected.
 */
esp_err_t stream_send_frame(uint32_t frame_id, const uint8_t *img_rgb888,
                            uint16_t ultrasonic_cm, uint8_t last_action);

/**
 * @brief Receive a command byte from the connected client.
 *
 * Commands: 'F'=forward, 'L'=left, 'R'=right, 'S'=stop, 'Q'=quit
 *
 * @param[out] cmd  Received command character.
 * @return ESP_OK on success, ESP_ERR_TIMEOUT if no data within 1s.
 */
esp_err_t stream_recv_command(char *cmd);

/**
 * @brief Check if a client is currently connected.
 */
bool stream_is_connected(void);

/** Default TCP port for data collection streaming. */
#define STREAM_DEFAULT_PORT  8888

/** Packet magic number "IMG1" */
#define STREAM_MAGIC  0x494D4731

#endif /* WIFI_STREAM_H */
