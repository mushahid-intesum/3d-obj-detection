
#ifndef WIFI_STREAM_H
#define WIFI_STREAM_H

#include "esp_err.h"
#include <stdint.h>
#include <stdbool.h>


esp_err_t wifi_init_sta(void);


esp_err_t stream_server_start(uint16_t port);


void stream_accept_start(void);


esp_err_t stream_send_frame(uint32_t frame_id, uint32_t timestep,
                            uint8_t action_taken, uint8_t prox_blocked,
                            float proximity, float heading_deg,
                            const uint8_t *jpeg_buf, uint32_t jpeg_len);


bool stream_is_connected(void);


#define STREAM_MAGIC  0x494D4735


#define STREAM_HEADER_SIZE  26

#endif
