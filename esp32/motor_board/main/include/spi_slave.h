/**
 * @file spi_slave.h
 * @brief SPI slave driver for Motor Board — receives data from Camera Board.
 *
 * Full-duplex SPI slave. Each transaction simultaneously:
 *   MOSI (Cam→Mot): obstacle_flag (+ features in nav mode)
 *   MISO (Mot→Cam): heading + action_taken
 */
#ifndef SPI_SLAVE_H
#define SPI_SLAVE_H

#include "esp_err.h"
#include <stdint.h>
#include <stdbool.h>

/** Data received from Camera Board (MOSI). */
typedef struct __attribute__((packed)) {
    uint8_t  obstacle_flag;        /**< 0=clear, 1=blocked */
    uint8_t  msg_type;             /**< 0=collection, 1=nav_features */
} spi_cam_header_t;

/** Data sent back to Camera Board (MISO). */
typedef struct __attribute__((packed)) {
    uint8_t  action_taken;         /**< Last action executed (0-3) */
    uint8_t  _pad;
    float    heading_deg;          /**< IMU heading [0, 360) */
} spi_slave_response_t;

/**
 * @brief Initialize SPI slave bus.
 * @return ESP_OK on success.
 */
esp_err_t spi_slave_init(void);

/**
 * @brief Prepare the MISO response for the next SPI transaction.
 *
 * Call this BEFORE the master initiates a transaction.
 * The slave hardware will clock out this data on MISO
 * while simultaneously receiving MOSI data.
 *
 * @param action   The action this board just executed.
 * @param heading  Current IMU heading.
 */
void spi_slave_set_response(uint8_t action, float heading);

/**
 * @brief Wait for one SPI transaction from the master.
 *
 * Blocks until the Camera Board initiates a transaction.
 *
 * @param out_obstacle  Receives obstacle_flag from Camera Board.
 * @param out_msg_type  Receives msg_type (0=collection, 1=nav).
 * @param out_features  If msg_type==1, receives 288 bytes of features.
 *                      Pass NULL if not expecting nav features.
 * @param timeout_ms    Max wait time.
 * @return ESP_OK on success, ESP_ERR_TIMEOUT if no transaction.
 */
esp_err_t spi_slave_receive(uint8_t *out_obstacle,
                            uint8_t *out_msg_type,
                            int8_t *out_features,
                            uint32_t timeout_ms);

#endif /* SPI_SLAVE_H */
