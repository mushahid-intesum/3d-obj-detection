/**
 * @file spi_slave.c
 * @brief SPI slave driver — Motor Board side.
 *
 * Uses SPI2_HOST. Responds to transactions initiated by Camera Board.
 * Each transaction: receive obstacle/features on MOSI, send heading+action on MISO.
 */
#include "spi_slave.h"
#include "correlation.h"
#include "config.h"
#include "esp_log.h"
#include "driver/spi_slave.h"
#include "esp_heap_caps.h"
#include "freertos/FreeRTOS.h"
#include <string.h>

static const char *TAG = "spi_slave";

/*
 * Collection mode: 6-byte transactions (2B header + no features).
 * Navigation mode: 2B header + ENCODER_FEAT_RAW_SIZE bytes features.
 *
 * Start with small internal DMA buffers for collection mode.
 * Call spi_slave_resize_for_nav() before entering navigation mode.
 */
#define SPI_COLLECTION_BUF_SIZE  8   /* enough for collection (6B rounded up) */

static uint8_t *s_rx_buf = NULL;
static uint8_t *s_tx_buf = NULL;
static int s_buf_size = 0;

esp_err_t spi_slave_init(void)
{
    spi_bus_config_t bus_cfg = {
        .mosi_io_num   = SPI_SLAVE_MOSI,
        .miso_io_num   = SPI_SLAVE_MISO,
        .sclk_io_num   = SPI_SLAVE_CLK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
        .max_transfer_sz = (2 + ENCODER_FEAT_RAW_SIZE),
    };

    spi_slave_interface_config_t slave_cfg = {
        .spics_io_num = SPI_SLAVE_CS,
        .flags        = 0,
        .queue_size   = 1,
        .mode         = 0,   /* CPOL=0, CPHA=0 — must match master */
    };

    /* Start with small internal DMA buffers for collection mode */
    s_buf_size = SPI_COLLECTION_BUF_SIZE;
    s_rx_buf = (uint8_t *)heap_caps_aligned_alloc(
        32, s_buf_size, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    s_tx_buf = (uint8_t *)heap_caps_aligned_alloc(
        32, s_buf_size, MALLOC_CAP_DMA | MALLOC_CAP_INTERNAL);
    if (!s_rx_buf || !s_tx_buf) {
        ESP_LOGE(TAG, "Failed to alloc SPI buffers");
        return ESP_FAIL;
    }

    esp_err_t ret = spi_slave_initialize(SPI2_HOST, &bus_cfg, &slave_cfg,
                                          SPI_DMA_CH_AUTO);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "SPI slave init failed: 0x%x", ret);
        return ret;
    }

    /* Clear buffers */
    memset(s_rx_buf, 0, s_buf_size);
    memset(s_tx_buf, 0, s_buf_size);

    ESP_LOGI(TAG, "SPI slave initialized");
    return ESP_OK;
}

void spi_slave_set_response(uint8_t action, float heading)
{
    /*
     * MISO layout (same as spi_slave_response_t):
     *   [0] action_taken
     *   [1] pad
     *   [2..5] heading_deg (float32 LE)
     *
     * Rest of tx_buf is zero-padded.
     */
    memset(s_tx_buf, 0, s_buf_size);
    s_tx_buf[0] = action;
    s_tx_buf[1] = 0;
    memcpy(&s_tx_buf[2], &heading, sizeof(float));
}

esp_err_t spi_slave_receive(uint8_t *out_obstacle,
                            uint8_t *out_msg_type,
                            int8_t *out_features,
                            uint32_t timeout_ms)
{
    memset(s_rx_buf, 0, s_buf_size);

    spi_slave_transaction_t t = {
        .length    = s_buf_size * 8,     /* in bits */
        .tx_buffer = s_tx_buf,
        .rx_buffer = s_rx_buf,
    };

    esp_err_t ret = spi_slave_transmit(SPI2_HOST, &t,
                                        pdMS_TO_TICKS(timeout_ms));
    if (ret != ESP_OK) {
        return ret;  /* ESP_ERR_TIMEOUT or other */
    }

    /* Parse MOSI header */
    *out_obstacle = s_rx_buf[0];
    *out_msg_type = s_rx_buf[1];

    /* If nav features, copy the feature payload */
    if (*out_msg_type == 1 && out_features != NULL) {
        memcpy(out_features, &s_rx_buf[2], ENCODER_FEAT_RAW_SIZE);
    }

    return ESP_OK;
}
