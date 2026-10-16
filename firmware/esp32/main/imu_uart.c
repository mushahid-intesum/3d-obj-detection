
#include "imu_uart.h"
#include "config.h"

#include "driver/uart.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <string.h>
#include <stdlib.h>
#include <math.h>

static const char *TAG = "imu";


#define IMU_UART_NUM      UART_NUM_1
#define IMU_UART_BAUD     115200
#define IMU_UART_BUF_SIZE 256
#define IMU_LINE_BUF_SIZE 64


static volatile float   s_heading   = -1.0f;
static volatile uint8_t s_proximity = 255;
static volatile bool    s_ready     = false;
static volatile uint32_t s_update_count = 0;


static void imu_reader_task(void *pvParam)
{
    uint8_t byte;
    char line[IMU_LINE_BUF_SIZE];
    int  line_pos = 0;
    int  valid_count = 0;

    ESP_LOGI(TAG, "IMU reader task started (UART%d, RX=GPIO%d)",
             IMU_UART_NUM, IMU_UART_RX_PIN);

    while (1) {
        int len = uart_read_bytes(IMU_UART_NUM, &byte, 1, pdMS_TO_TICKS(100));
        if (len <= 0) continue;

        if (byte == '\n' || byte == '\r') {
            if (line_pos > 0) {
                line[line_pos] = '\0';


                if (line[0] == 'H' && line[1] == ':') {

                    char *h_start = &line[2];
                    char *comma = strchr(h_start, ',');

                    float h;
                    if (comma) {

                        *comma = '\0';
                        h = strtof(h_start, NULL);
                        *comma = ',';


                        if (comma[1] == 'P' && comma[2] == ':') {
                            int p = atoi(&comma[3]);
                            if (p >= 0 && p <= 255) {
                                s_proximity = (uint8_t)p;
                            }
                        }
                    } else {

                        h = strtof(h_start, NULL);
                    }

                    if (h >= 0.0f && h < 360.0f) {
                        s_heading = h;
                        s_update_count++;
                        if (!s_ready) {
                            valid_count++;
                            if (valid_count >= 5) {
                                s_ready = true;
                                ESP_LOGI(TAG, "IMU ready -- heading: %.1f deg, proximity: %d",
                                         h, s_proximity);
                            }
                        }

                        if (s_update_count % 100 == 0) {
                            ESP_LOGI(TAG, "IMU heading: %.1f deg  prox: %d (%lu updates)",
                                     h, s_proximity,
                                     (unsigned long)s_update_count);
                        }
                    }
                }

                else if (line[0] == 'I' && line[1] == ':') {
                    ESP_LOGI(TAG, "Arduino: %s", &line[2]);
                }
                else if (line[0] == 'E' && line[1] == ':') {
                    ESP_LOGE(TAG, "Arduino: %s", &line[2]);
                }

                line_pos = 0;
            }
        } else if (line_pos < IMU_LINE_BUF_SIZE - 1) {
            line[line_pos++] = (char)byte;
        } else {
            line_pos = 0;
        }
    }
}

esp_err_t imu_init(void)
{

    uart_config_t uart_config = {
        .baud_rate  = IMU_UART_BAUD,
        .data_bits  = UART_DATA_8_BITS,
        .parity     = UART_PARITY_DISABLE,
        .stop_bits  = UART_STOP_BITS_1,
        .flow_ctrl  = UART_HW_FLOWCTRL_DISABLE,
        .source_clk = UART_SCLK_DEFAULT,
    };
    ESP_ERROR_CHECK(uart_param_config(IMU_UART_NUM, &uart_config));
    ESP_ERROR_CHECK(uart_set_pin(IMU_UART_NUM,
                                 IMU_UART_TX_PIN,
                                 IMU_UART_RX_PIN,
                                 UART_PIN_NO_CHANGE,
                                 UART_PIN_NO_CHANGE));
    ESP_ERROR_CHECK(uart_driver_install(IMU_UART_NUM,
                                        IMU_UART_BUF_SIZE, 0, 0, NULL, 0));

    ESP_LOGI(TAG, "UART%d initialized (baud=%d, RX=GPIO%d, TX=GPIO%d)",
             IMU_UART_NUM, IMU_UART_BAUD, IMU_UART_RX_PIN, IMU_UART_TX_PIN);


    xTaskCreatePinnedToCore(
        imu_reader_task, "imu_read",
        4096, NULL, 6, NULL, 0
    );

    return ESP_OK;
}

float imu_get_heading(void)
{
    return s_heading;
}

uint8_t imu_get_proximity(void)
{
    return s_proximity;
}

bool imu_proximity_blocked(void)
{
    return s_proximity < PROX_THRESHOLD;
}

bool imu_is_ready(void)
{
    return s_ready;
}

float imu_angle_diff(float target, float current)
{
    float diff = target - current;
    while (diff > 180.0f)  diff -= 360.0f;
    while (diff < -180.0f) diff += 360.0f;
    return diff;
}

float imu_normalize(float angle)
{
    while (angle < 0.0f)    angle += 360.0f;
    while (angle >= 360.0f) angle -= 360.0f;
    return angle;
}
