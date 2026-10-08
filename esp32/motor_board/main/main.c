/**
 * @file main.c
 * @brief Motor Board main — collection mode.
 *
 * Loop:
 *   1. Generate next action from pink noise explorer
 *   2. Prepare SPI response (heading + action)
 *   3. Wait for SPI transaction from Camera Board
 *   4. If obstacle_flag set → override action to TURN_RIGHT
 *   5. Execute motor action
 *   6. Loop
 *
 * The Camera Board drives the timing via SPI. This board
 * is the SPI slave — it blocks until the master transacts.
 */
#include "config.h"
#include "motor.h"
#include "imu_uart.h"
#include "explorer.h"
#include "spi_slave.h"

#include "esp_log.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "mot_main";

/**
 * @brief Collection mode exploration loop.
 *
 * Synchronized with Camera Board via SPI:
 *   - Generates next exploration action
 *   - Prepares SPI response with heading + intended action
 *   - Waits for Camera Board to initiate SPI transaction
 *   - Reads obstacle flag from Camera Board
 *   - Overrides action if obstacle detected
 *   - Executes the final action on motors
 */
static void exploration_task(void *pvParam)
{
    ESP_LOGI(TAG, "Exploration loop started");

    while (1) {
        /* 1. Generate next action from pink noise sequence */
        uint8_t planned_action = explorer_next_action();

        /* 2. Get current IMU heading */
        float heading = imu_get_heading();
        if (heading < 0.0f) heading = 0.0f;  /* IMU not ready yet */

        /* 3. Prepare SPI response — Camera Board will read this
         *    during the next transaction */
        spi_slave_set_response(planned_action, heading);

        /* 4. Wait for SPI transaction from Camera Board.
         *    This blocks until the master sends obstacle_flag. */
        uint8_t obstacle_flag = 0;
        uint8_t msg_type = 0;

        esp_err_t ret = spi_slave_receive(
            &obstacle_flag, &msg_type, NULL,
            2000   /* 2s timeout — long enough for any cycle delay */
        );

        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "SPI receive timeout — Camera Board offline?");
            /* Don't execute action if no SPI communication */
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }

        /* 5. Safety override: if depth guard reports obstacle,
         *    swap forward → turn right */
        uint8_t final_action = planned_action;
        if (obstacle_flag && final_action == ACTION_FORWARD) {
            final_action = ACTION_TURN_RIGHT;
            ESP_LOGW(TAG, "Obstacle override! FWD → RIGHT");
        }

        /* 6. Execute motor action */
        motor_execute_action(final_action);

        /* Update SPI response with the ACTUAL action taken
         * (in case it was overridden). The Camera Board will
         * read this in the NEXT transaction. */
        spi_slave_set_response(final_action, imu_get_heading());

        /* Log periodically */
        uint32_t step = explorer_get_step();
        if (step % 20 == 0) {
            ESP_LOGI(TAG, "Step %lu: action=%d heading=%.1f obstacle=%d",
                     (unsigned long)step, final_action, heading, obstacle_flag);
        }
    }
}

void app_main(void)
{
    ESP_LOGI(TAG, "╔═══════════════════════════════════════╗");
    ESP_LOGI(TAG, "║   Motor Board — Collection Mode       ║");
    ESP_LOGI(TAG, "╚═══════════════════════════════════════╝");

    /* NVS */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    /* 1. Initialize motor driver */
    ESP_LOGI(TAG, "[1/4] Initializing motors...");
    ESP_ERROR_CHECK(motor_init());

    /* 2. Initialize IMU UART (Arduino heading reader) */
    ESP_LOGI(TAG, "[2/4] Initializing IMU UART...");
    ESP_ERROR_CHECK(imu_init());

    /* 3. Initialize SPI slave */
    ESP_LOGI(TAG, "[3/4] Initializing SPI slave...");
    ESP_ERROR_CHECK(spi_slave_init());

    /* 4. Initialize pink noise explorer */
    ESP_LOGI(TAG, "[4/4] Initializing explorer...");
    ESP_ERROR_CHECK(explorer_init(42));

    /* Launch exploration loop on core 1
     * (core 0 has IMU reader task) */
    xTaskCreatePinnedToCore(
        exploration_task, "explore",
        8192,           /* stack */
        NULL,           /* param */
        5,              /* priority */
        NULL,           /* handle */
        1               /* core 1 */
    );

    ESP_LOGI(TAG, "Motor Board running. Waiting for Camera Board SPI...");
}
