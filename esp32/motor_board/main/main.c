/**
 * @file main.c
 * @brief Motor Board main — dual-mode: collection + navigation.
 *
 * Collection mode (default — no policy model):
 *   Pink noise explorer → SPI slave → obstacle override → motors
 *
 * Navigation mode (when policy model is flashed):
 *   SPI slave receives features → correlate → policy → motors
 *
 * Mode is auto-detected based on whether policy model is embedded.
 */
#include "config.h"
#include "motor.h"
#include "imu_uart.h"
#include "explorer.h"
#include "spi_slave.h"
#include "navigator.h"

#include "inference.h"

#if defined(TEST_MODE) && TEST_MODE
#include "test_mode.h"
#endif

#include "esp_log.h"
#include "nvs_flash.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "mot_main";

/* ═══════════════════════════════════════════════════════════════════════════
 *  Collection Mode — pink noise exploration
 * ═══════════════════════════════════════════════════════════════════════════ */

static void exploration_task(void *pvParam)
{
    ESP_LOGI(TAG, "Step-and-stop exploration started (burst=%d ms)", STEP_BURST_MS);

    while (1) {
        /* ── 1. Get next pink-noise action ── */
        uint8_t planned_action = explorer_next_action();

        /* ── 2. SPI exchange: send planned action + heading to cam board.
         *       Cam board runs depth guard during this window. ── */
        float heading = imu_get_heading();
        if (heading < 0.0f) heading = 0.0f;

        spi_slave_set_response(planned_action, heading);

        uint8_t obstacle_flag = 0;
        uint8_t msg_type = 0;

        esp_err_t ret = spi_slave_receive(
            &obstacle_flag, &msg_type, NULL, 2000
        );

        if (ret != ESP_OK) {
            ESP_LOGW(TAG, "SPI receive timeout — skipping step");
            vTaskDelay(pdMS_TO_TICKS(500));
            continue;
        }

        /* ── 3. Apply obstacle override ── */
        uint8_t final_action = planned_action;
        if (obstacle_flag && final_action == ACTION_FORWARD) {
            final_action = ACTION_TURN_RIGHT;
            ESP_LOGW(TAG, "Obstacle override! FWD → RIGHT");
        }

        /* ── 4. Execute action (blocking — 500ms burst or IMU turn) ── */
        motor_execute_action(final_action);

        /* ── 5. Post-action settle: motors are now stopped.
         *       Cam board captures the "result" frame during this window.
         *       Update SPI response with final action + post-action heading. ── */
        vTaskDelay(pdMS_TO_TICKS(50));   /* brief settle for vibration */
        spi_slave_set_response(final_action, imu_get_heading());

        /* ── 6. Log ── */
        uint32_t step = explorer_get_step();
        if (step % 20 == 0) {
            ESP_LOGI(TAG, "Step %lu: planned=%d final=%d heading=%.1f obstacle=%d",
                     (unsigned long)step, planned_action, final_action,
                     imu_get_heading(), obstacle_flag);
        }

        /* No extra delay — action burst + SPI exchange already fills the cycle */
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  App Main
 * ═══════════════════════════════════════════════════════════════════════════ */

void app_main(void)
{
    ESP_LOGI(TAG, "╔═══════════════════════════════════════╗");
    ESP_LOGI(TAG, "║   Motor Board — Dual Mode             ║");
    ESP_LOGI(TAG, "╚═══════════════════════════════════════╝");

    /* NVS */
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    /* Initialize motor driver */
    ESP_LOGI(TAG, "[1/4] Initializing motors...");
    ESP_ERROR_CHECK(motor_init());

    /* Initialize IMU */
    ESP_LOGI(TAG, "[2/4] Initializing IMU UART...");
    ESP_ERROR_CHECK(imu_init());

    /* Initialize SPI slave */
    ESP_LOGI(TAG, "[3/4] Initializing SPI slave...");
    ESP_ERROR_CHECK(spi_slave_init());

#if defined(TEST_MODE) && TEST_MODE
    /* ── TEST MODE: connect to test server instead of normal operation ── */
    ESP_LOGI(TAG, "*** TEST MODE ENABLED ***");
    test_mode_run_motor();
    ESP_LOGI(TAG, "Test mode complete — halting.");
    return;
#else
    /* Try to initialize policy — determines mode */
    bool nav_mode = false;
    ret = inference_init();
    if (ret == ESP_OK) {
        ESP_LOGI(TAG, "Policy model loaded — NAVIGATION mode");
        nav_mode = true;
    } else {
        ESP_LOGI(TAG, "No policy model — COLLECTION mode");
    }

    if (nav_mode) {
        /* Navigation mode — skip explorer, use navigator */
        ESP_LOGI(TAG, "[4/4] Starting navigator...");
        ESP_ERROR_CHECK(navigator_start(NAV_GOAL_FROM_FLASH));
    } else {
        /* Collection mode — use pink noise explorer */
        ESP_LOGI(TAG, "[4/4] Initializing explorer...");
        ESP_ERROR_CHECK(explorer_init(42));

        xTaskCreatePinnedToCore(
            exploration_task, "explore",
            8192, NULL, 5, NULL, 1
        );
    }

    ESP_LOGI(TAG, "Motor Board running (%s mode).",
             nav_mode ? "navigation" : "collection");
#endif
}
