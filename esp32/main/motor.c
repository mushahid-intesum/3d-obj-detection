/**
 * @file motor.c
 * @brief DRV8833 motor driver using LEDC PWM on ESP32-S3.
 *
 * Motor A (left):  IN1/IN2 direction, ENA speed
 * Motor B (right): IN3/IN4 direction, ENB speed
 *
 * Actions:
 *   0 NORTH:    both forward
 *   1 SOUTH:    both reverse
 *   2 EAST:     pivot right (left fwd, right rev)
 *   3 WEST:     pivot left  (left rev, right fwd)
 *   4 STAY:     stop
 *   5 INTERACT: stop (server handles game logic)
 */
#include "motor.h"
#include "config.h"

#include "driver/gpio.h"
#include "driver/ledc.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "motor";

/* ── PWM channels ── */

static void set_speed(uint32_t speed)
{
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0, speed);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_0);
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_1, speed);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_1);
}

/* ── Public API ── */

esp_err_t motor_init(void)
{
    /* Configure direction GPIOs */
    gpio_config_t io_conf = {
        .pin_bit_mask = (1ULL << MOTOR_IN1) | (1ULL << MOTOR_IN2) |
                        (1ULL << MOTOR_IN3) | (1ULL << MOTOR_IN4),
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    ESP_ERROR_CHECK(gpio_config(&io_conf));

    /* LEDC timer */
    ledc_timer_config_t timer = {
        .speed_mode      = LEDC_LOW_SPEED_MODE,
        .duty_resolution = (ledc_timer_bit_t)MOTOR_PWM_BITS,
        .timer_num       = LEDC_TIMER_0,
        .freq_hz         = MOTOR_PWM_FREQ_HZ,
        .clk_cfg         = LEDC_AUTO_CLK,
    };
    ESP_ERROR_CHECK(ledc_timer_config(&timer));

    /* Channel A (left motor) */
    ledc_channel_config_t ch_a = {
        .gpio_num   = MOTOR_ENA,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel    = LEDC_CHANNEL_0,
        .timer_sel  = LEDC_TIMER_0,
        .duty       = 0,
        .hpoint     = 0,
    };
    ESP_ERROR_CHECK(ledc_channel_config(&ch_a));

    /* Channel B (right motor) */
    ledc_channel_config_t ch_b = {
        .gpio_num   = MOTOR_ENB,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel    = LEDC_CHANNEL_1,
        .timer_sel  = LEDC_TIMER_0,
        .duty       = 0,
        .hpoint     = 0,
    };
    ESP_ERROR_CHECK(ledc_channel_config(&ch_b));

    motor_stop();
    ESP_LOGI(TAG, "Motor driver initialized (DRV8833)");
    return ESP_OK;
}

void motor_stop(void)
{
    set_speed(0);
}

void motor_forward(uint32_t duration_ms)
{
    gpio_set_level(MOTOR_IN1, 1); gpio_set_level(MOTOR_IN2, 0);
    gpio_set_level(MOTOR_IN3, 1); gpio_set_level(MOTOR_IN4, 0);
    set_speed(MOTOR_SPEED);
    vTaskDelay(pdMS_TO_TICKS(duration_ms));
    motor_stop();
}

void motor_reverse(uint32_t duration_ms)
{
    gpio_set_level(MOTOR_IN1, 0); gpio_set_level(MOTOR_IN2, 1);
    gpio_set_level(MOTOR_IN3, 0); gpio_set_level(MOTOR_IN4, 1);
    set_speed(MOTOR_SPEED);
    vTaskDelay(pdMS_TO_TICKS(duration_ms));
    motor_stop();
}

void motor_turn_right(uint32_t duration_ms)
{
    gpio_set_level(MOTOR_IN1, 1); gpio_set_level(MOTOR_IN2, 0);  /* left fwd  */
    gpio_set_level(MOTOR_IN3, 0); gpio_set_level(MOTOR_IN4, 1);  /* right rev */
    set_speed(MOTOR_SPEED);
    vTaskDelay(pdMS_TO_TICKS(duration_ms));
    motor_stop();
}

void motor_turn_left(uint32_t duration_ms)
{
    gpio_set_level(MOTOR_IN1, 0); gpio_set_level(MOTOR_IN2, 1);  /* left rev  */
    gpio_set_level(MOTOR_IN3, 1); gpio_set_level(MOTOR_IN4, 0);  /* right fwd */
    set_speed(MOTOR_SPEED);
    vTaskDelay(pdMS_TO_TICKS(duration_ms));
    motor_stop();
}

void motor_execute_action(uint8_t action_id)
{
    switch (action_id) {
        case ACTION_NORTH:    ESP_LOGI(TAG, "NORTH");    motor_forward(FORWARD_MS);    break;
        case ACTION_SOUTH:    ESP_LOGI(TAG, "SOUTH");    motor_reverse(FORWARD_MS);    break;
        case ACTION_EAST:     ESP_LOGI(TAG, "EAST");     motor_turn_right(TURN_MS);    break;
        case ACTION_WEST:     ESP_LOGI(TAG, "WEST");     motor_turn_left(TURN_MS);     break;
        case ACTION_STAY:     ESP_LOGI(TAG, "STAY");     motor_stop();                 break;
        case ACTION_INTERACT: ESP_LOGI(TAG, "INTERACT"); motor_stop();                 break;
        default:              ESP_LOGW(TAG, "Unknown action %d", action_id);           break;
    }
}
