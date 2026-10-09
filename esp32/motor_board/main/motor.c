/**
 * @file motor.c
 * @brief DRV8833 motor driver using LEDC PWM on ESP32-S3.
 *
 * Motor A (left):  IN1/IN2 direction, ENA speed
 * Motor B (right): IN3/IN4 direction, ENB speed
 *
 * Actions (4-action space for free exploration):
 *   0 FORWARD:    both forward
 *   1 TURN_RIGHT: pivot right (left fwd, right rev)
 *   2 TURN_LEFT:  pivot left  (left rev, right fwd)
 *   3 STOP:       stop
 */
#include "motor.h"
#include "config.h"

#include "driver/gpio.h"
#include "driver/ledc.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#include <math.h>

static const char *TAG = "motor";

/* ── PWM channels ── */

static void set_speed(uint32_t speed)
{
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_2, speed);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_2);
    ledc_set_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_3, speed);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, LEDC_CHANNEL_3);
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
        .timer_num       = LEDC_TIMER_1,
        .freq_hz         = MOTOR_PWM_FREQ_HZ,
        .clk_cfg         = LEDC_AUTO_CLK,
    };
    ESP_ERROR_CHECK(ledc_timer_config(&timer));

    /* Channel A (left motor) */
    ledc_channel_config_t ch_a = {
        .gpio_num   = MOTOR_ENA,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel    = LEDC_CHANNEL_2,
        .timer_sel  = LEDC_TIMER_1,
        .duty       = 0,
        .hpoint     = 0,
    };
    ESP_ERROR_CHECK(ledc_channel_config(&ch_a));

    /* Channel B (right motor) */
    ledc_channel_config_t ch_b = {
        .gpio_num   = MOTOR_ENB,
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel    = LEDC_CHANNEL_3,
        .timer_sel  = LEDC_TIMER_1,
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
    gpio_set_level(MOTOR_IN1, 0); gpio_set_level(MOTOR_IN2, 1);  
    gpio_set_level(MOTOR_IN3, 1); gpio_set_level(MOTOR_IN4, 0);  
    set_speed(MOTOR_SPEED);
    vTaskDelay(pdMS_TO_TICKS(duration_ms));
    motor_stop();
}

void motor_reverse(uint32_t duration_ms)
{
    gpio_set_level(MOTOR_IN1, 1); gpio_set_level(MOTOR_IN2, 0);  
    gpio_set_level(MOTOR_IN3, 0); gpio_set_level(MOTOR_IN4, 1);  
    set_speed(MOTOR_SPEED);
    vTaskDelay(pdMS_TO_TICKS(duration_ms));
    motor_stop();
}

void motor_turn_right(uint32_t duration_ms)
{
    gpio_set_level(MOTOR_IN1, 1); gpio_set_level(MOTOR_IN2, 0);  
    gpio_set_level(MOTOR_IN3, 1); gpio_set_level(MOTOR_IN4, 0);  
    set_speed(MOTOR_SPEED);
    vTaskDelay(pdMS_TO_TICKS(duration_ms));
    motor_stop();
}

void motor_turn_left(uint32_t duration_ms)
{
    gpio_set_level(MOTOR_IN1, 0); gpio_set_level(MOTOR_IN2, 1);  
    gpio_set_level(MOTOR_IN3, 0); gpio_set_level(MOTOR_IN4, 1);  
    set_speed(MOTOR_SPEED);
    vTaskDelay(pdMS_TO_TICKS(duration_ms));
    motor_stop();
}

void motor_execute_action(uint8_t action_id)
{
    switch (action_id) {
        case ACTION_FORWARD:    ESP_LOGI(TAG, "FWD");   motor_forward(FORWARD_MS);       break;
        case ACTION_TURN_RIGHT: ESP_LOGI(TAG, "RIGHT"); motor_turn_right(TURN_45_MS);    break;
        case ACTION_TURN_LEFT:  ESP_LOGI(TAG, "LEFT");  motor_turn_left(TURN_45_MS);     break;
        case ACTION_STOP:       ESP_LOGI(TAG, "STOP");  motor_stop();                    break;
        default:                ESP_LOGW(TAG, "Unknown action %d", action_id);           break;
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
 *  Closed-loop heading control (requires IMU via UART)
 * ═══════════════════════════════════════════════════════════════════════════ */
#include "imu_uart.h"

#define HEADING_TOLERANCE  3.0f    /* default degrees */
#define HEADING_POLL_MS    20      /* check heading every 20ms */

bool motor_turn_to_heading(float target_heading, float tolerance_deg,
                           uint32_t timeout_ms)
{
    if (!imu_is_ready()) {
        ESP_LOGW(TAG, "IMU not ready — falling back to timed turn");
        /* Rough fallback: assume ~180°/s turn rate */
        float diff = imu_angle_diff(target_heading, imu_get_heading());
        uint32_t ms = (uint32_t)(fabsf(diff) / 180.0f * TURN_90_MS * 2);
        if (diff > 0) motor_turn_right(ms);
        else          motor_turn_left(ms);
        return false;
    }

    target_heading = imu_normalize(target_heading);
    uint32_t elapsed = 0;

    while (elapsed < timeout_ms) {
        float current = imu_get_heading();
        float diff = imu_angle_diff(target_heading, current);

        if (fabsf(diff) <= tolerance_deg) {
            motor_stop();
            ESP_LOGI(TAG, "Heading reached: %.1f° (target %.1f°, err %.1f°)",
                     current, target_heading, diff);
            return true;
        }

        /* Turn in the shortest direction */
        if (diff > 0) {
            /* Need to turn right */
            gpio_set_level(MOTOR_IN1, 1); gpio_set_level(MOTOR_IN2, 0);
            gpio_set_level(MOTOR_IN3, 0); gpio_set_level(MOTOR_IN4, 1);
        } else {
            /* Need to turn left */
            gpio_set_level(MOTOR_IN1, 0); gpio_set_level(MOTOR_IN2, 1);
            gpio_set_level(MOTOR_IN3, 1); gpio_set_level(MOTOR_IN4, 0);
        }
        set_speed(MOTOR_SPEED);

        vTaskDelay(pdMS_TO_TICKS(HEADING_POLL_MS));
        elapsed += HEADING_POLL_MS;
    }

    motor_stop();
    ESP_LOGW(TAG, "Turn timeout! current=%.1f° target=%.1f°",
             imu_get_heading(), target_heading);
    return false;
}

bool motor_turn_relative(float delta_deg, float tolerance, uint32_t timeout_ms)
{
    float current = imu_get_heading();
    if (current < 0) {
        /* IMU not ready — open-loop fallback */
        uint32_t ms = (uint32_t)(fabsf(delta_deg) / 45.0f * TURN_45_MS);
        if (delta_deg > 0) motor_turn_right(ms);
        else               motor_turn_left(ms);
        return false;
    }
    float target = imu_normalize(current + delta_deg);
    ESP_LOGI(TAG, "Turn %.0f°: %.1f° → %.1f°", delta_deg, current, target);
    return motor_turn_to_heading(target, tolerance, timeout_ms);
}
