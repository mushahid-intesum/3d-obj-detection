/**
 * @file motor.c
 * @brief Differential-drive motor control using LEDC PWM.
 */
#include "motor.h"
#include "pin_config.h"
#include "driver/ledc.h"
#include "driver/gpio.h"
#include "esp_log.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

static const char *TAG = "motor";

/** Duration (ms) for a single forward step (~0.25m at moderate speed). */
#define FORWARD_DURATION_MS   200

/** Duration (ms) for a single turn step (~30° at moderate speed). */
#define TURN_DURATION_MS      300

/**
 * @brief Configure direction GPIOs for one motor.
 */
static void setup_dir_pins(gpio_num_t in1, gpio_num_t in2)
{
    gpio_config_t conf = {
        .pin_bit_mask = (1ULL << in1) | (1ULL << in2),
        .mode         = GPIO_MODE_OUTPUT,
        .pull_up_en   = GPIO_PULLUP_DISABLE,
        .pull_down_en = GPIO_PULLDOWN_DISABLE,
        .intr_type    = GPIO_INTR_DISABLE,
    };
    gpio_config(&conf);
    gpio_set_level(in1, 0);
    gpio_set_level(in2, 0);
}

esp_err_t motor_init(void)
{
    /* Direction pins */
    setup_dir_pins(MOTOR_L_IN1, MOTOR_L_IN2);
    setup_dir_pins(MOTOR_R_IN1, MOTOR_R_IN2);

    /* LEDC timer (shared by both channels) */
    ledc_timer_config_t timer_conf = {
        .speed_mode      = LEDC_LOW_SPEED_MODE,
        .timer_num       = LEDC_TIMER_1,
        .duty_resolution = MOTOR_PWM_RESOLUTION,
        .freq_hz         = MOTOR_PWM_FREQ_HZ,
        .clk_cfg         = LEDC_AUTO_CLK,
    };
    ledc_timer_config(&timer_conf);

    /* Left motor PWM channel */
    ledc_channel_config_t left_ch = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel    = MOTOR_L_LEDC_CH,
        .timer_sel  = LEDC_TIMER_1,
        .gpio_num   = MOTOR_L_PWM,
        .duty       = 0,
        .hpoint     = 0,
    };
    ledc_channel_config(&left_ch);

    /* Right motor PWM channel */
    ledc_channel_config_t right_ch = {
        .speed_mode = LEDC_LOW_SPEED_MODE,
        .channel    = MOTOR_R_LEDC_CH,
        .timer_sel  = LEDC_TIMER_1,
        .gpio_num   = MOTOR_R_PWM,
        .duty       = 0,
        .hpoint     = 0,
    };
    ledc_channel_config(&right_ch);

    ESP_LOGI(TAG, "Motors initialized (PWM: %d Hz)", MOTOR_PWM_FREQ_HZ);
    return ESP_OK;
}

/**
 * @brief Set motor direction and speed.
 */
static void set_motor(gpio_num_t in1, gpio_num_t in2,
                      ledc_channel_t ch, int dir, uint8_t speed)
{
    if (dir > 0) {
        gpio_set_level(in1, 1);
        gpio_set_level(in2, 0);
    } else if (dir < 0) {
        gpio_set_level(in1, 0);
        gpio_set_level(in2, 1);
    } else {
        gpio_set_level(in1, 0);
        gpio_set_level(in2, 0);
    }

    ledc_set_duty(LEDC_LOW_SPEED_MODE, ch, speed);
    ledc_update_duty(LEDC_LOW_SPEED_MODE, ch);
}

void motor_forward(uint8_t speed)
{
    set_motor(MOTOR_L_IN1, MOTOR_L_IN2, MOTOR_L_LEDC_CH, +1, speed);
    set_motor(MOTOR_R_IN1, MOTOR_R_IN2, MOTOR_R_LEDC_CH, +1, speed);
}

void motor_turn_left(uint8_t speed)
{
    set_motor(MOTOR_L_IN1, MOTOR_L_IN2, MOTOR_L_LEDC_CH, -1, speed);
    set_motor(MOTOR_R_IN1, MOTOR_R_IN2, MOTOR_R_LEDC_CH, +1, speed);
}

void motor_turn_right(uint8_t speed)
{
    set_motor(MOTOR_L_IN1, MOTOR_L_IN2, MOTOR_L_LEDC_CH, +1, speed);
    set_motor(MOTOR_R_IN1, MOTOR_R_IN2, MOTOR_R_LEDC_CH, -1, speed);
}

void motor_stop(void)
{
    set_motor(MOTOR_L_IN1, MOTOR_L_IN2, MOTOR_L_LEDC_CH, 0, 0);
    set_motor(MOTOR_R_IN1, MOTOR_R_IN2, MOTOR_R_LEDC_CH, 0, 0);
}

void motor_execute_action(int action, uint8_t speed)
{
    switch (action) {
    case ACTION_FORWARD:
        motor_forward(speed);
        vTaskDelay(pdMS_TO_TICKS(FORWARD_DURATION_MS));
        motor_stop();
        break;
    case ACTION_TURN_LEFT:
        motor_turn_left(speed);
        vTaskDelay(pdMS_TO_TICKS(TURN_DURATION_MS));
        motor_stop();
        break;
    case ACTION_TURN_RIGHT:
        motor_turn_right(speed);
        vTaskDelay(pdMS_TO_TICKS(TURN_DURATION_MS));
        motor_stop();
        break;
    case ACTION_STOP:
    default:
        motor_stop();
        break;
    }
}
