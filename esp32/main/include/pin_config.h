/**
 * @file pin_config.h
 * @brief Pin assignments for MCU ImageNav robot.
 *
 * All pins are placeholders — update to match your wiring.
 */
#ifndef PIN_CONFIG_H
#define PIN_CONFIG_H

#include "driver/gpio.h"

/* ─── OV3660 Camera (24-pin, 160° wide-angle) ─── */
#define CAM_PIN_PWDN    GPIO_NUM_NC   /* -1 if not used   */
#define CAM_PIN_RESET   GPIO_NUM_NC   /* -1 if not used   */
#define CAM_PIN_XCLK    GPIO_NUM_XX   /* TODO: set pin    */
#define CAM_PIN_SIOD    GPIO_NUM_XX   /* I2C SDA (SCCB)   */
#define CAM_PIN_SIOC    GPIO_NUM_XX   /* I2C SCL (SCCB)   */
#define CAM_PIN_D7      GPIO_NUM_XX
#define CAM_PIN_D6      GPIO_NUM_XX
#define CAM_PIN_D5      GPIO_NUM_XX
#define CAM_PIN_D4      GPIO_NUM_XX
#define CAM_PIN_D3      GPIO_NUM_XX
#define CAM_PIN_D2      GPIO_NUM_XX
#define CAM_PIN_D1      GPIO_NUM_XX
#define CAM_PIN_D0      GPIO_NUM_XX
#define CAM_PIN_VSYNC   GPIO_NUM_XX
#define CAM_PIN_HREF    GPIO_NUM_XX
#define CAM_PIN_PCLK    GPIO_NUM_XX

/* ─── HC-SR04 Ultrasonic ─── */
#define US_TRIG_PIN     GPIO_NUM_XX   /* TODO: set pin    */
#define US_ECHO_PIN     GPIO_NUM_XX   /* TODO: set pin    */

/* ─── Motor Driver (L298N / DRV8833 / TB6612) ─── */
#define MOTOR_L_IN1     GPIO_NUM_XX   /* Left motor dir A */
#define MOTOR_L_IN2     GPIO_NUM_XX   /* Left motor dir B */
#define MOTOR_L_PWM     GPIO_NUM_XX   /* Left motor speed */
#define MOTOR_R_IN1     GPIO_NUM_XX   /* Right motor dir A*/
#define MOTOR_R_IN2     GPIO_NUM_XX   /* Right motor dir B*/
#define MOTOR_R_PWM     GPIO_NUM_XX   /* Right motor speed*/

/* ─── PWM (LEDC) config ─── */
#define MOTOR_PWM_FREQ_HZ   1000
#define MOTOR_PWM_RESOLUTION LEDC_TIMER_8_BIT   /* 0-255 duty */
#define MOTOR_L_LEDC_CH      LEDC_CHANNEL_0
#define MOTOR_R_LEDC_CH      LEDC_CHANNEL_1

/* ─── Status LED ─── */
#define LED_STATUS_PIN  GPIO_NUM_XX   /* TODO: set pin    */

/* ─── Goal trigger button ─── */
#define BTN_GOAL_PIN    GPIO_NUM_XX   /* TODO: set pin    */

#endif /* PIN_CONFIG_H */
