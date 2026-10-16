
#ifndef CONFIG_H
#define CONFIG_H


#define MOTOR_IN1       38
#define MOTOR_IN2       39
#define MOTOR_IN3       40
#define MOTOR_IN4       41
#define MOTOR_ENA       42
#define MOTOR_ENB       14


#define CAM_PIN_PWDN    (-1)
#define CAM_PIN_RESET   (-1)
#define CAM_PIN_XCLK    15
#define CAM_PIN_SIOD    4
#define CAM_PIN_SIOC    5
#define CAM_PIN_D7      16
#define CAM_PIN_D6      17
#define CAM_PIN_D5      18
#define CAM_PIN_D4      12
#define CAM_PIN_D3      10
#define CAM_PIN_D2      8
#define CAM_PIN_D1      9
#define CAM_PIN_D0      11
#define CAM_PIN_VSYNC   6
#define CAM_PIN_HREF    7
#define CAM_PIN_PCLK    13

#define CAM_XCLK_FREQ   20000000
#define CAM_FB_COUNT     1
#define CAM_JPEG_QUALITY 12


#define MOTOR_PWM_FREQ_HZ  1000
#define MOTOR_PWM_BITS     8
#define LEFT_MOTOR_SPEED       100
#define RIGHT_MOTOR_SPEED      90


#define STEP_BURST_MS   500
#define TURN_90_MS      350
#define TURN_45_MS      175
#define TURN_ANGLE_DEG  45.0f
#define TURN_TOLERANCE  3.0f
#define TURN_TIMEOUT_MS 2000


#define IMU_UART_RX_PIN   19
#define IMU_UART_TX_PIN   21


#define PROX_THRESHOLD       50


#define EXPLORE_CYCLE_MS     1500
#define EXPLORE_SEQ_LENGTH   2048


#define ACTION_FORWARD    0
#define ACTION_TURN_RIGHT 1
#define ACTION_TURN_LEFT  2
#define ACTION_STOP       3
#define NUM_ACTIONS       4


#define NAV_MAX_STEPS   200
#define NAV_RATE_HZ     5


#define STREAM_DEFAULT_PORT  8888

#endif
