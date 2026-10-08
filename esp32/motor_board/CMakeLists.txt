# Motor Board — ESP32-S3 Firmware
# Handles: DRV8833 motors, IMU UART, pink noise exploration, SPI slave
cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(motor_board)
