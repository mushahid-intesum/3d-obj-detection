# Camera Board — ESP32-S3 Firmware
# Handles: OV3660 camera, depth guard inference, WiFi streaming, SPI master
cmake_minimum_required(VERSION 3.16)

include($ENV{IDF_PATH}/tools/cmake/project.cmake)
project(cam_board)
