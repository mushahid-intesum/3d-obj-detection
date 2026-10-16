
#ifndef CAMERA_H
#define CAMERA_H

#include "esp_camera.h"
#include "esp_err.h"
#include <stdint.h>


esp_err_t camera_init_jpeg(void);


esp_err_t camera_init_rgb(void);


camera_fb_t *camera_capture_frame(void);


void camera_release_frame(camera_fb_t *fb);

#endif
