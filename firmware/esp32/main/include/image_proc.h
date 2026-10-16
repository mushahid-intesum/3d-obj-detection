
#ifndef IMAGE_PROC_H
#define IMAGE_PROC_H

#include <stdint.h>


#define IMG_TARGET_W    96
#define IMG_TARGET_H    96
#define IMG_TARGET_CH   3
#define IMG_TARGET_SIZE (IMG_TARGET_W * IMG_TARGET_H * IMG_TARGET_CH)


void image_downsample(const uint8_t *src_rgb565, int src_w, int src_h,
                      uint8_t *dst_rgb888);


static inline void rgb565_to_rgb888(uint16_t pixel, uint8_t *r, uint8_t *g, uint8_t *b)
{

    pixel = (pixel >> 8) | (pixel << 8);
    *r = (uint8_t)((pixel >> 11) & 0x1F) << 3;
    *g = (uint8_t)((pixel >> 5)  & 0x3F) << 2;
    *b = (uint8_t)((pixel)       & 0x1F) << 3;
}

#endif
