
#ifndef EXPLORER_H
#define EXPLORER_H

#include "esp_err.h"
#include <stdint.h>


esp_err_t explorer_init(uint32_t seed);


uint8_t explorer_next_action(void);


uint32_t explorer_get_step(void);


void explorer_reset(void);

#endif
