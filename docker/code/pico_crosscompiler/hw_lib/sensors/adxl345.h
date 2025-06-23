#ifndef ADXL345_H
#define ADXL345_H

#include "pico/stdlib.h"
#include "hardware/i2c.h"

int adxl345_onBus(void);
int adxl345_init(i2c_inst_t *i2c);
void adxl345_readData(int16_t *xAccl, int16_t *yAccl, int16_t *zAccl);

#endif
