#include "adxl345.h"

#define ADXL345_SLAVE_ADDR 0x53 // when ALT is connected to GND

static i2c_inst_t *I2C_CH_ADXL345;

int adxl345_on_bus(void)
{
    int ret;
    uint8_t readbuffer[2];
    ret = i2c_read_blocking(I2C_CH_ADXL345, ADXL345_SLAVE_ADDR, readbuffer, 1, false);
    return ret;
}

int adxl345_init(i2c_inst_t *i2c)
{
    uint8_t cmdbuffer[2];

    I2C_CH_ADXL345 = i2c;

    if (adxl345_on_bus() < 0)
        return -1;

    // Select Bandwidth ate register(0x2C)
    // Normal mode, Output data rate = 100 Hz(0x0A)
    cmdbuffer[0] = 0x2C;
    cmdbuffer[1] = 0x0A;
    i2c_write_blocking(I2C_CH_ADXL345, ADXL345_SLAVE_ADDR, cmdbuffer, 2, false);

    // Select Power control register(0x2D)
    // Auto-sleep disable(0x08)
    cmdbuffer[0] = 0x2D;
    cmdbuffer[1] = 0x08;
    i2c_write_blocking(I2C_CH_ADXL345, ADXL345_SLAVE_ADDR, cmdbuffer, 2, false);

    // Select Data format register(0x31)
    // Self test disabled, 4-wire interface, Full resolution, range = +/-2g(0x08)
    cmdbuffer[0] = 0x31;
    cmdbuffer[1] = 0x08;
    i2c_write_blocking(I2C_CH_ADXL345, ADXL345_SLAVE_ADDR, cmdbuffer, 2, false);

    sleep_ms(10);

    return 1;
}

void adxl345_readData(int16_t *xAccl, int16_t *yAccl, int16_t *zAccl)
{
    uint8_t readbuffer[6];
    uint8_t cmdbuffer[2];
    int16_t acc_x, acc_y, acc_z;

    // Read 6 bytes of data from register(0x32)
    // xAccl lsb, xAccl msb, yAccl lsb, yAccl msb, zAccl lsb, zAccl msb
    cmdbuffer[0] = 0x32;
    i2c_write_blocking(I2C_CH_ADXL345, ADXL345_SLAVE_ADDR, cmdbuffer, 1, false);
    i2c_read_blocking(I2C_CH_ADXL345, ADXL345_SLAVE_ADDR, readbuffer, sizeof(readbuffer), false);

    // Convert the data to 10-bits
    acc_x = (int16_t)((readbuffer[1] & 0x03) * 256 + (readbuffer[0] & 0xFF));
    if (acc_x > 511)
        acc_x = (int16_t)(acc_x - 1024);

    acc_y = (int16_t)((readbuffer[3] & 0x03) * 256 + (readbuffer[2] & 0xFF));
    if (acc_y > 511)
    {
        acc_y = (int16_t)(acc_y - 1024);
    }

    acc_z = (int16_t)((readbuffer[5] & 0x03) * 256 + (readbuffer[4] & 0xFF));
    if (acc_z > 511)
    {
        acc_z = (int16_t)(acc_z - 1024);
    }

    *xAccl = acc_x;
    *yAccl = acc_y;
    *zAccl = acc_z;
}
