#ifndef HARDWARE_SETUP_H
#define HARDWARE_SETUP_H

#include <stdint.h>
#include <stdio.h>

#ifndef PICO_TINY2040_LEDS
#define PICO_TINY2040_LEDS 1
#define PICO_DEFAULT_LED_PIN_R 18
#define PICO_DEFAULT_LED_PIN_G 19
#define PICO_DEFAULT_LED_PIN_B 20
#endif
#define I2C_FOR_PAC1934 i2c1
#define I2C_FOR_SHT31 i2c1
#define I2C_FOR_ADXL345 i2c1

typedef uint8_t errorCode;
enum {
  NO_ERROR = 0x00,
  INIT_ERROR = 0x10,
  UNKNOWN_ERROR = 0x99,
};

void setup_i2c0_sda_line(void);
void setup_i2c0_scl_line(void);
void setup_i2c0(void);

void setup_i2c1_sda_line(void);
void setup_i2c1_scl_line(void);
void setup_i2c1(void);

void initializePeripherals(void);
void setup_adc_sampling_rate(uint32_t sampling_rate);

errorCode setup_pac193x(void);
errorCode setup_sht31(void);
errorCode setup_adxl345(void);
errorCode setup_adc(void);

float adc_measure_voltage(void);
void adc_print_audio_record();

#endif
