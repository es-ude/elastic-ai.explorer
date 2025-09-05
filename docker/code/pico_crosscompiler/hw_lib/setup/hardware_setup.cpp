#include "pico/stdlib.h"
#include "hardware/adc.h"
#include "hardware/i2c.h"
#include "hardware_setup.h"
#include "adxl345.h"

void setup_i2c0_sda_line(void) {
  gpio_set_function(0, GPIO_FUNC_I2C);
  gpio_pull_up(0);
}
void setup_i2c0_scl_line(void) {
  gpio_set_function(1, GPIO_FUNC_I2C);
  gpio_pull_up(1);
}
void setup_i2c0(void) {
  int baud_rate = 100 * 1000;
  i2c_init(i2c0, baud_rate);
  setup_i2c0_sda_line();
  setup_i2c0_scl_line();
}

void setup_i2c1_sda_line(void) {
  gpio_set_function(6, GPIO_FUNC_I2C);
  gpio_pull_up(6);
}
void setup_i2c1_scl_line(void) {
  gpio_set_function(7, GPIO_FUNC_I2C);
  gpio_pull_up(7);
}
void setup_i2c1(void) {
  int baud_rate = 100 * 1000;
  i2c_init(i2c1, baud_rate);
  setup_i2c1_sda_line();
  setup_i2c1_scl_line();
}

void initializePeripherals(void) {
  stdio_init_all();

  setup_i2c1();
}
void setup_adc_sampling_rate(uint32_t sampling_rate) {
  uint32_t clock_div = 48000000 / sampling_rate;
  adc_set_clkdiv(clock_div);
}

errorCode setup_adxl345(void) {
  int ret = 1;
  ret = adxl345_init(I2C_FOR_ADXL345);
  if (ret < 0) {
    printf("Error, the sensor ADXL345 is not responding.\r\n");
    return INIT_ERROR;
  }
  return NO_ERROR;
}
errorCode setup_adc(void) {
  adc_init();

  adc_gpio_init(26);
  adc_gpio_init(27);

  uint32_t adc_sampling_rate = 5000;
  setup_adc_sampling_rate(adc_sampling_rate);

  return NO_ERROR;
}

float adc_measure_voltage(void) {
  adc_select_input(1);
  const float conversion_factor = 3.3f / (1 << 12);
  uint16_t result = adc_read();
  return (result * conversion_factor);
}

void __not_in_flash_func(adc_capture)(uint16_t *buf, size_t count) {
  adc_fifo_setup(true, false, 0, false, false);
  adc_run(true);
  for (int i = 0; i < count; i = i + 1)
    buf[i] = adc_fifo_get_blocking();
  adc_run(false);
  adc_fifo_drain();
}

#define NSAMP 1000
void adc_print_audio_record(void) {
  uint16_t adc_buf[NSAMP];

  adc_run(false);
  adc_select_input(0);
  adc_capture(adc_buf, NSAMP);
  printf("audio:\r\n");
  for (int ii = 0; ii < NSAMP; ii++) {
    printf("%d,", adc_buf[ii]);
  }
  printf("\r\n");
  adc_run(false);
}
