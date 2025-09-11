#include <cstdint>
#include "hardware/gpio.h"
#include "hardware/pwm.h"
#include "led.h"

const uint LED_RED = 18;
const uint LED_GREEN = 19;
const uint LED_BLUE = 20;
const uint16_t LEVEL_STEP_SIZE = 65535 / 255;

bool ledInitialized = false;


void setupLed() {
    uint pwm_slice_red = pwm_gpio_to_slice_num(LED_RED);
    uint pwm_slice_green = pwm_gpio_to_slice_num(LED_GREEN);
    uint pwm_slice_blue = pwm_gpio_to_slice_num(LED_BLUE);

    gpio_set_function(LED_RED, GPIO_FUNC_PWM);
    gpio_set_function(LED_GREEN, GPIO_FUNC_PWM);
    gpio_set_function(LED_BLUE, GPIO_FUNC_PWM);
    
    pwm_config config = pwm_get_default_config();
    pwm_config_set_clkdiv(&config, 4);
    
    pwm_init(pwm_slice_red, &config, true);
    pwm_init(pwm_slice_green, &config, true);
    pwm_init(pwm_slice_blue, &config, true);
}


void setRgbLed(uint8_t red, uint8_t green, uint8_t blue) {
    if (!ledInitialized) {
        setupLed();
        ledInitialized = true;
    }

    pwm_set_gpio_level(LED_RED, LEVEL_STEP_SIZE * (255 - red));
    pwm_set_gpio_level(LED_GREEN, LEVEL_STEP_SIZE * (255 - green));
    pwm_set_gpio_level(LED_BLUE, LEVEL_STEP_SIZE * (255 - blue));
}
