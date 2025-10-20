## DEVICE COMMUNICATION SETTINGS FOR SYSTEM TESTS##
from pathlib import Path

# Set the device path to your PICO in bootsel-mode, for Linux probably just fill in your username.
PICO_DEVICE_PATH = Path("/media/robin/RPI-RP2")

# Set the ssh params of your RPi. #TODO obscure names
RPI_HOSTNAME = "transfair"
RPI_USERNAME = "robin"

# add configurations for new devices here
