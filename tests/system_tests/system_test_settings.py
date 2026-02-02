
# TODO go back to toml
## DEVICE COMMUNICATION SETTINGS FOR SYSTEM TESTS##
from pathlib import Path

# Set the device path to your PICO in bootsel-mode, for Linux probably just fill in your username.
PICO_DEVICE_PATH = Path("/media/<host>/RPI-RP2")

# Set the ssh params of your RPi.
RPI_USERNAME = "<user>"
RPI_HOSTNAME = "<host>"


# add configurations for new devices here
