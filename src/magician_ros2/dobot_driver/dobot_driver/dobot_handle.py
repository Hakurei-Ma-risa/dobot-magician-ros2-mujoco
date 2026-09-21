import os

from dobot_driver.interface import Interface


# USB-UART device names are not stable across Dobot revisions or reconnects.
# Keep the upstream default for compatibility, but allow deployments to select
# a persistent /dev/serial/by-id path without editing the package again.
DOBOT_PORT = os.environ.get("DOBOT_PORT", "/dev/ttyUSB0")
print(f"[INFO] Opening Dobot Magician on {DOBOT_PORT}")
bot = Interface(DOBOT_PORT)
