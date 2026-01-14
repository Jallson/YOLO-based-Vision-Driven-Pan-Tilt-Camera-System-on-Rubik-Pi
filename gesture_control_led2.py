#!/usr/bin/env python3
import cv2
import numpy as np
import time
import signal
import sys
from smbus2 import SMBus
from edge_impulse_linux.image import ImageImpulseRunner

# -----------------------
# Graceful exit
# -----------------------
def signal_handler(sig, frame):
    print("Exiting...")
    cap.release()
    runner.stop()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# -----------------------
# Edge Impulse model
# -----------------------
MODEL_PATH = "/home/ubuntu/gestures.eim"
runner = ImageImpulseRunner(MODEL_PATH)
model_info = runner.init()
mp = model_info["model_parameters"]

input_width  = mp["image_input_width"]
input_height = mp["image_input_height"]

print(f"Model input size: {input_width} x {input_height}")

# -----------------------
# PCA9685 setup
# -----------------------
ADDRESS = 0x40
bus = SMBus(1)

def init_pca9685():
    bus.write_byte_data(ADDRESS, 0x00, 0x00)  # MODE1
    time.sleep(0.005)

    prescale = int(25000000 / (4096 * 1000) - 1)  # 1 kHz PWM
    bus.write_byte_data(ADDRESS, 0x00, 0x10)      # sleep
    bus.write_byte_data(ADDRESS, 0xFE, prescale)
    bus.write_byte_data(ADDRESS, 0x00, 0x00)      # wake
    time.sleep(0.005)

def set_led_pwm(ch, pwm):
    """ pwm: 0–4095 """
    pwm = max(0, min(4095, pwm))
    base = 0x06 + 4 * ch
    bus.write_byte_data(ADDRESS, base + 0, 0x00)
    bus.write_byte_data(ADDRESS, base + 1, 0x00)
    bus.write_byte_data(ADDRESS, base + 2, pwm & 0xFF)
    bus.write_byte_data(ADDRESS, base + 3, pwm >> 8)

def led_off(ch):
    base = 0x06 + 4 * ch
    bus.write_byte_data(ADDRESS, base + 2, 0x00)
    bus.write_byte_data(ADDRESS, base + 3, 0x00)

init_pca9685()

# Turn off all LEDs initially
for i in range(16):
    led_off(i)

# -----------------------
# Camera
# -----------------------
cap = cv2.VideoCapture(0)
print("Camera ready")
time.sleep(2)

# LED test
print("LED test...")
for i in range(16):
    set_led_pwm(i, 2048)
    time.sleep(0.08)
    led_off(i)
print("LED test done")

current_led = -1
current_pwm = 0

# -----------------------
# Main loop
# -----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    features, _ = runner.get_features_from_image(frame)
    result = runner.classify(features)

    if "bounding_boxes" not in result["result"]:
        time.sleep(0.03)
        continue

    for bb in result["result"]["bounding_boxes"]:
        if bb["label"] != "peace":
            continue
        if bb["value"] < 0.3:
            continue

        # -----------------------
        # X → LED index
        # -----------------------
        x_center = bb["x"] + bb["width"] / 2
        led = int((x_center / input_width) * 16)
        led = max(0, min(15, led))

        # -----------------------
        # Y → Brightness
        # -----------------------
        y_center = bb["y"] + bb["height"] / 2

        # Invert so top = bright
        brightness = int((1.0 - (y_center / input_height)) * 4095)
        brightness = max(50, min(4095, brightness))  # avoid fully off

        # -----------------------
        # Apply LED change
        # -----------------------
        if led != current_led:
            if current_led >= 0:
                led_off(current_led)
            current_led = led

        set_led_pwm(current_led, brightness)

    time.sleep(0.03)
