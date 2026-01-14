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

input_width = mp["image_input_width"]
input_height = mp["image_input_height"]
channels = mp["image_channel_count"]

print(f"Model input size: {input_width} x {input_height}")
print(f"Model channels: {channels}")
print("Expected feature length:", mp["input_features_count"])

# -----------------------
# PCA9685 setup
# -----------------------
ADDRESS = 0x40
bus = SMBus(1)   # change to 16 if needed

def init_pca9685():
    bus.write_byte_data(ADDRESS, 0x00, 0x00)
    bus.write_byte_data(ADDRESS, 0x01, 0x04)
    time.sleep(0.005)

    prescale = int(25000000 / (4096 * 1000) - 1)
    bus.write_byte_data(ADDRESS, 0x00, 0x10)
    bus.write_byte_data(ADDRESS, 0xFE, prescale)
    bus.write_byte_data(ADDRESS, 0x00, 0x00)
    time.sleep(0.005)

def set_led(ch, on):
    base = 0x06 + 4 * ch
    if on:
        bus.write_byte_data(ADDRESS, base+2, 0xFF)
        bus.write_byte_data(ADDRESS, base+3, 0x0F)
    else:
        bus.write_byte_data(ADDRESS, base+2, 0x00)
        bus.write_byte_data(ADDRESS, base+3, 0x00)

init_pca9685()

# Turn off all LEDs initially
for i in range(16):
    set_led(i, False)

# -----------------------
# Camera
# -----------------------
cap = cv2.VideoCapture(0)
print("Camera ready")
time.sleep(2)

current_led = -1

print("LED test...")
for i in range(16):
    set_led(i, True)
    time.sleep(0.1)
    set_led(i, False)
print("LED test done")

# -----------------------
# Main loop
# -----------------------
while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    features, cropped = runner.get_features_from_image(frame)
    result = runner.classify(features)
    # DEBUG: print raw result once in a while
    print(result["result"])

    if "bounding_boxes" in result["result"]:
        for bb in result["result"]["bounding_boxes"]:
            print("Model labels:", mp["labels"])
            if bb["label"] != "peace":
                continue
            if bb["value"] < 0.3:
                continue

            x_center = bb["x"] + bb["width"] / 2

            # Map X position to LED 0–15
            led = int((x_center / input_width) * 16)
            led = max(0, min(15, led))

            if led != current_led:
                if current_led >= 0:
                    set_led(current_led, False)
                set_led(led, True)
                current_led = led

    time.sleep(0.03)  # reduce flicker / CPU load
