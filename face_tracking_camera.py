#!/usr/bin/env python3
import cv2
import time
import signal
import sys
from smbus2 import SMBus
from edge_impulse_linux.image import ImageImpulseRunner

# ===============================
# Graceful exit
# ===============================
def signal_handler(sig, frame):
    print("\nExiting...")
    cap.release()
    runner.stop()
    set_servo_angle(PAN_CH, 90)
    set_servo_angle(TILT_CH, 90)
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ===============================
# Edge Impulse model
# ===============================
MODEL_PATH = "/home/ubuntu/face.eim"
runner = ImageImpulseRunner(MODEL_PATH)
model_info = runner.init()
mp = model_info["model_parameters"]

W = mp["image_input_width"]
H = mp["image_input_height"]

print(f"Model input: {W}x{H}")

# ===============================
# PCA9685 / SG90 config
# ===============================
ADDRESS = 0x40
bus = SMBus(1)     # try 16 if not working

PAN_CH  = 0
TILT_CH = 1

PAN_MIN,  PAN_MAX  = 20, 160
TILT_MIN, TILT_MAX = 70, 120

SERVO_MIN_US = 500
SERVO_MAX_US = 2400

pan_angle  = 90
tilt_angle = 90

# ===============================
# PCA9685 helpers
# ===============================
def init_pca9685():
    bus.write_byte_data(ADDRESS, 0x00, 0x00)
    time.sleep(0.01)

    prescale = int(25000000 / (4096 * 50) - 1)
    oldmode = bus.read_byte_data(ADDRESS, 0x00)
    bus.write_byte_data(ADDRESS, 0x00, (oldmode & 0x7F) | 0x10)
    bus.write_byte_data(ADDRESS, 0xFE, prescale)
    bus.write_byte_data(ADDRESS, 0x00, oldmode)
    time.sleep(0.005)
    bus.write_byte_data(ADDRESS, 0x00, oldmode | 0x80)

def set_pwm(ch, on, off):
    base = 0x06 + 4 * ch
    bus.write_byte_data(ADDRESS, base+0, on & 0xFF)
    bus.write_byte_data(ADDRESS, base+1, on >> 8)
    bus.write_byte_data(ADDRESS, base+2, off & 0xFF)
    bus.write_byte_data(ADDRESS, base+3, off >> 8)

def angle_to_pwm(angle):
    angle = max(0, min(180, angle))
    pulse = SERVO_MIN_US + (angle / 180.0) * (SERVO_MAX_US - SERVO_MIN_US)
    return int(pulse * 4096 / 20000)

def set_servo_angle(ch, angle):
    set_pwm(ch, 0, angle_to_pwm(angle))

# ===============================
# Init hardware
# ===============================
init_pca9685()

# ===============================
# SG90 SERVO TEST
# ===============================
print("Running SG90 servo test...")
for _ in range(1):
    for a in range(PAN_MIN, PAN_MAX, 1):
        set_servo_angle(PAN_CH, a)
        time.sleep(0.01)
    for a in range(PAN_MAX, 90, -1):
        set_servo_angle(PAN_CH, a)
        time.sleep(0.01)

    for a in range(TILT_MIN, TILT_MAX, 1):
        set_servo_angle(TILT_CH, a)
        time.sleep(0.01)
    for a in range(TILT_MAX, 90, -1):
        set_servo_angle(TILT_CH, a)
        time.sleep(0.01)

print("Servo test done")

set_servo_angle(PAN_CH, pan_angle)
set_servo_angle(TILT_CH, tilt_angle)

# ===============================
# Camera
# ===============================
cap = cv2.VideoCapture(0)
time.sleep(2)
print("Camera ready")

# ===============================
# Face tracking loop
# ===============================
DEADZONE = 0.06
GAIN = 0.10

SMOOTHING = 0.85
MAX_STEP = 2.0

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)

    display = frame.copy()
    infer = cv2.resize(frame, (W, H))

    features, _ = runner.get_features_from_image(infer)
    result = runner.classify(features)

    scale_x = display.shape[1] / W
    scale_y = display.shape[0] / H

    if "bounding_boxes" in result["result"]:
        for bb in result["result"]["bounding_boxes"]:
            if bb["label"] != "face" or bb["value"] < 0.5:
                continue

            cx = bb["x"] + bb["width"] / 2
            cy = bb["y"] + bb["height"] / 2

            nx = (cx / W) - 0.5
            ny = (cy / H) - 0.5

            target_pan  = pan_angle
            target_tilt = tilt_angle

            if abs(nx) > DEADZONE:
                target_pan += nx * GAIN * 180

            if abs(ny) > DEADZONE:
                target_tilt -= ny * GAIN * 180

            # Clamp targets
            target_pan  = max(PAN_MIN,  min(PAN_MAX,  target_pan))
            target_tilt = max(TILT_MIN, min(TILT_MAX, target_tilt))

            # Smooth movement (low-pass filter)
            pan_angle  = pan_angle  * SMOOTHING + target_pan  * (1 - SMOOTHING)
            tilt_angle = tilt_angle * SMOOTHING + target_tilt * (1 - SMOOTHING)

            # Limit max step per frame
            pan_angle  += max(-MAX_STEP, min(MAX_STEP, target_pan  - pan_angle))
            tilt_angle += max(-MAX_STEP, min(MAX_STEP, target_tilt - tilt_angle))

            pan_angle  = max(PAN_MIN,  min(PAN_MAX,  pan_angle))
            tilt_angle = max(TILT_MIN, min(TILT_MAX, tilt_angle))

            set_servo_angle(PAN_CH, pan_angle)
            set_servo_angle(TILT_CH, tilt_angle)

            print(f"PAN={pan_angle:.1f}  TILT={tilt_angle:.1f}")

            # Draw scaled box
            x = int(bb["x"] * scale_x)
            y = int(bb["y"] * scale_y)
            w = int(bb["width"] * scale_x)
            h = int(bb["height"] * scale_y)

            cx_d = int(cx * scale_x)
            cy_d = int(cy * scale_y)

            cv2.rectangle(display, (x, y), (x+w, y+h), (0,255,0), 2)
            cv2.circle(display, (cx_d, cy_d), 5, (0,0,255), -1)

    cv2.imshow("Face Tracker", display)
    if cv2.waitKey(1) & 0xFF == 27:
        break
