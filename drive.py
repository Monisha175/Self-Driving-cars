"""
drive.py — Autonomous driving client for the Udacity self-driving car simulator.

Connects to the simulator via SocketIO, receives camera frames,
runs inference through the trained CNN, and sends back steering + throttle commands.

Usage:
    python drive.py model/model.h5
    python drive.py model/model.h5 run1     # record frames to run1/ folder
"""

import argparse
import base64
import os
import shutil
from datetime import datetime
from io import BytesIO

import cv2
import eventlet
import numpy as np
import socketio
from PIL import Image
from flask import Flask
from tensorflow import keras

from utils.preprocess import preprocess

# ── Config ────────────────────────────────────────────────────────────────────

MAX_SPEED    = 25.0   # mph cap (increase for highway tracks)
MIN_SPEED    = 10.0   # maintain minimum momentum
SPEED_LIMIT  = 20.0   # target cruising speed

# PID-style throttle: back off when steering hard (taking a corner)
STEER_THROTTLE_PENALTY = 0.3

# ── SocketIO server (the simulator connects TO us as a client) ────────────────

sio  = socketio.Server()
app  = Flask(__name__)
app  = socketio.WSGIApp(sio, app)

model         = None
image_folder  = None
frame_count   = 0


@sio.on("connect")
def connect(sid, environ):
    print(f"[{datetime.now():%H:%M:%S}] Simulator connected (sid={sid})")
    send_control(0.0, 0.0)   # stop the car on connect until first frame arrives


@sio.on("disconnect")
def disconnect(sid):
    print(f"[{datetime.now():%H:%M:%S}] Simulator disconnected.")


@sio.on("telemetry")
def telemetry(sid, data):
    """
    Called every simulator tick with telemetry + camera frame.

    data keys:
      - image:          base64-encoded JPEG of the center camera
      - speed:          current vehicle speed (mph)
      - steering_angle: current steering angle (for display only)
      - throttle:       current throttle (for display only)
    """
    global frame_count

    if data is None:
        send_control(0.0, 0.0)
        return

    # ── Decode camera frame ───────────────────────────────────────────────────
    img_bytes = base64.b64decode(data["image"])
    pil_img   = Image.open(BytesIO(img_bytes))
    bgr_img   = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

    # ── Preprocess (same pipeline as training) ────────────────────────────────
    processed = preprocess(bgr_img)
    batch     = np.expand_dims(processed, axis=0).astype(np.float32)

    # ── Inference ─────────────────────────────────────────────────────────────
    steering_angle = float(model.predict(batch, verbose=0)[0][0])

    # ── Throttle control ──────────────────────────────────────────────────────
    current_speed = float(data.get("speed", 0.0))
    steer_penalty = abs(steering_angle) * STEER_THROTTLE_PENALTY

    if current_speed < MIN_SPEED:
        throttle = 0.8                          # accelerate from stop
    elif current_speed > MAX_SPEED:
        throttle = -0.1                         # gentle brake
    else:
        throttle = max(0.1, 0.5 - steer_penalty)   # ease off on corners

    send_control(steering_angle, throttle)

    # ── Console output ────────────────────────────────────────────────────────
    frame_count += 1
    if frame_count % 10 == 0:
        print(f"  Frame {frame_count:05d} | "
              f"Steer: {steering_angle:+.4f} | "
              f"Throttle: {throttle:.2f} | "
              f"Speed: {current_speed:.1f} mph")

    # ── Save frame if recording ───────────────────────────────────────────────
    if image_folder is not None:
        timestamp = datetime.utcnow().strftime("%Y_%m_%d_%H_%M_%S_%f")[:-3]
        out_path  = os.path.join(image_folder, f"{timestamp}.jpg")
        cv2.imwrite(out_path, bgr_img)


def send_control(steering_angle: float, throttle: float):
    """Emit driving control command back to the simulator."""
    sio.emit("steer", data={
        "steering_angle": str(steering_angle),
        "throttle":       str(throttle),
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Autonomous driving client")
    parser.add_argument("model_path",  help="Path to trained model .h5 file")
    parser.add_argument("image_folder", nargs="?", default=None,
                        help="Optional folder to save driving frames")
    args = parser.parse_args()

    # Load model
    print(f"Loading model: {args.model_path}")
    model = keras.models.load_model(args.model_path)
    print("Model loaded. Waiting for simulator connection on port 4567...")

    # Set up image recording folder
    if args.image_folder is not None:
        if os.path.exists(args.image_folder):
            shutil.rmtree(args.image_folder)
        os.makedirs(args.image_folder)
        image_folder = args.image_folder
        print(f"Recording frames to: {image_folder}/")

    # Start server
    eventlet.wsgi.server(eventlet.listen(("", 4567)), app)
