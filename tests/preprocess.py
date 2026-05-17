"""
preprocess.py — Image loading and preprocessing for the self-driving CNN.

Preprocessing steps (applied to ALL images, train + inference):
  1. Load BGR image from disk
  2. Convert to YUV colour space
     - Y  = luminance (road edges, lane markings)
     - UV = chrominance (road surface colour, grass/sky distinction)
  3. Gaussian blur (reduce sensor noise)
  4. Resize to model input: 66 × 200 (NVIDIA PilotNet standard)

Crop is done inside the model as a Keras layer (GPU-accelerated).
"""

import cv2
import numpy as np


# Target size expected by the model (after Keras Cropping2D layer)
# We resize to 160×320 and let the model crop internally
MODEL_INPUT_H = 160
MODEL_INPUT_W = 320


def load_and_preprocess(image_path: str) -> np.ndarray:
    """
    Load an image from disk and apply the full preprocessing pipeline.

    Args:
        image_path: absolute or relative path to the image file.

    Returns:
        np.ndarray of shape (MODEL_INPUT_H, MODEL_INPUT_W, 3), dtype uint8, YUV.

    Raises:
        FileNotFoundError: if the image cannot be read.
    """
    img = cv2.imread(image_path.strip())
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path!r}")

    return preprocess(img)


def preprocess(img: np.ndarray) -> np.ndarray:
    """
    Apply preprocessing to an already-loaded BGR numpy array.
    Safe to call during both training and live inference from the simulator.

    Args:
        img: HxWx3 BGR image (as returned by cv2.imread or simulator feed).

    Returns:
        HxWx3 YUV image resized to (MODEL_INPUT_H, MODEL_INPUT_W).
    """
    # 1. Convert BGR → YUV
    yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)

    # 2. Light Gaussian blur to suppress sensor/compression noise
    #    Kernel size 3 is small enough not to blur lane markings
    yuv = cv2.GaussianBlur(yuv, (3, 3), sigmaX=0)

    # 3. Resize to model input dimensions
    resized = cv2.resize(yuv, (MODEL_INPUT_W, MODEL_INPUT_H),
                         interpolation=cv2.INTER_AREA)

    return resized


def decode_image_for_display(img_yuv: np.ndarray) -> np.ndarray:
    """Convert a preprocessed YUV image back to BGR for OpenCV display."""
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2BGR)
