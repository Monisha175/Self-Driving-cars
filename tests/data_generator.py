"""
data_generator.py — Keras Sequence data generator with augmentation.

Loads images on-the-fly (memory-efficient for large datasets).
Augmentation applied only during training, not validation.

Augmentation techniques:
  - Random horizontal flip + negate angle  (balance left/right turns)
  - Random brightness adjustment           (simulate day/night/shadow)
  - Random shadow overlay                  (Track 2 robustness)
  - Random horizontal translation          (simulate lane offset)
"""

import cv2
import numpy as np
from tensorflow import keras
from utils.preprocess import load_and_preprocess


class DataGenerator(keras.utils.Sequence):
    """
    Thread-safe Keras Sequence that yields (images, steering_angles) batches.

    Using Sequence (instead of a plain generator) enables:
      - Multiprocessing workers (use_multiprocessing=True in fit())
      - Proper epoch boundary handling — every sample seen exactly once per epoch
      - Automatic shuffling between epochs
    """

    def __init__(self, samples: list, batch_size: int = 32, augment: bool = True):
        """
        Args:
            samples:    list of (image_path, steering_angle) tuples
            batch_size: number of samples per batch
            augment:    if True, apply random augmentation (training only)
        """
        self.samples    = samples
        self.batch_size = batch_size
        self.augment    = augment
        self.on_epoch_end()

    def __len__(self) -> int:
        """Number of batches per epoch."""
        return int(np.ceil(len(self.samples) / self.batch_size))

    def __getitem__(self, index: int):
        """Generate one batch of (images, angles)."""
        batch = self.samples[index * self.batch_size:(index + 1) * self.batch_size]
        images, angles = [], []

        for img_path, angle in batch:
            img = load_and_preprocess(img_path)
            if self.augment:
                img, angle = self._augment(img, angle)
            images.append(img)
            angles.append(angle)

        return np.array(images, dtype=np.float32), np.array(angles, dtype=np.float32)

    def on_epoch_end(self):
        """Shuffle samples after every epoch to prevent ordering bias."""
        np.random.shuffle(self.samples)

    # ── Augmentation methods ──────────────────────────────────────────────────

    def _augment(self, image: np.ndarray, angle: float):
        """Apply a random subset of augmentations."""
        if np.random.rand() < 0.5:
            image, angle = self._flip(image, angle)
        if np.random.rand() < 0.5:
            image = self._random_brightness(image)
        if np.random.rand() < 0.4:
            image = self._random_shadow(image)
        if np.random.rand() < 0.4:
            image, angle = self._random_translate(image, angle)
        return image, angle

    @staticmethod
    def _flip(image: np.ndarray, angle: float):
        """Horizontally flip image and negate steering angle."""
        return cv2.flip(image, 1), -angle

    @staticmethod
    def _random_brightness(image: np.ndarray) -> np.ndarray:
        """
        Randomly adjust image brightness in HSV space.
        Simulates different lighting: bright sun, overcast, tunnel entrance.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_YUV2BGR)   # back to BGR first
        hsv = cv2.cvtColor(hsv, cv2.COLOR_BGR2HSV).astype(np.float32)
        factor = 0.4 + np.random.uniform()             # [0.4, 1.4] range
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * factor, 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        return cv2.cvtColor(result, cv2.COLOR_BGR2YUV)

    @staticmethod
    def _random_shadow(image: np.ndarray) -> np.ndarray:
        """
        Overlay a random polygonal shadow on part of the image.
        Dramatically improves Track 2 (jungle/bridge shadows) performance.
        """
        h, w = image.shape[:2]
        x1, x2 = np.random.randint(0, w, 2)
        shadow = image.copy()

        # Random shadow polygon covering left or right side of the image
        pts = np.array([[x1, 0], [x2, h], [0 if x1 > w // 2 else w, h],
                         [0 if x1 > w // 2 else w, 0]], dtype=np.int32)
        mask = np.zeros_like(image)
        cv2.fillPoly(mask, [pts], (50, 50, 50))

        shadow = cv2.addWeighted(image, 1.0, mask, -0.4, 0)
        return shadow

    @staticmethod
    def _random_translate(image: np.ndarray, angle: float,
                           max_shift_px: int = 50, correction_per_px: float = 0.004):
        """
        Randomly shift the image horizontally and adjust the steering angle
        proportionally — teaches recovery from lane offset.
        """
        h, w = image.shape[:2]
        shift = np.random.randint(-max_shift_px, max_shift_px)
        angle_correction = shift * correction_per_px

        M = np.float32([[1, 0, shift], [0, 1, 0]])
        translated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        return translated, angle + angle_correction
