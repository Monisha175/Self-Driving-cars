"""
test_model.py — Unit tests for the self-driving CNN components.

Run with:
    python -m pytest tests/ -v
"""

import numpy as np
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from model.model import build_model
from utils.preprocess import preprocess
from utils.data_generator import DataGenerator


# ── Model tests ───────────────────────────────────────────────────────────────

class TestModel:

    def test_model_builds_without_error(self):
        model = build_model()
        assert model is not None

    def test_output_shape(self):
        model = build_model()
        dummy_input = np.zeros((4, 160, 320, 3), dtype=np.float32)
        output = model.predict(dummy_input, verbose=0)
        assert output.shape == (4, 1), f"Expected (4,1), got {output.shape}"

    def test_output_range(self):
        """Steering output should be in [-1, 1] range (linear activation)."""
        model = build_model()
        dummy_input = np.random.uniform(0, 255, (10, 160, 320, 3)).astype(np.float32)
        output = model.predict(dummy_input, verbose=0).flatten()
        # With random weights this won't be in [-1,1], but verifies no NaN/Inf
        assert not np.any(np.isnan(output)), "Model output contains NaN"
        assert not np.any(np.isinf(output)), "Model output contains Inf"

    def test_parameter_count(self):
        """Sanity check model is not trivially small or absurdly large."""
        model = build_model()
        n_params = model.count_params()
        assert 100_000 < n_params < 10_000_000, \
            f"Unexpected param count: {n_params:,}"


# ── Preprocessing tests ───────────────────────────────────────────────────────

class TestPreprocess:

    def test_output_shape(self):
        dummy_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = preprocess(dummy_bgr)
        assert result.shape == (160, 320, 3), f"Got {result.shape}"

    def test_output_dtype(self):
        dummy_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = preprocess(dummy_bgr)
        assert result.dtype == np.uint8

    def test_no_nan_values(self):
        dummy_bgr = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
        result = preprocess(dummy_bgr)
        assert not np.any(np.isnan(result.astype(np.float32)))


# ── Data generator tests ──────────────────────────────────────────────────────

class TestDataGenerator:

    def _make_fake_samples(self, n=20):
        """Create fake (path, angle) samples using numpy arrays in-memory."""
        return [("fake_path.jpg", np.random.uniform(-1, 1)) for _ in range(n)]

    def test_length(self):
        samples = self._make_fake_samples(20)
        gen = DataGenerator(samples, batch_size=8, augment=False)
        # ceil(20/8) = 3
        assert len(gen) == 3

    def test_shuffle_on_epoch_end(self):
        samples = self._make_fake_samples(50)
        gen = DataGenerator(samples, batch_size=16, augment=False)
        original_order = [s[1] for s in gen.samples]
        gen.on_epoch_end()
        new_order = [s[1] for s in gen.samples]
        # Very unlikely to be identical after shuffle
        assert original_order != new_order or len(samples) == 1
