"""Unit tests for OpenCV/Pillow image preprocessor."""

import os
import tempfile
import pytest
import numpy as np
from PIL import Image

from app.pipeline.preprocessor import (
    OpenCVPreprocessor,
    ImageValidationError,
)


@pytest.fixture
def preprocessor():
    return OpenCVPreprocessor()


def create_test_image(width=400, height=300, color=(100, 150, 200)) -> str:
    """Create a temporary valid JPEG image."""
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    img = Image.new("RGB", (width, height), color=color)
    img.save(path, "JPEG")
    return path


def test_valid_image_loading(preprocessor):
    """Verify clean loading of valid image and quality metric computation."""
    path = create_test_image(640, 480)
    try:
        prep = preprocessor.validate_and_load(path)
        assert prep.metrics.width == 640
        assert prep.metrics.height == 480
        assert prep.metrics.channels == 3
        assert prep.bgr_image.shape == (480, 640, 3)
        assert prep.metrics.mean_brightness > 0
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_corrupt_image_detection(preprocessor):
    """Corrupt image must raise ImageValidationError."""
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.write(fd, b"NOT_AN_IMAGE_RANDOM_CORRUPT_BYTES_1234567890")
    os.close(fd)

    try:
        with pytest.raises(ImageValidationError) as exc:
            preprocessor.validate_and_load(path)
        assert "Corrupt or unsupported image" in str(exc.value)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_oversized_pixel_guardrail(preprocessor, monkeypatch):
    """Oversized pixel counts should be rejected to protect M1 memory."""
    path = create_test_image(200, 200)
    monkeypatch.setattr(preprocessor, "MAX_PIXELS", 10_000)

    try:
        with pytest.raises(ImageValidationError) as exc:
            preprocessor.validate_and_load(path)
        assert "exceeds safe M1 limit" in str(exc.value)
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_safe_crop_clamping(preprocessor):
    """Bounding boxes extending outside image boundaries must be clamped safely."""
    # Create 200x200 canvas
    canvas = np.zeros((200, 200, 3), dtype=np.uint8)

    # Box with negative coordinates and coordinates extending past 200
    out_of_bounds_box = [-50, -30, 250, 300]
    cropped, clamped_box = preprocessor.safe_crop(canvas, out_of_bounds_box, padding_ratio=0.0)

    x1, y1, x2, y2 = clamped_box
    assert x1 == 0
    assert y1 == 0
    assert x2 == 200
    assert y2 == 200
    assert cropped.shape == (200, 200, 3)


def test_safe_crop_with_padding(preprocessor):
    """Safe crop should apply proportional padding."""
    canvas = np.zeros((100, 100, 3), dtype=np.uint8)
    box = [40, 40, 60, 60]  # 20x20 box
    # 10% padding = 2px on each side
    cropped, clamped_box = preprocessor.safe_crop(canvas, box, padding_ratio=0.1)

    assert clamped_box == [38, 38, 62, 62]
    assert cropped.shape == (24, 24, 3)


def test_resize_for_inference(preprocessor):
    """Large images should be downscaled while maintaining aspect ratio."""
    canvas = np.zeros((1600, 1200, 3), dtype=np.uint8)  # max dim = 1600
    resized, scale = preprocessor.resize_for_inference(canvas, max_dim=800)

    assert max(resized.shape[:2]) == 800
    assert scale == 0.5
    assert resized.shape == (800, 600, 3)
