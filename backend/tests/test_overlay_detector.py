"""Tests for MultiFormatOverlayDetector candidate generation and enhancement."""

import cv2
import numpy as np
import pytest

from app.pipeline.overlay_extractor import (
    MultiFormatOverlayDetector,
    detect_green_overlay,
    detect_yellow_overlay,
)


@pytest.fixture
def detector():
    return MultiFormatOverlayDetector(upscale_factor=2.0)


def test_green_overlay_detection(detector):
    """Green saturated text in top half should generate green_dws candidate."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    # Put green text pixels in top-left (BGR: [0, 220, 0])
    img[30:100, 30:250] = [0, 220, 0]

    mask = detect_green_overlay(img)
    assert mask is not None
    assert np.count_nonzero(mask) > 100

    candidates = detector.detect_green_candidates(img)
    assert len(candidates) >= 1
    assert candidates[0].format_hint == "green_dws"
    assert candidates[0].bbox_xyxy[0] < 35
    assert candidates[0].bbox_xyxy[2] >= 250


def test_yellow_overlay_detection(detector):
    """Yellow saturated text in top half should generate yellow_dws candidate."""
    img = np.zeros((400, 600, 3), dtype=np.uint8)
    # Put yellow text pixels in top-left (BGR: [0, 220, 220])
    img[20:90, 20:200] = [0, 220, 220]

    mask = detect_yellow_overlay(img)
    assert mask is not None
    assert np.count_nonzero(mask) > 100

    candidates = detector.detect_yellow_candidates(img)
    assert len(candidates) >= 1
    assert candidates[0].format_hint == "yellow_dws"


def test_white_on_dark_detection(detector):
    """Top-left white monospaced text on dark background."""
    img = np.ones((500, 600, 3), dtype=np.uint8) * 120  # light grey background
    # Dark panel in top-left
    img[10:150, 10:280] = [15, 15, 15]
    # Draw white text/lines inside
    cv2.putText(img, "AWB: 987654321", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "WT: 2.5 KG", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    candidates = detector.detect_white_on_dark_candidates(img)
    assert len(candidates) >= 1
    assert any(c.format_hint in ("white_text_scanner", "ekart") for c in candidates)


def test_bottom_panel_detection(detector):
    """Bottom grey/white panel with dark text should generate bottom_panel candidate."""
    img = np.zeros((600, 800, 3), dtype=np.uint8)  # dark background
    # Bottom grey panel
    img[500:580, :] = [200, 200, 200]
    # Dark text inside bottom panel
    cv2.putText(img, "L: 30 W: 20 H: 10 A.W: 1.5 V.W: 1.2", (50, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10, 10, 10), 2)

    candidates = detector.detect_bottom_panel_candidates(img)
    assert len(candidates) >= 1
    assert candidates[0].format_hint == "bottom_panel"


def test_fallback_proposals(detector):
    """Plain image should return fallback region proposals."""
    img = np.ones((400, 600, 3), dtype=np.uint8) * 128
    candidates = detector.get_candidate_regions(img)
    assert len(candidates) >= 1
    hints = [c.format_hint for c in candidates]
    assert "generic_overlay" in hints or "bottom_panel" in hints


def test_multi_scale_enhancement(detector):
    """Crop enhancement should produce upscaled, clahe, and binarized variants."""
    small_crop = np.zeros((60, 120, 3), dtype=np.uint8)
    cv2.putText(small_crop, "TEST", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    variants = detector.enhance_crop(small_crop)
    assert "original" in variants
    assert "upscaled" in variants
    assert "clahe" in variants
    assert "binarized" in variants

    # Upscaled should be ~2x
    assert variants["upscaled"].shape[0] >= 120
    assert variants["upscaled"].shape[1] >= 240
