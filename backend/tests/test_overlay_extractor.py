"""Unit tests for warehouse scanner yellow overlay extraction."""

import numpy as np
import cv2
from app.pipeline.overlay_extractor import (
    clean_ocr_line,
    parse_overlay_fields,
    detect_yellow_overlay,
    build_ocr_result_from_overlay,
)


def test_clean_ocr_line():
    """Verify OCR misrecognition cleanup."""
    assert clean_ocr_line("LENGTH: 60-300 CM") == "LENGTH: 60.300 CM"
    assert clean_ocr_line("HEIGHT : 40-300 CM") == "HEIGHT : 40.300 CM"
    assert clean_ocr_line("WEIGHT: 14060.000 9M") == "WEIGHT: 14060.000 gm"
    assert clean_ocr_line("R VOL.. 9520/.000 CM") == "R VOL.. 95207.000 CM"


def test_parse_overlay_fields():
    """Verify structured parsing of scanner overlay lines."""
    lines = [
        "LOCATION: BANGALORE_HOSKOTE_GW (KARNATAKA)",
        "AWB NO: 19106721621035",
        "LENGTH: 75.300 CM",
        "WIDTH: 49.200 CM",
        "HEIGHT: 44.400 CM",
        "R.VOL.: 134230.000 CM (CUBIC)",
        "WEIGHT: 14060.000 9M",
        "TIME: 7/4/2025 12:43:07 PM",
    ]
    data = parse_overlay_fields(lines)

    assert data.location == "BANGALORE_HOSKOTE_GW (KARNATAKA)"
    assert data.awb_no == "19106721621035"
    assert data.length == 75.3
    assert data.width == 49.2
    assert data.height == 44.4
    assert data.dimension_unit == "cm"
    assert data.volume == 134230.0
    assert data.weight_value == 14060.0
    assert data.weight_unit == "g"
    assert data.timestamp == "7/4/2025 12:43:07 PM"
    assert data.is_populated() is True


def test_detect_yellow_overlay_synthetic():
    """Yellow pixels must be detected, black image must return None."""
    black_img = np.zeros((200, 200, 3), dtype=np.uint8)
    assert detect_yellow_overlay(black_img) is None

    # Image with yellow text/patch (BGR: [0, 255, 255])
    yellow_img = np.zeros((200, 200, 3), dtype=np.uint8)
    yellow_img[20:50, 20:80] = [0, 220, 220]
    mask = detect_yellow_overlay(yellow_img)
    assert mask is not None
    assert np.sum(mask > 0) > 100


def test_build_ocr_result_from_overlay():
    """Verify synthesized OcrResult structure."""
    lines = [
        "AWB NO: 19108610903346",
        "LENGTH: 23.000 CM",
        "WIDTH: 31.500 CM",
        "HEIGHT: 14.300 CM",
        "WEIGHT: 2630.000 GM",
    ]
    data = parse_overlay_fields(lines)
    ocr_res = build_ocr_result_from_overlay(data)

    assert ocr_res.engine.startswith("scanner-overlay")
    assert "AWB No: 19108610903346" in ocr_res.full_text
    assert "23.0" in ocr_res.full_text
    assert "2630.0 g" in ocr_res.full_text
