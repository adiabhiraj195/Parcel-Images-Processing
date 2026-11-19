"""
Unit tests for Milestone 4: PaddleOCR-VL 1.6 Advanced Model Adapter.
Verifies optional/configurable initialization, model health reporting,
second-stage interpretation, and strict evidence corroboration against
deterministic OCR text and barcodes.
"""

import numpy as np
import pytest

from app.pipeline.interfaces import BarcodeResult, OcrTextBox
from app.pipeline.paddleocr_vl_adapter import (
    PaddleOcrVlAdapter,
    MockPaddleOcrVlAdapter,
    corroborate_proposed_tracking,
    corroborate_proposed_weight,
    corroborate_proposed_dimensions,
)
from app.api.health import get_models_health


def _make_box(text: str) -> OcrTextBox:
    return OcrTextBox(text=text, score=0.95, box=[[0, 0], [10, 0], [10, 10], [0, 10]])


def test_paddleocr_vl_disabled_health():
    """When disabled, adapter should report readiness='disabled' without loading weights."""
    adapter = PaddleOcrVlAdapter(enabled=False, model_version="1.6", device="cpu")
    health = adapter.get_health()
    assert health["engine"] == "paddleocr_vl"
    assert health["model_version"] == "1.6"
    assert health["readiness"] == "disabled"
    assert "Apple Silicon M1" in health["runtime"]


def test_paddleocr_vl_mock_health():
    """Mock adapter reports readiness='ready'."""
    adapter = MockPaddleOcrVlAdapter()
    health = adapter.get_health()
    assert health["engine"] == "paddleocr_vl"
    assert health["readiness"] == "ready"


def test_models_health_endpoint_includes_paddleocr_vl():
    """Health check endpoint should report paddleocr_vl status."""
    res = get_models_health()
    assert hasattr(res, "paddleocr_vl")
    assert res.paddleocr_vl is not None
    assert res.paddleocr_vl.engine == "paddleocr_vl"
    assert res.paddleocr_vl.model_version in ("1.6", "1.6-mock")


def test_corroborate_tracking_with_barcode():
    """VLM candidate matching a decoded barcode must be accepted."""
    barcodes = [BarcodeResult(raw_value="1Z9999999999999999", symbology="Code128")]
    ocr_lines = [_make_box("SOME OTHER TEXT")]

    ok, src, reason = corroborate_proposed_tracking("1Z9999999999999999", ocr_lines, barcodes)
    assert ok is True
    assert "barcode" in src
    assert reason is None


def test_corroborate_tracking_with_ocr_text():
    """VLM candidate present in deterministic OCR line must be accepted."""
    barcodes = []
    ocr_lines = [
        _make_box("TRACKING NO: PROC0000316077"),
        _make_box("WEIGHT: 2.5 KG"),
    ]

    ok, src, reason = corroborate_proposed_tracking("PROC0000316077", ocr_lines, barcodes)
    assert ok is True
    assert "ocr_line" in src
    assert reason is None


def test_reject_uncorroborated_tracking_hallucination():
    """VLM candidate NOT found in OCR text or barcodes must be strictly rejected."""
    barcodes = [BarcodeResult(raw_value="ACTUAL_BARCODE_123", symbology="Code128")]
    ocr_lines = [_make_box("ACTUAL_BARCODE_123 WT: 1.0KG")]

    # Hallucinated candidate
    ok, src, reason = corroborate_proposed_tracking("HALLUCINATED999", ocr_lines, barcodes)
    assert ok is False
    assert src is None
    assert "No corroborating evidence" in reason


def test_reject_alpha_only_tracking_candidate():
    """VLM proposing alphabetic-only tracking (like INVENTIMENT) must be rejected."""
    barcodes = []
    ocr_lines = [_make_box("INVENTIMENT")]

    ok, src, reason = corroborate_proposed_tracking("INVENTIMENT", ocr_lines, barcodes)
    assert ok is False
    assert "Alpha-only" in reason


def test_corroborate_and_reject_weight():
    """Weight is corroborated only if its numeric value appears in deterministic OCR lines."""
    ocr_lines = [_make_box("A.W: 3.25 KG  V.W: 2.10 KG")]

    # True weight present in OCR
    ok_true, src, _ = corroborate_proposed_weight(3.25, ocr_lines)
    assert ok_true is True
    assert "ocr_line" in src

    # Hallucinated weight not in OCR
    ok_false, _, reason = corroborate_proposed_weight(9.99, ocr_lines)
    assert ok_false is False
    assert "not found" in reason


def test_corroborate_and_reject_dimensions():
    """Dimensions are corroborated only if numbers appear in deterministic OCR lines."""
    ocr_lines = [_make_box("DIMS: 30x20x15 CM")]

    # Correct dimensions
    ok_true, src, _ = corroborate_proposed_dimensions(30.0, 20.0, 15.0, ocr_lines)
    assert ok_true is True

    # Hallucinated dimensions
    ok_false, _, reason = corroborate_proposed_dimensions(99.0, 88.0, 77.0, ocr_lines)
    assert ok_false is False
    assert "not corroborated" in reason


def test_second_stage_interpretation_workflow():
    """Mock VLM interpreter should parse crop and corroborate against provided OCR lines."""
    mock_adapter = MockPaddleOcrVlAdapter(
        mock_raw_output="Tracking: 1Z9999999999999999 Weight: 2.5 KG Dims: 30x20x15",
        enabled=True,
    )
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    ocr_lines = [
        _make_box("1Z9999999999999999"),
        _make_box("WT: 2.5 KG"),
        _make_box("DIMS: 30x20x15 CM"),
    ]
    barcodes = [BarcodeResult(raw_value="1Z9999999999999999", symbology="Code128")]

    result = mock_adapter.interpret_region(crop, full_ocr_lines=ocr_lines, barcodes=barcodes)
    assert result is not None
    assert result.tracking_number == "1Z9999999999999999"
    assert result.weight_value == 2.5
    assert result.dimensions == (30.0, 20.0, 15.0)
    assert "tracking_number" in result.accepted_fields
    assert "weight_value" in result.accepted_fields
    assert "dimensions" in result.accepted_fields
    assert len(result.rejected_fields) == 0
