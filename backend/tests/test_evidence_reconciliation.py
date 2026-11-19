"""Unit tests for evidence reconciliation, field provenance, and source conflict detection."""

import pytest

from app.pipeline.interfaces import (
    STATUS_SOURCE_CONFLICT,
    BarcodeResult,
    DetectionBox,
    DetectionResult,
    OcrResult,
    OcrTextBox,
)
from app.pipeline.parser import DeterministicFieldParser
from app.pipeline.parser_registry import parser_registry


@pytest.fixture
def parser():
    return DeterministicFieldParser()


def make_dummy_detection():
    return DetectionResult(
        model="test",
        detections=[DetectionBox(class_name="parcel", score=0.9, bbox_xyxy=[0, 0, 100, 100])],
    )


def test_field_provenance_structure(parser):
    """Every extracted field must record complete provenance metadata."""
    text = (
        "AWB No: 19106721621035\n"
        "Length: 50.0 cm\n"
        "Width: 40.0 cm\n"
        "Height: 30.0 cm\n"
        "Weight: 2.5 kg"
    )
    ocr = OcrResult(engine="test", lines=[OcrTextBox(text=text, score=0.94)], full_text=text)
    res = parser.parse(ocr, [], make_dummy_detection())

    for field_name in ("tracking_number", "weight", "dimensions"):
        prov = res.provenance.get(field_name)
        assert prov is not None
        assert prov.value is not None
        assert prov.source != ""
        assert prov.raw_text != ""
        assert prov.ocr_confidence > 0.0
        assert prov.parser_rule != ""


def test_prefer_measured_scanner_data_over_label(parser):
    """Measured DWS scanner metadata must be preferred over label OCR declarations."""
    overlay_text = (
        "AWB No: 19106721621035\n"
        "Length: 75.3 cm\n"
        "Width: 49.2 cm\n"
        "Height: 44.4 cm\n"
        "Weight: 14060 gm"
    )
    overlay_meta = parser_registry.parse(overlay_text, format_hint="yellow_dws")

    # Paper label OCR declares different dimensions
    label_text = "Dims: 70 x 45 x 40 cm\nWeight: 13.5 kg"
    ocr = OcrResult(engine="test", lines=[], full_text=label_text)

    res = parser.parse(ocr, [], make_dummy_detection(), overlay_meta=overlay_meta)

    # Must prefer measured scanner overlay data
    assert res.length == 75.3
    assert res.width == 49.2
    assert res.height == 44.4
    assert res.provenance["dimensions"].source == "scanner_metadata"
    assert res.weight_value == 14060.0
    assert res.provenance["weight"].source == "scanner_metadata"


def test_independent_awb_conflict_flagged_for_review(parser):
    """When scanner overlay AWB conflicts with printed barcode AWB, flag SOURCE_CONFLICT."""
    overlay_text = "AWB No: 19106721621035"
    overlay_meta = parser_registry.parse(overlay_text, format_hint="green_dws")

    barcodes = [BarcodeResult(raw_value="TRACK9999999999", symbology="Code128")]
    ocr = OcrResult(engine="test", lines=[], full_text=overlay_text)

    res = parser.parse(ocr, barcodes, make_dummy_detection(), overlay_meta=overlay_meta)

    assert res.status == STATUS_SOURCE_CONFLICT
    assert res.provenance["tracking_number"].conflict_status == "CONFLICT"
    # Confidence must drop significantly upon conflict
    assert res.field_confidence["tracking_number"] < 0.50


def test_actual_and_volumetric_weight_separated(parser):
    """Actual weight (A.W) and volumetric weight (V.W) must not be collapsed."""
    text = (
        "L: 40 W: 30 H: 20 cm\n"
        "A.W: 3.50 kg\n"
        "V.W: 4.80 kg\n"
        "AWB: 987654321012"
    )
    ocr = OcrResult(engine="test", lines=[OcrTextBox(text=text, score=0.91)], full_text=text)
    res = parser.parse(ocr, [], make_dummy_detection())

    assert res.actual_weight == 3.50
    assert res.actual_weight_unit == "kg"
    assert res.volumetric_weight == 4.80
    assert res.volumetric_weight_unit == "kg"
    assert res.weight_value == 3.50  # Primary weight is actual weight
    assert "volumetric_weight" in res.provenance


def test_dynamic_confidence_calculation(parser):
    """Confidence scores must vary dynamically based on OCR score and agreement."""
    # Low-scoring OCR line
    low_ocr = OcrResult(
        engine="test",
        lines=[OcrTextBox(text="AWB: 1Z999AA10123456784", score=0.45)],
        full_text="AWB: 1Z999AA10123456784",
    )
    res_low = parser.parse(low_ocr, [], make_dummy_detection())

    # High-scoring OCR line with matching barcode
    high_ocr = OcrResult(
        engine="test",
        lines=[OcrTextBox(text="AWB: 1Z999AA10123456784", score=0.98)],
        full_text="AWB: 1Z999AA10123456784",
    )
    barcodes = [BarcodeResult(raw_value="1Z999AA10123456784", symbology="Code128")]
    res_high = parser.parse(high_ocr, barcodes, make_dummy_detection())

    conf_low = res_low.field_confidence["tracking_number"]
    conf_high = res_high.field_confidence["tracking_number"]

    assert conf_high > conf_low
    assert conf_high != 0.99 or conf_low != 0.85  # Dynamic, not fixed constants
