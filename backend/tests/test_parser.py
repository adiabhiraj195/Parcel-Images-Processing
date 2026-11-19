"""Unit tests for deterministic parsing, conflict handling, status redesign, and scene flags."""

import pytest
from app.pipeline.interfaces import (
    OcrResult,
    OcrTextBox,
    BarcodeResult,
    DetectionResult,
    DetectionBox,
    STATUS_SUCCESS,
    STATUS_PARTIAL_EXTRACTION,
    STATUS_SOURCE_CONFLICT,
    STATUS_LOW_CONFIDENCE,
    SCENE_FLAG_NO_PARCEL,
    SCENE_FLAG_MULTIPLE_PARCELS,
    SCENE_FLAG_LABEL_NOT_FOUND,
)
from app.pipeline.parser import DeterministicFieldParser


@pytest.fixture
def parser():
    return DeterministicFieldParser()


def make_detection(parcels=1, labels=1) -> DetectionResult:
    boxes = []
    for _ in range(parcels):
        boxes.append(DetectionBox(class_name="parcel", score=0.9, bbox_xyxy=[0, 0, 500, 500]))
    for _ in range(labels):
        boxes.append(DetectionBox(class_name="shipping_label", score=0.9, bbox_xyxy=[50, 50, 200, 200]))
    return DetectionResult(model="test", detections=boxes)


def test_ups_tracking_parsing(parser):
    """UPS 1Z format should be parsed with dynamic confidence."""
    text = "SHIP TO DOCK\nTRACKING: 1Z999AA10123456784\nWEIGHT: 5 KG"
    ocr = OcrResult(engine="test", lines=[OcrTextBox(text="TRACKING: 1Z999AA10123456784", score=0.95)], full_text=text)
    res = parser.parse(ocr, [], make_detection(1, 1))

    assert res.tracking_number == "1Z999AA10123456784"
    assert res.weight_value == 5.0
    assert res.weight_unit == "kg"
    assert res.status == STATUS_PARTIAL_EXTRACTION  # Missing dimensions
    assert res.provenance["tracking_number"].source == "label_ocr"


def test_barcode_and_ocr_agreement(parser):
    """When both barcode and OCR read the same tracking number, dynamic confidence is computed."""
    text = "AWB: 1Z8888888888888888\nWT: 2.5 KG\nDIMS: 20x15x10 cm"
    ocr = OcrResult(engine="test", lines=[OcrTextBox(text="AWB: 1Z8888888888888888", score=0.96)], full_text=text)
    barcodes = [BarcodeResult(raw_value="1Z8888888888888888", symbology="Code128")]
    res = parser.parse(ocr, barcodes, make_detection(1, 1))

    assert res.tracking_number == "1Z8888888888888888"
    assert res.provenance["tracking_number"].conflict_status == "AGREED"
    assert res.field_confidence["tracking_number"] >= 0.95
    assert res.status == STATUS_SUCCESS


def test_barcode_and_ocr_conflict(parser):
    """When barcode and OCR conflict, record conflict and flag SOURCE_CONFLICT."""
    text = "AWB: 1Z1111111111111111"
    ocr = OcrResult(engine="test", lines=[OcrTextBox(text=text, score=0.92)], full_text=text)
    barcodes = [BarcodeResult(raw_value="1Z9999999999999999", symbology="Code128")]
    res = parser.parse(ocr, barcodes, make_detection(1, 1))

    assert res.provenance["tracking_number"].conflict_status == "CONFLICT"
    assert res.status == STATUS_SOURCE_CONFLICT
    assert res.field_confidence["tracking_number"] <= 0.5


def test_status_redesign_scene_flag_no_parcel_preserves_metadata(parser):
    """Even if scene flag is NO_PARCEL, extracted metadata is preserved in extraction fields!"""
    text = "AWB No: 19106721621035\nLength: 75.3 cm\nWidth: 49.2 cm\nHeight: 44.4 cm\nWeight: 14060 gm"
    ocr = OcrResult(engine="test", lines=[OcrTextBox(text=text, score=0.95)], full_text=text)
    res = parser.parse(ocr, [], make_detection(parcels=0, labels=0))

    # Scene flag records NO_PARCEL
    assert SCENE_FLAG_NO_PARCEL in res.scene_flags
    # Extraction status reflects field extraction (SUCCESS), NOT discarding data!
    assert res.status == STATUS_SUCCESS
    assert res.tracking_number == "19106721621035"
    assert res.weight_value == 14060.0
    assert res.length == 75.3


def test_status_redesign_multiple_parcels(parser):
    """Multiple parcel detections sets MULTIPLE_PARCELS scene flag while retaining extraction."""
    ocr = OcrResult(engine="test", lines=[], full_text="AWB: 1Z9999999999999999")
    res = parser.parse(ocr, [], make_detection(parcels=2, labels=1))

    assert SCENE_FLAG_MULTIPLE_PARCELS in res.scene_flags
    assert res.tracking_number == "1Z9999999999999999"
    assert res.status == STATUS_PARTIAL_EXTRACTION


def test_status_redesign_label_not_found(parser):
    """Zero labels detected sets LABEL_NOT_FOUND scene flag."""
    ocr = OcrResult(engine="test", lines=[], full_text="")
    res = parser.parse(ocr, [], make_detection(parcels=1, labels=0))

    assert SCENE_FLAG_LABEL_NOT_FOUND in res.scene_flags


def test_scanner_overlay_intake_parsing(parser):
    """Warehouse scanner intake overlay text extracts all logistics fields deterministically."""
    scanner_text = (
        "Location: Bangalore_Hoskote_GW (Karnataka)\n"
        "AWB No: 19106721621035\n"
        "Length: 75.300 cm\n"
        "Width: 49.200 cm\n"
        "Height: 44.400 cm\n"
        "R.Vol.: 134230.000 cm (cubic)\n"
        "Weight: 14060.000 gm\n"
        "Time: 7/4/2025 12:43:07 PM"
    )
    ocr = OcrResult(engine="scanner-overlay-segmentation-v1", lines=[], full_text=scanner_text)
    res = parser.parse(ocr, [], make_detection(parcels=1, labels=0))

    assert res.tracking_number == "19106721621035"
    assert res.weight_value == 14060.0
    assert res.weight_unit == "g"
    assert res.length == 75.3
    assert res.width == 49.2
    assert res.height == 44.4
    assert res.dimension_unit == "cm"
    assert res.status == STATUS_SUCCESS
    assert res.parcel_count == 1
    assert res.provenance["tracking_number"].source == "scanner_metadata"
