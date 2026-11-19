"""
Unit tests for Milestone 3: Physical Shipping Label Extraction.
Verifies multi-cue localization (text clustering, barcode anchors, OWLv2 evidence),
crop rotations/enhancements, independent barcode decoding, and full text line return.
"""

import numpy as np
import pytest

from app.pipeline.interfaces import (
    BarcodeResult,
    DetectionBox,
    DetectionResult,
    OcrResult,
    OcrTextBox,
)
from app.pipeline.spatial_clustering import SpatialCluster
from app.pipeline.label_extractor import (
    PhysicalLabelExtractor,
    LabelCandidate,
    find_label_candidates_from_barcodes,
    find_label_candidates_from_text,
    find_label_candidates_from_detector,
    resolve_label_candidates,
    generate_label_crops,
    detect_visual_barcode_stripes,
)
from app.pipeline.ocr import MockOcrEngine
from app.pipeline.barcode import MockBarcodeDecoder


def _make_box(x1: int, y1: int, x2: int, y2: int, text: str, score: float = 0.95) -> OcrTextBox:
    quad = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return OcrTextBox(text=text, score=score, box=quad)


def test_find_label_candidates_from_barcodes():
    """Barcode bounding box should expand into a shipping label candidate region."""
    barcodes = [
        BarcodeResult(
            raw_value="PROC0000316077",
            symbology="Code128",
            bbox_xyxy=[200, 400, 600, 480],
        )
    ]
    cands = find_label_candidates_from_barcodes(barcodes, img_w=1000, img_h=1000)
    assert len(cands) == 1
    c = cands[0]
    assert c.source == "barcode_anchor"
    assert c.score >= 0.90
    # Candidate should extend above and below the barcode
    assert c.bbox_xyxy[1] < 400  # y1 extends above
    assert c.bbox_xyxy[3] > 480  # y2 extends below
    assert c.bbox_xyxy[0] < 200  # x1 extends left
    assert c.bbox_xyxy[2] > 600  # x2 extends right


def test_find_label_candidates_from_text_clusters():
    """Text clusters with >= 3 lines in parcel body should form label candidates."""
    lines = [
        [_make_box(100, 200, 300, 230, "SHIP TO: JOHN DOE")],
        [_make_box(100, 240, 300, 270, "123 MAIN STREET")],
        [_make_box(100, 280, 250, 310, "TRACKING: 9876543210")],
        [_make_box(100, 320, 200, 350, "WT: 2.5 KG")],
    ]
    cluster = SpatialCluster(
        bbox_xyxy=[100, 200, 300, 350],
        region_label="label_area",
        lines=lines,
    )
    cands = find_label_candidates_from_text([cluster], img_w=1000, img_h=1000)
    assert len(cands) == 1
    c = cands[0]
    assert c.source == "text_cluster"
    assert c.score >= 0.80
    assert c.details["text_lines_count"] == 4


def test_resolve_label_candidates_multi_evidence():
    """Overlapping barcode and text cluster should fuse into multi_evidence candidate with boosted score."""
    b_cand = LabelCandidate(
        bbox_xyxy=[100, 200, 500, 600],
        source="barcode_anchor",
        score=0.90,
    )
    t_cand = LabelCandidate(
        bbox_xyxy=[120, 220, 480, 580],
        source="text_cluster",
        score=0.85,
    )
    det_cand = LabelCandidate(
        bbox_xyxy=[110, 210, 490, 590],
        source="owlv2_detector",
        score=0.48,
    )

    fused = resolve_label_candidates([b_cand, t_cand, det_cand], img_w=1000, img_h=1000)
    assert len(fused) == 1
    assert fused[0].source == "multi_evidence"
    assert fused[0].score >= 0.95


def test_generate_label_crops_rotations_and_enhancements():
    """Label crop should generate original, 90°, 180°, 270°, and CLAHE variants."""
    crop = np.zeros((200, 300, 3), dtype=np.uint8)
    crop[50:150, 50:250] = 180

    variants = generate_label_crops(crop, try_rotations=True)
    assert "0_deg" in variants
    assert "90_deg" in variants
    assert "180_deg" in variants
    assert "270_deg" in variants
    assert "clahe" in variants

    # Check 90° rotation swaps dimensions
    assert variants["90_deg"].shape == (300, 200, 3)
    assert variants["180_deg"].shape == (200, 300, 3)
    assert variants["270_deg"].shape == (300, 200, 3)


def test_physical_label_extractor_without_owlv2():
    """Pipeline must successfully extract label and barcodes even when OWLv2 detects nothing."""
    mock_ocr = MockOcrEngine(
        predefined_lines=[
            OcrTextBox(text="SHIP TO: RECIPIENT", score=0.95, box=[[10, 10], [100, 10], [100, 30], [10, 30]]),
            OcrTextBox(text="AWB: 1Z9999999999999999", score=0.98, box=[[10, 40], [200, 40], [200, 60], [10, 60]]),
            OcrTextBox(text="WEIGHT: 1.5 KG", score=0.92, box=[[10, 70], [120, 70], [120, 90], [10, 90]]),
        ]
    )
    mock_barcode = MockBarcodeDecoder(
        predefined_results=[
            BarcodeResult(raw_value="1Z9999999999999999", symbology="Code128", bbox_xyxy=[150, 250, 450, 320]),
        ]
    )
    extractor = PhysicalLabelExtractor(ocr_engine=mock_ocr, barcode_decoder=mock_barcode)

    dummy_image = np.full((600, 800, 3), 100, dtype=np.uint8)
    # OWLv2 detects empty list (no labels found by detector)
    empty_det = DetectionResult(model="owlv2", detections=[])

    ocr_res, barcodes, bbox, crop = extractor.extract_label_and_barcodes(
        bgr_image=dummy_image,
        det_result=empty_det,
    )

    # Must find label from barcode anchor despite OWLv2 finding zero labels
    assert bbox is not None
    assert crop is not None
    assert len(barcodes) >= 1
    assert any(b.raw_value == "1Z9999999999999999" for b in barcodes)
    assert ocr_res is not None
    assert len(ocr_res.lines) >= 1


def test_independent_barcode_detection_full_and_crop():
    """Barcodes should be decoded independently on both full image and crop variants."""
    class DynamicBarcodeDecoder(MockBarcodeDecoder):
        def decode(self, img_input):
            # If large image (full image), return full barcode
            arr = np.array(img_input)
            if arr.shape[0] > 400:
                return [BarcodeResult(raw_value="FULL_IMAGE_BC", symbology="Code128", bbox_xyxy=[100, 100, 300, 150])]
            else:
                return [BarcodeResult(raw_value="CROP_ROTATED_BC", symbology="Code39")]

    extractor = PhysicalLabelExtractor(
        ocr_engine=MockOcrEngine(),
        barcode_decoder=DynamicBarcodeDecoder(),
    )
    dummy_image = np.full((600, 800, 3), 120, dtype=np.uint8)
    ocr_res, barcodes, bbox, crop = extractor.extract_label_and_barcodes(dummy_image)

    values = [b.raw_value for b in barcodes]
    sources = [b.source_crop for b in barcodes]
    assert "FULL_IMAGE_BC" in values
    assert "CROP_ROTATED_BC" in values
    assert "full_image" in sources
    assert any("label_crop" in s for s in sources)
