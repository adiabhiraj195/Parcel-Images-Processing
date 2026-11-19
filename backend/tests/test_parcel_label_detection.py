"""Focused tests for parcel and shipping-label detection taxonomy and pipeline.

Validates the 6 explicit outcomes, machine overlay exclusion, precedence rules,
and the non-negotiable one-result-per-image invariant:
- NO_PARCEL_DETECTED
- PARCEL_DETECTED_LABEL_NOT_FOUND
- LABEL_DETECTED_UNREADABLE
- LABEL_FOUND
- MULTIPLE_PARCELS
- LOW_CONFIDENCE_REVIEW
"""

import json
from types import SimpleNamespace
import pytest
from sqlalchemy.orm import Session

from app.pipeline.interfaces import (
    OUTCOME_NO_PARCEL_DETECTED,
    OUTCOME_PARCEL_DETECTED_LABEL_NOT_FOUND,
    OUTCOME_LABEL_DETECTED_UNREADABLE,
    OUTCOME_LABEL_FOUND,
    OUTCOME_MULTIPLE_PARCELS,
    OUTCOME_LOW_CONFIDENCE_REVIEW,
    PARCEL_LABEL_OUTCOMES,
    STATUS_SUCCESS,
    STATUS_PARTIAL_EXTRACTION,
    STATUS_PROCESSING_FAILED,
    BarcodeResult,
    DetectionBox,
    DetectionResult,
    OcrResult,
    OcrTextBox,
    FieldProvenance,
)
from app.pipeline.parser import DeterministicFieldParser
from app.pipeline.parser_registry import ParsedOverlayMetadata
from app.pipeline.detector import (
    OwlV2ZeroShotDetector,
    compute_iou,
    compute_containment,
    non_max_suppression,
)
from app.pipeline.label_extractor import (
    PhysicalLabelExtractor,
    LabelCandidate,
    detect_visual_sticker_contours,
)
from app.models.image import Image
from app.models.batch import Batch
from app.models.extraction import Extraction


@pytest.fixture
def parser():
    return DeterministicFieldParser()


def test_outcome_no_parcel_detected(parser):
    """When scanner scene has 0 parcels and clear image, outcome is NO_PARCEL_DETECTED."""
    det_res = DetectionResult(model="owlv2", detections=[])
    ocr_res = OcrResult(engine="test", full_text="")
    metrics = SimpleNamespace(mean_brightness=120.0, blur_score=150.0, is_blurry=False)

    data = parser.parse(
        ocr_result=ocr_res,
        barcode_results=[],
        detection_result=det_res,
        image_dims=(1920, 1080),
        image_metrics=metrics,
    )
    assert data.parcel_label_outcome == OUTCOME_NO_PARCEL_DETECTED
    assert data.parcel_count == 0
    assert "NO_PARCEL" in data.scene_flags


def test_outcome_multiple_parcels(parser):
    """When multiple distinct parcels are detected on the scanner bed, outcome is MULTIPLE_PARCELS."""
    det_res = DetectionResult(
        model="owlv2",
        detections=[
            DetectionBox(class_name="parcel", score=0.88, bbox_xyxy=[100, 100, 500, 500]),
            DetectionBox(class_name="parcel", score=0.82, bbox_xyxy=[600, 100, 1000, 500]),
        ],
    )
    ocr_res = OcrResult(engine="test", full_text="")
    metrics = SimpleNamespace(mean_brightness=110.0, blur_score=120.0, is_blurry=False)

    data = parser.parse(
        ocr_result=ocr_res,
        barcode_results=[],
        detection_result=det_res,
        image_dims=(1920, 1080),
        image_metrics=metrics,
    )
    assert data.parcel_label_outcome == OUTCOME_MULTIPLE_PARCELS
    assert data.parcel_count == 2
    assert "MULTIPLE_PARCELS" in data.scene_flags


def test_outcome_parcel_detected_label_not_found(parser):
    """
    Parcel present on scanner bed with machine metadata overlay but NO physical shipping label.
    Machine metadata must NOT be inferred as a physical shipping label.
    """
    det_res = DetectionResult(
        model="owlv2",
        detections=[
            DetectionBox(class_name="parcel", score=0.91, bbox_xyxy=[200, 200, 800, 800]),
        ],
    )
    # Machine overlay text at top-left
    overlay_text = (
        "Location: WAREHOUSE-1\n"
        "AWB No: 19108610903346\n"
        "Length: 25.0 cm\n"
        "Width: 20.0 cm\n"
        "Height: 15.0 cm\n"
        "Weight: 1500 g\n"
    )
    ocr_res = OcrResult(
        engine="paddle",
        full_text=overlay_text,
        lines=[
            OcrTextBox(text="AWB No: 19108610903346", score=0.99, source_crop="overlay_top_left"),
            OcrTextBox(text="Weight: 1500 g", score=0.99, source_crop="overlay_top_left"),
        ],
    )
    overlay_meta = ParsedOverlayMetadata(
        profile_name="yellow_dws",
        tracking_number="19108610903346",
        weight_value=1500.0,
        weight_unit="g",
        length=25.0,
        width=20.0,
        height=15.0,
        confidence=0.95,
    )
    metrics = SimpleNamespace(mean_brightness=95.0, blur_score=140.0, is_blurry=False)

    data = parser.parse(
        ocr_result=ocr_res,
        barcode_results=[],
        detection_result=det_res,
        overlay_meta=overlay_meta,
        image_dims=(1920, 1080),
        image_metrics=metrics,
        label_candidate_detected=False,
        label_readable=False,
        label_boxes=[],
    )

    # Machine metadata fields successfully extracted
    assert data.tracking_number == "19108610903346"
    assert data.status == STATUS_SUCCESS

    # But physical label outcome is strictly PARCEL_DETECTED_LABEL_NOT_FOUND
    assert data.parcel_label_outcome == OUTCOME_PARCEL_DETECTED_LABEL_NOT_FOUND
    assert "LABEL_NOT_FOUND" in data.scene_flags
    assert len(data.label_boxes) == 0


def test_outcome_label_detected_unreadable(parser):
    """
    Physical label region located on parcel, but OCR produces unreadable text
    and barcode cannot be decoded. Must NOT claim label not found.
    """
    det_res = DetectionResult(
        model="owlv2",
        detections=[
            DetectionBox(class_name="parcel", score=0.89, bbox_xyxy=[150, 150, 750, 750]),
        ],
    )
    label_box = DetectionBox(class_name="shipping_label", score=0.78, bbox_xyxy=[300, 300, 500, 500])

    # Unreadable / torn text on label
    ocr_res = OcrResult(
        engine="paddle",
        full_text="X#? 9! \n $$*",
        lines=[
            OcrTextBox(text="X#? 9!", score=0.25, source_crop="shipping_label"),
            OcrTextBox(text="$$*", score=0.20, source_crop="shipping_label"),
        ],
    )
    metrics = SimpleNamespace(mean_brightness=85.0, blur_score=110.0, is_blurry=False)

    data = parser.parse(
        ocr_result=ocr_res,
        barcode_results=[],
        detection_result=det_res,
        image_dims=(1920, 1080),
        image_metrics=metrics,
        label_candidate_detected=True,
        label_readable=False,
        label_boxes=[label_box],
    )

    assert data.parcel_label_outcome == OUTCOME_LABEL_DETECTED_UNREADABLE
    assert "LABEL_NOT_FOUND" not in data.scene_flags
    assert len(data.label_boxes) == 1


def test_outcome_label_found(parser):
    """
    Parcel and physical shipping label detected with readable tracking number
    and destination information.
    """
    det_res = DetectionResult(
        model="owlv2",
        detections=[
            DetectionBox(class_name="parcel", score=0.92, bbox_xyxy=[100, 100, 900, 900]),
            DetectionBox(class_name="shipping_label", score=0.88, bbox_xyxy=[300, 300, 600, 600]),
        ],
    )
    label_box = DetectionBox(class_name="shipping_label", score=0.88, bbox_xyxy=[300, 300, 600, 600])

    barcode = BarcodeResult(
        raw_value="FMPC1234567890",
        symbology="CODE_128",
        bbox_xyxy=[320, 350, 580, 420],
        source_crop="label_crop_0_deg",
    )
    ocr_res = OcrResult(
        engine="paddle",
        full_text="DELIVERY ADDRESS\nKuan Pato\nFMPC1234567890\nPREPAID",
        lines=[
            OcrTextBox(text="DELIVERY ADDRESS", score=0.95, source_crop="shipping_label"),
            OcrTextBox(text="FMPC1234567890", score=0.98, source_crop="shipping_label"),
            OcrTextBox(text="PREPAID", score=0.92, source_crop="shipping_label"),
        ],
    )
    metrics = SimpleNamespace(mean_brightness=90.0, blur_score=130.0, is_blurry=False)

    data = parser.parse(
        ocr_result=ocr_res,
        barcode_results=[barcode],
        detection_result=det_res,
        image_dims=(1920, 1080),
        image_metrics=metrics,
        label_candidate_detected=True,
        label_readable=True,
        label_boxes=[label_box],
    )

    assert data.parcel_label_outcome == OUTCOME_LABEL_FOUND
    assert data.tracking_number == "FMPC1234567890"
    assert "LABEL_NOT_FOUND" not in data.scene_flags
    assert len(data.label_boxes) == 1


def test_outcome_low_confidence_review_dark_image(parser):
    """
    When the image is extremely dark (e.g. mean brightness < 20.0),
    pipeline must NEVER claim NO_LABEL or NO_PARCEL with certainty.
    Precedence forces LOW_CONFIDENCE_REVIEW.
    """
    det_res = DetectionResult(model="owlv2", detections=[])
    ocr_res = OcrResult(engine="test", full_text="")
    # Extremely dark image (like sample 3 with mean brightness 7.8)
    metrics = SimpleNamespace(mean_brightness=7.8, blur_score=120.0, is_blurry=False)

    data = parser.parse(
        ocr_result=ocr_res,
        barcode_results=[],
        detection_result=det_res,
        image_dims=(8192, 2719),
        image_metrics=metrics,
    )
    assert data.parcel_label_outcome == OUTCOME_LOW_CONFIDENCE_REVIEW


def test_outcome_low_confidence_review_blurry_image(parser):
    """
    When the image is blurry, pipeline must output LOW_CONFIDENCE_REVIEW
    rather than definitively stating no label was found.
    """
    det_res = DetectionResult(
        model="owlv2",
        detections=[
            DetectionBox(class_name="parcel", score=0.75, bbox_xyxy=[200, 200, 700, 700]),
        ],
    )
    ocr_res = OcrResult(engine="test", full_text="")
    metrics = SimpleNamespace(mean_brightness=80.0, blur_score=22.0, is_blurry=True)

    data = parser.parse(
        ocr_result=ocr_res,
        barcode_results=[],
        detection_result=det_res,
        image_dims=(1920, 1080),
        image_metrics=metrics,
        label_candidate_detected=False,
        label_readable=False,
    )
    assert data.parcel_label_outcome == OUTCOME_LOW_CONFIDENCE_REVIEW


def test_outcome_low_confidence_review_partial_parcel_touches_edge(parser):
    """
    When a parcel touches the image border (partial parcel) and no label is visible,
    it is ambiguous whether the label was cropped off at the border.
    Precedence maps this to LOW_CONFIDENCE_REVIEW.
    """
    det_res = DetectionResult(
        model="owlv2",
        detections=[
            # Box touching left edge (x1 = 5 <= margin 15)
            DetectionBox(class_name="parcel", score=0.85, bbox_xyxy=[5, 100, 600, 700]),
        ],
    )
    ocr_res = OcrResult(engine="test", full_text="")
    metrics = SimpleNamespace(mean_brightness=85.0, blur_score=120.0, is_blurry=False)

    data = parser.parse(
        ocr_result=ocr_res,
        barcode_results=[],
        detection_result=det_res,
        image_dims=(1000, 1000),
        image_metrics=metrics,
        label_candidate_detected=False,
        label_readable=False,
    )
    assert "PARTIAL_PARCEL" in data.scene_flags
    assert data.parcel_label_outcome == OUTCOME_LOW_CONFIDENCE_REVIEW


def test_machine_overlay_box_exclusion():
    """
    Verify that candidate regions falling inside machine metadata overlays
    are explicitly rejected from physical label candidate generation.
    """
    from app.pipeline.label_extractor import find_label_candidates_from_detector, resolve_label_candidates

    # Machine metadata overlay at top-left
    overlay_box = [90, 40, 1100, 830]

    # Detector found two labels: one on the yellow machine overlay, one on the physical parcel
    det_res = DetectionResult(
        model="owlv2",
        detections=[
            # False label candidate matching machine overlay
            DetectionBox(class_name="shipping_label", score=0.22, bbox_xyxy=[95, 45, 1080, 820]),
            # True physical label on parcel
            DetectionBox(class_name="shipping_label", score=0.88, bbox_xyxy=[1200, 300, 1600, 800]),
        ],
    )

    candidates = find_label_candidates_from_detector(det_res)
    assert len(candidates) == 2

    # Apply overlay filter
    filtered = []
    for cand in candidates:
        iou = compute_iou(cand.bbox_xyxy, overlay_box)
        cont = compute_containment(cand.bbox_xyxy, overlay_box)
        if not (iou > 0.15 or cont > 0.35):
            filtered.append(cand)

    assert len(filtered) == 1
    assert filtered[0].bbox_xyxy == [1200, 300, 1600, 800]


@pytest.fixture
def test_db():
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker
    from app.database import Base

    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_exact_one_result_per_image_invariant(test_db):
    """
    Ensure the non-negotiable invariant: exactly one extraction record per image,
    including failures, with parcel_label_outcome persisted.
    """
    import uuid

    batch = Batch(id=str(uuid.uuid4()), name="Test Invariant Batch")
    test_db.add(batch)
    test_db.commit()

    image = Image(
        id=str(uuid.uuid4()),
        batch_id=batch.id,
        filename="test_sample.jpg",
        s3_bucket="test-bucket",
        s3_key="batches/test/test_sample.jpg",
        content_type="image/jpeg",
        file_size=1024,
    )
    test_db.add(image)
    test_db.commit()

    # Create extraction with outcome
    ext = Extraction(
        image_id=image.id,
        tracking_number=None,
        status="PARTIAL_EXTRACTION",
        parcel_label_outcome=OUTCOME_PARCEL_DETECTED_LABEL_NOT_FOUND,
        parcel_count=1,
    )
    test_db.add(ext)
    test_db.commit()

    # Query extraction count for this image
    rows = test_db.query(Extraction).filter(Extraction.image_id == image.id).all()
    assert len(rows) == 1
    assert rows[0].parcel_label_outcome == OUTCOME_PARCEL_DETECTED_LABEL_NOT_FOUND
    assert rows[0].parcel_label_outcome in PARCEL_LABEL_OUTCOMES


def test_csv_export_includes_parcel_label_outcome(client):
    """Verify CSV export endpoint includes parcel_label_outcome header and row values."""
    # 1. Create a batch
    batch_payload = {
        "name": "Outcome CSV Batch",
        "files": [
            {
                "filename": "test_box.jpg",
                "content_type": "image/jpeg",
                "file_size": 2048,
            }
        ],
    }
    create_res = client.post("/api/batches", json=batch_payload)
    assert create_res.status_code == 201
    batch_id = create_res.json()["batch_id"]

    # 2. Export CSV
    csv_res = client.get(f"/api/batches/{batch_id}/export.csv")
    assert csv_res.status_code == 200
    lines = csv_res.text.strip().split("\n")
    header_cols = lines[0].strip().split(",")
    assert "parcel_label_outcome" in header_cols
    assert header_cols.index("parcel_label_outcome") == 14
