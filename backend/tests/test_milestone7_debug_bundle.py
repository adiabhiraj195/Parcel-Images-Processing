"""
Unit tests for Milestone 7: Improve Extraction Logging & Debug Bundling.
Verifies that every processed image creates a comprehensive debug bundle with:
- Original image metrics and coordinates
- All candidate regions
- All OCR detections and quad bounding boxes
- Text from each region
- Enhanced image variants
- Label crops
- Barcode decoder attempts
- Profile scores and matched anchors
- Parsed values and validation failures
- Final field provenance and confidence
"""

import os
import json
import numpy as np
import pytest

from app.models.image import Image as ImageModel
from app.pipeline.interfaces import (
    BarcodeResult,
    DetectionBox,
    DetectionResult,
    ExtractedLogisticsData,
    FieldProvenance,
    OcrResult,
    OcrTextBox,
)
from app.pipeline.preprocessor import ImageQualityMetrics, PreprocessedImage
from app.pipeline.spatial_clustering import SpatialCluster
from app.worker.orchestrator import PipelineOrchestrator


def _make_box(x1: int, y1: int, x2: int, y2: int, text: str, score: float = 0.95) -> OcrTextBox:
    quad = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return OcrTextBox(text=text, score=score, box=quad)


def test_build_comprehensive_debug_bundle(tmp_path):
    orchestrator = PipelineOrchestrator()

    image = ImageModel(
        id="img_test_123",
        batch_id="batch_debug_test",
        filename="parcel_sample.jpg",
        content_type="image/jpeg",
        file_size=102400,
        s3_bucket="test-bucket",
        s3_key="batches/test/originals/parcel_sample.jpg",
    )

    from PIL import Image as PILImage
    bgr = np.zeros((600, 800, 3), dtype=np.uint8)
    pil = PILImage.new("RGB", (800, 600))
    metrics = ImageQualityMetrics(
        width=800,
        height=600,
        channels=3,
        blur_score=150.0,
        is_blurry=False,
        mean_brightness=110.0,
        contrast_std=45.0,
        exif_rotated=False,
    )
    preprocessed = PreprocessedImage(
        bgr_image=bgr,
        pil_image=pil,
        metrics=metrics,
    )

    rois_debug = [
        {"format_hint": "top_left", "bbox_xyxy": [10, 10, 200, 100], "score": 0.85},
    ]

    det_result = DetectionResult(
        model="owlv2",
        detections=[
            DetectionBox(class_name="parcel", score=0.92, bbox_xyxy=[50, 50, 700, 550]),
            DetectionBox(class_name="shipping_label", score=0.88, bbox_xyxy=[150, 150, 450, 400]),
        ],
    )

    overlay_ocr = OcrResult(
        engine="PaddleOCR",
        model_version="PP-OCRv5",
        lines=[_make_box(10, 10, 150, 40, "WT: 2.5 KG", 0.94)],
        full_text="WT: 2.5 KG",
        preprocessing_variant="clahe",
    )

    label_ocr = OcrResult(
        engine="PaddleOCR",
        model_version="PP-OCRv5",
        lines=[
            _make_box(150, 150, 400, 190, "AWB: 1Z9999999999999999", 0.98),
            _make_box(150, 200, 350, 240, "INVENTIMENT", 0.80),
        ],
        full_text="AWB: 1Z9999999999999999\nINVENTIMENT",
        preprocessing_variant="0_deg",
    )

    barcode_results = [
        BarcodeResult(raw_value="1Z9999999999999999", symbology="Code128", bbox_xyxy=[160, 260, 380, 320]),
    ]

    parsed_data = ExtractedLogisticsData(
        tracking_number="1Z9999999999999999",
        weight_value=2.5,
        weight_unit="kg",
        status="SUCCESS",
        provenance={
            "tracking_number": FieldProvenance(
                value="1Z9999999999999999",
                source="printed_barcode+label_ocr",
                raw_text="AWB: 1Z9999999999999999",
                ocr_confidence=0.98,
                conflict_status="AGREED",
                raw_candidates=["1Z9999999999999999", "INVENTIMENT"],
            ),
        },
        field_confidence={"tracking_number": 0.98, "weight": 0.94},
    )

    clusters = [
        SpatialCluster(
            bbox_xyxy=[150, 150, 450, 400],
            region_label="label_area",
            lines=[[_make_box(150, 150, 400, 190, "AWB: 1Z9999999999999999")]],
        )
    ]

    bundle = orchestrator._build_debug_bundle(
        image=image,
        rois_debug=rois_debug,
        overlay_ocr=overlay_ocr,
        label_ocr=label_ocr,
        label_crop_bbox=[150, 150, 450, 400],
        barcode_results=barcode_results,
        parsed_data=parsed_data,
        preprocessed=preprocessed,
        det_result=det_result,
        spatial_clusters=clusters,
    )

    # 1. Original image
    assert "original_image" in bundle
    assert bundle["original_image"]["dimensions"]["width"] == 800
    assert bundle["original_image"]["quality_metrics"]["blur_score"] == 150.0

    # 2. All candidate regions
    assert "candidate_regions" in bundle
    assert len(bundle["candidate_regions"]) >= 2

    # 3. All OCR detections and bounding boxes
    assert "all_ocr_detections" in bundle
    assert len(bundle["all_ocr_detections"]) == 3
    assert all("quad_box" in b and "bbox_xyxy" in b for b in bundle["all_ocr_detections"])

    # 4. Text from each region
    assert "text_by_region" in bundle
    assert "overlay_full_text" in bundle["text_by_region"]
    assert "label_full_text" in bundle["text_by_region"]

    # 5. Enhanced image variants
    assert "enhanced_image_variants" in bundle
    assert "clahe" in bundle["enhanced_image_variants"]

    # 6. Label crops
    assert "label_crop_info" in bundle
    assert bundle["label_crop_info"]["label_crop_bbox"] == [150, 150, 450, 400]

    # 7. Barcode decoder attempts
    assert "barcode_decoder_attempts" in bundle
    assert len(bundle["barcode_decoder_attempts"]) == 1

    # 8. Profile scores and matched anchors
    assert "profile_scores_and_matched_anchors" in bundle
    assert len(bundle["profile_scores_and_matched_anchors"]) >= 5
    profile_names = [p["profile_name"] for p in bundle["profile_scores_and_matched_anchors"]]
    assert "ekart" in profile_names
    assert "bottom_panel" in profile_names

    # 9. Parsed values and validation failures
    val_info = bundle["parsed_values_and_validation_failures"]
    assert val_info["parsed_values"]["tracking_number"] == "1Z9999999999999999"
    # INVENTIMENT should be recorded as validation failure
    assert len(val_info["validation_failures"]) >= 1
    fail_cands = [f["candidate"] for f in val_info["validation_failures"]]
    assert "INVENTIMENT" in fail_cands

    # 10. Final field provenance and confidence
    final_info = bundle["final_field_provenance_and_confidence"]
    assert "provenance" in final_info
    assert "field_confidence" in final_info
    assert final_info["provenance"]["tracking_number"]["value"] == "1Z9999999999999999"

    # Disk file created
    bundle_path = os.path.join("./debug_bundles", "batch_debug_test", "img_test_123_debug_bundle.json")
    assert os.path.exists(bundle_path)
    with open(bundle_path, "r", encoding="utf-8") as f:
        loaded = json.load(f)
        assert loaded["original_image"]["filename"] == "parcel_sample.jpg"
