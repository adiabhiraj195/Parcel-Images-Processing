"""Unit tests for object detector adapter, IoU, and NMS suppression."""

from PIL import Image

from app.pipeline.interfaces import DetectionBox
from app.pipeline.detector import (
    compute_iou,
    non_max_suppression,
    MockObjectDetector,
)


def test_iou_calculation():
    """Verify standard Intersection over Union calculations."""
    box1 = [0, 0, 100, 100]
    box2 = [0, 0, 100, 100]
    assert compute_iou(box1, box2) == 1.0

    box3 = [100, 100, 200, 200]
    assert compute_iou(box1, box3) == 0.0

    # 50% horizontal overlap
    box4 = [50, 0, 150, 100]
    # Inter area = 50 * 100 = 5000; Union area = 10000 + 10000 - 5000 = 15000
    assert abs(compute_iou(box1, box4) - (1.0 / 3.0)) < 1e-4


def test_nms_suppresses_same_class_duplicates():
    """Overlapping detections of the same category should be merged."""
    boxes = [
        DetectionBox(class_name="parcel", score=0.95, bbox_xyxy=[10, 10, 100, 100]),
        DetectionBox(class_name="parcel", score=0.85, bbox_xyxy=[15, 12, 102, 98]),
        DetectionBox(class_name="parcel", score=0.91, bbox_xyxy=[200, 200, 300, 300]),
    ]
    deduped = non_max_suppression(boxes, iou_threshold=0.5)

    assert len(deduped) == 2
    assert deduped[0].score == 0.95
    assert deduped[1].score == 0.91


def test_nms_preserves_different_classes():
    """Overlapping detections of different classes (parcel vs shipping_label) must not suppress each other."""
    boxes = [
        DetectionBox(class_name="parcel", score=0.90, bbox_xyxy=[0, 0, 100, 100]),
        DetectionBox(class_name="shipping_label", score=0.88, bbox_xyxy=[10, 10, 80, 80]),
    ]
    deduped = non_max_suppression(boxes, iou_threshold=0.3)
    assert len(deduped) == 2


def test_nms_suppresses_contained_sub_boxes():
    """A sub-box contained inside a larger box of the same class should be merged."""
    boxes = [
        DetectionBox(class_name="parcel", score=0.92, bbox_xyxy=[100, 100, 800, 800]),
        DetectionBox(class_name="parcel", score=0.55, bbox_xyxy=[250, 250, 550, 550]),  # fully contained
    ]
    deduped = non_max_suppression(boxes, iou_threshold=0.5, containment_threshold=0.7)
    assert len(deduped) == 1
    assert deduped[0].score == 0.92


def test_mock_detector_schema():
    """Verify normalized output from mock detector adapter."""
    detector = MockObjectDetector()
    img = Image.new("RGB", (500, 400), color="white")
    res = detector.detect(img)

    assert res.model.startswith("mock-detector")
    assert len(res.detections) == 2
    classes = {d.class_name for d in res.detections}
    assert "parcel" in classes
    assert "shipping_label" in classes


def test_owlv2_detector_grounded_post_process():
    """Verify OwlV2ZeroShotDetector uses post_process_grounded_object_detection correctly."""
    import torch
    from unittest.mock import MagicMock
    from app.pipeline.detector import OwlV2ZeroShotDetector

    detector = OwlV2ZeroShotDetector()
    mock_processor = MagicMock()
    mock_model = MagicMock()

    mock_processor.return_value = {"pixel_values": torch.zeros((1, 3, 200, 200))}
    mock_model.return_value = MagicMock()
    mock_processor.post_process_grounded_object_detection.return_value = [
        {
            "boxes": torch.tensor([[10.0, 10.0, 100.0, 100.0], [30.0, 30.0, 80.0, 80.0]]),
            "scores": torch.tensor([0.95, 0.88]),
            "text_labels": ["cardboard box", "shipping label"],
            "labels": torch.tensor([1, 4]),
        }
    ]

    detector._processor = mock_processor
    detector._model = mock_model

    img = Image.new("RGB", (200, 200), color="white")
    res = detector.detect(img)

    assert mock_processor.post_process_grounded_object_detection.called
    assert len(res.detections) == 2
    classes = {d.class_name for d in res.detections}
    assert "parcel" in classes
    assert "shipping_label" in classes


def test_edge_proximity_evidence():
    """Box touching borders is detected as edge proximity evidence."""
    from app.pipeline.detector import check_edge_proximity

    # Box touching left edge
    near_edge, edges = check_edge_proximity([5, 50, 200, 200], image_width=500, image_height=500, margin_px=15)
    assert near_edge is True
    assert "left" in edges

    # Box centered, far from edges
    near_edge_center, _ = check_edge_proximity([100, 100, 300, 300], image_width=500, image_height=500, margin_px=15)
    assert near_edge_center is False


def test_scanner_bed_roi_filtering():
    """Only parcels inside active scanner-bed ROI should be counted as active parcels."""
    from app.pipeline.detector import filter_parcels_in_roi

    scanner_bed = [100, 100, 400, 400]
    detections = [
        DetectionBox(class_name="parcel", score=0.92, bbox_xyxy=[150, 150, 350, 350]),  # on bed
        DetectionBox(class_name="parcel", score=0.88, bbox_xyxy=[450, 50, 550, 150]),   # background parcel
        DetectionBox(class_name="shipping_label", score=0.90, bbox_xyxy=[200, 200, 250, 250]),
    ]

    active, background = filter_parcels_in_roi(detections, scanner_bed_roi=scanner_bed)
    assert len(active) == 1
    assert active[0].bbox_xyxy == [150, 150, 350, 350]
    assert len(background) == 1
    assert background[0].bbox_xyxy == [450, 50, 550, 150]
