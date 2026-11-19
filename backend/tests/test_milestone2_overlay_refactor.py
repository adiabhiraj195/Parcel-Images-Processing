"""
Unit tests for Milestone 2: Replace overlay ROI-first OCR.
Tests spatial clustering, line grouping, merge_text_boxes, controlled variant generation,
giant region filtering, and overlay deduplication.
"""

import numpy as np
import pytest

from app.pipeline.interfaces import OcrTextBox
from app.pipeline.spatial_clustering import (
    SpatialCluster,
    cluster_lines_spatially,
    group_boxes_into_lines,
    merge_text_boxes,
)
from app.pipeline.overlay_extractor import (
    CandidateOverlay,
    compute_overlay_iou,
    deduplicate_candidate_overlays,
    evaluate_image_quality,
    generate_controlled_variants,
    MultiFormatOverlayDetector,
)


def _make_box(x1: int, y1: int, x2: int, y2: int, text: str, score: float = 0.95) -> OcrTextBox:
    quad = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]
    return OcrTextBox(text=text, score=score, box=quad)


def test_group_boxes_into_lines():
    boxes = [
        _make_box(10, 20, 50, 40, "AWB:", 0.95),
        _make_box(60, 22, 160, 41, "123456789", 0.96),
        _make_box(10, 60, 50, 80, "WT:", 0.94),
        _make_box(60, 61, 120, 81, "2.5 KG", 0.93),
    ]
    lines = group_boxes_into_lines(boxes)
    assert len(lines) == 2
    # First line should have AWB: 123456789 in reading order
    assert [b.text for b in lines[0]] == ["AWB:", "123456789"]
    # Second line should have WT: 2.5 KG
    assert [b.text for b in lines[1]] == ["WT:", "2.5 KG"]


def test_cluster_lines_spatially():
    # Top-left metadata
    boxes_top_left = [
        _make_box(20, 30, 80, 50, "AWB:", 0.95),
        _make_box(90, 30, 200, 50, "PROC123456", 0.95),
        _make_box(20, 60, 80, 80, "WT: 1.5KG", 0.92),
    ]
    # Bottom panel metadata
    boxes_bottom = [
        _make_box(50, 920, 300, 950, "L: 20 W: 15 H: 10", 0.90),
    ]
    all_boxes = boxes_top_left + boxes_bottom
    lines = group_boxes_into_lines(all_boxes)
    clusters = cluster_lines_spatially(lines, image_width=1000, image_height=1000)

    # Should separate into spatial blocks
    assert len(clusters) >= 2
    labels = [c.region_label for c in clusters]
    assert "top_left" in labels
    assert "bottom_panel" in labels


def test_merge_text_boxes_deduplication():
    # Base box and overlapping box with slightly different coords
    base_boxes = [_make_box(10, 10, 100, 50, "TEST1", 0.90)]
    supplementary_boxes = [
        _make_box(12, 11, 98, 49, "TEST1", 0.95),  # High overlap, higher score
        _make_box(200, 200, 300, 250, "NEW_BOX", 0.88),  # No overlap
    ]
    merged = merge_text_boxes(base_boxes, supplementary_boxes, iou_thresh=0.4)
    assert len(merged) == 2
    # Should have updated the higher score box for TEST1
    test1_box = [b for b in merged if b.text == "TEST1"][0]
    assert test1_box.score == 0.95


def test_filter_giant_regions():
    detector = MultiFormatOverlayDetector()
    img_w, img_h = 1000, 1000
    # Region 1: Normal crop (100x200 = 20,000 px, 2%)
    normal_cand = CandidateOverlay("top_left", [10, 10, 210, 110], 0.9, np.zeros((100, 200, 3), dtype=np.uint8))
    # Region 2: Giant crop (800x800 = 640,000 px, 64%)
    giant_cand = CandidateOverlay("giant_noise", [50, 50, 850, 850], 0.5, np.zeros((800, 800, 3), dtype=np.uint8))

    valid = detector.filter_giant_regions([normal_cand, giant_cand], img_w, img_h, max_area_ratio=0.30)
    assert len(valid) == 1
    assert valid[0].format_hint == "top_left"


def test_deduplicate_candidate_overlays():
    c1 = CandidateOverlay("ekart_1", [10, 10, 200, 100], 0.85, np.zeros((90, 190, 3), dtype=np.uint8))
    c2 = CandidateOverlay("ekart_2", [15, 12, 195, 98], 0.92, np.zeros((86, 180, 3), dtype=np.uint8))
    c3 = CandidateOverlay("bottom_panel", [10, 800, 600, 950], 0.70, np.zeros((150, 590, 3), dtype=np.uint8))

    deduped = deduplicate_candidate_overlays([c1, c2, c3], iou_thresh=0.35)
    assert len(deduped) == 2
    # c2 had higher score (0.92 > 0.85), so c2 is retained over c1
    assert deduped[0].score == 0.92
    assert deduped[1].format_hint == "bottom_panel"


def test_evaluate_image_quality_and_controlled_variants():
    # 1. Dark image
    dark_img = np.full((100, 100, 3), 20, dtype=np.uint8)
    q_dark = evaluate_image_quality(dark_img)
    assert q_dark["mean_brightness"] < 80
    assert q_dark["dark_ratio"] > 0.5

    # Controlled variants selected adaptively based on quality
    dark_variants = generate_controlled_variants(dark_img, select_based_on_quality=True)
    assert "original" in dark_variants
    assert "gamma_corrected" in dark_variants
    assert "clahe" in dark_variants
    # Should include adaptive_thresholded for very dark
    assert "adaptive_thresholded" in dark_variants

    # 2. Bright normal image
    normal_img = np.full((100, 100, 3), 150, dtype=np.uint8)
    q_normal = evaluate_image_quality(normal_img)
    assert q_normal["mean_brightness"] >= 80

    normal_variants = generate_controlled_variants(normal_img, select_based_on_quality=True)
    assert "original" in normal_variants
    assert "clahe" in normal_variants
    # Should NOT waste time running gamma_corrected on bright image
    assert "gamma_corrected" not in normal_variants
