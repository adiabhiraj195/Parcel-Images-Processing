"""Spatial text box line grouping and spatial clustering for full-resolution OCR.

Preserves all detected text boxes, their exact quad coordinates, and confidences.
Groups text boxes into reading-order lines and spatial clusters (e.g. top-left overlay,
bottom panel, physical shipping label area).
Provides tiled OCR to preserve full pixel resolution for small text.
"""

import logging
from typing import Any, List, Tuple
import cv2
import numpy as np

from app.pipeline.interfaces import IOcrEngine, OcrResult, OcrTextBox

logger = logging.getLogger("spatial_clustering")


class SpatialCluster:
    """A cohesive spatial grouping of text lines in the full image."""

    def __init__(
        self,
        bbox_xyxy: list[int],
        region_label: str,
        lines: list[list[OcrTextBox]],
    ):
        self.bbox_xyxy = bbox_xyxy
        self.region_label = region_label
        self.lines = lines
        self.text_lines: list[str] = []
        all_scores: list[float] = []

        for line_boxes in lines:
            line_str = " ".join(b.text for b in line_boxes if b.text.strip())
            if line_str:
                self.text_lines.append(line_str)
            for b in line_boxes:
                all_scores.append(b.score)

        self.full_text = "\n".join(self.text_lines)
        self.confidence = float(np.mean(all_scores)) if all_scores else 0.0

    def __repr__(self) -> str:
        return f"<SpatialCluster {self.region_label} bbox={self.bbox_xyxy} lines={len(self.text_lines)} conf={self.confidence:.2f}>"


def get_box_bounds(box: OcrTextBox) -> tuple[int, int, int, int]:
    """Extract [x1, y1, x2, y2] bounding rectangle from quad coordinates."""
    if not box.box:
        return (0, 0, 0, 0)
    xs = [pt[0] for pt in box.box]
    ys = [pt[1] for pt in box.box]
    return (int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys)))


def group_boxes_into_lines(boxes: list[OcrTextBox]) -> list[list[OcrTextBox]]:
    """
    Group detected OCR text boxes into reading-order lines.
    Preserves text box geometries and positions.
    """
    if not boxes:
        return []

    # Calculate bounding attributes for each box
    box_meta = []
    for b in boxes:
        x1, y1, x2, y2 = get_box_bounds(b)
        h = max(1, y2 - y1)
        cy = (y1 + y2) / 2.0
        box_meta.append((b, x1, y1, x2, y2, cy, h))

    # Sort boxes by top / center-y
    box_meta.sort(key=lambda item: item[5])

    lines: list[list[tuple[Any, int, int, int, int, float, int]]] = []

    for item in box_meta:
        b, x1, y1, x2, y2, cy, h = item
        matched_line = None

        for line in lines:
            # Check vertical overlap with average center-y of existing line
            line_cy = float(np.mean([m[5] for m in line]))
            line_h = float(np.mean([m[6] for m in line]))
            v_tolerance = max(8.0, line_h * 0.6)

            if abs(cy - line_cy) <= v_tolerance:
                matched_line = line
                break

        if matched_line is not None:
            matched_line.append(item)
        else:
            lines.append([item])

    # Sort lines vertically from top to bottom
    lines.sort(key=lambda l: float(np.mean([m[5] for m in l])))

    # Within each line, sort boxes horizontally from left to right
    sorted_lines: list[list[OcrTextBox]] = []
    for line in lines:
        line.sort(key=lambda m: m[1])  # Sort by x1
        sorted_lines.append([m[0] for m in line])

    return sorted_lines


def cluster_lines_spatially(
    lines: list[list[OcrTextBox]],
    image_width: int,
    image_height: int,
    vertical_gap_mult: float = 2.5,
) -> list[SpatialCluster]:
    """
    Group lines into spatial clusters based on geometric proximity.
    Labels clusters by spatial position (e.g. top_left, bottom_panel, center).
    """
    if not lines:
        return []

    line_meta = []
    for line in lines:
        if not line:
            continue
        xs = []
        ys = []
        for b in line:
            x1, y1, x2, y2 = get_box_bounds(b)
            xs.extend([x1, x2])
            ys.extend([y1, y2])
        if not xs or not ys:
            continue
        lx1, ly1, lx2, ly2 = min(xs), min(ys), max(xs), max(ys)
        lh = max(1, ly2 - ly1)
        line_meta.append((line, lx1, ly1, lx2, ly2, lh))

    if not line_meta:
        return []

    # Cluster adjacent lines
    raw_clusters: list[list[tuple[Any, int, int, int, int, int]]] = []

    for lm in line_meta:
        line, lx1, ly1, lx2, ly2, lh = lm
        matched_cluster = None

        for cluster in raw_clusters:
            # Check vertical proximity to last line in cluster
            last_line = cluster[-1]
            last_ly2 = last_line[4]
            last_lh = last_line[5]
            gap = ly1 - last_ly2
            max_gap = max(20.0, float(max(lh, last_lh)) * vertical_gap_mult)

            # Check horizontal overlap
            c_x1 = min(m[1] for m in cluster)
            c_x2 = max(m[3] for m in cluster)
            h_overlap = min(lx2, c_x2) - max(lx1, c_x1)
            min_width = min(lx2 - lx1, c_x2 - c_x1)

            if gap <= max_gap and (h_overlap > -30 or (min_width > 0 and h_overlap / float(min_width) > -0.2)):
                matched_cluster = cluster
                break

        if matched_cluster is not None:
            matched_cluster.append(lm)
        else:
            raw_clusters.append([lm])

    clusters: list[SpatialCluster] = []

    for rc in raw_clusters:
        c_lines = [item[0] for item in rc]
        c_x1 = max(0, min(item[1] for item in rc))
        c_y1 = max(0, min(item[2] for item in rc))
        c_x2 = min(image_width, max(item[3] for item in rc))
        c_y2 = min(image_height, max(item[4] for item in rc))

        # Determine region label based on position
        cx = (c_x1 + c_x2) / 2.0
        cy = (c_y1 + c_y2) / 2.0

        if cy < (image_height * 0.50) and cx < (image_width * 0.60):
            region_label = "top_left"
        elif cy > (image_height * 0.65):
            region_label = "bottom_panel"
        elif c_y1 > (image_height * 0.15) and c_y2 < (image_height * 0.85):
            region_label = "label_area"
        else:
            region_label = "general_cluster"

        clusters.append(
            SpatialCluster(
                bbox_xyxy=[c_x1, c_y1, c_x2, c_y2],
                region_label=region_label,
                lines=c_lines,
            )
        )

    return clusters


def compute_box_iou(box1: list[int], box2: list[int]) -> float:
    """Compute 2D Intersection-over-Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / float(union)


def merge_text_boxes(
    primary_boxes: list[OcrTextBox],
    supplementary_boxes: list[OcrTextBox],
    iou_thresh: float = 0.45,
) -> list[OcrTextBox]:
    """
    Merge text boxes from multiple passes (e.g. full-image + tiled OCR),
    deduplicating overlapping detections and retaining the higher confidence.
    """
    merged: list[OcrTextBox] = list(primary_boxes)

    for supp in supplementary_boxes:
        supp_b = get_box_bounds(supp)
        overlap = False

        for prim in merged:
            prim_b = get_box_bounds(prim)
            iou = compute_box_iou(supp_b, prim_b)
            if iou > iou_thresh:
                overlap = True
                # If supplementary has significantly better confidence or text, update
                if supp.score > prim.score and len(supp.text) >= len(prim.text):
                    prim.text = supp.text
                    prim.score = supp.score
                    prim.box = supp.box
                break

        if not overlap:
            merged.append(supp)

    return merged


def run_tiled_ocr(
    ocr_engine: IOcrEngine,
    bgr_image: np.ndarray,
    tile_w: int = 960,
    tile_h: int = 960,
    overlap_px: int = 140,
) -> list[OcrTextBox]:
    """
    Run tiled text detection on large images, preserving full native resolution
    without downscaling small text. Maps local tile quads back to global image coordinates.
    """
    h, w = bgr_image.shape[:2]

    # If image fits within a single tile, return empty (standard OCR handles it)
    if w <= tile_w and h <= tile_h:
        return []

    tiled_boxes: list[OcrTextBox] = []
    x_stride = max(100, tile_w - overlap_px)
    y_stride = max(100, tile_h - overlap_px)

    y_starts = list(range(0, h, y_stride))
    x_starts = list(range(0, w, x_stride))

    # Clamp the last tile to image edge
    if y_starts[-1] + tile_h > h and len(y_starts) > 1:
        y_starts[-1] = max(0, h - tile_h)
    if x_starts[-1] + tile_w > w and len(x_starts) > 1:
        x_starts[-1] = max(0, w - tile_w)

    # De-duplicate start points
    y_starts = sorted(list(set(y_starts)))
    x_starts = sorted(list(set(x_starts)))

    for y0 in y_starts:
        for x0 in x_starts:
            y1 = min(h, y0 + tile_h)
            x1 = min(w, x0 + tile_w)
            tile = bgr_image[y0:y1, x0:x1]

            if tile.size == 0 or tile.shape[0] < 30 or tile.shape[1] < 30:
                continue

            try:
                tile_res = ocr_engine.extract_text(
                    tile,
                    variant_label=f"tile_{x0}_{y0}",
                    source_crop="tile",
                )
                for tb in tile_res.lines:
                    # Offset local polygon to global image coordinates
                    global_quad = []
                    for pt in tb.box:
                        global_quad.append([int(pt[0] + x0), int(pt[1] + y0)])
                    tiled_boxes.append(
                        OcrTextBox(
                            text=tb.text,
                            score=tb.score,
                            box=global_quad,
                            source_crop="tiled_ocr",
                            preprocessing_variant=f"tile_{x0}_{y0}",
                        )
                    )
            except Exception as e:
                logger.debug(f"Tile OCR failed at ({x0}, {y0}): {e}")

    return tiled_boxes
