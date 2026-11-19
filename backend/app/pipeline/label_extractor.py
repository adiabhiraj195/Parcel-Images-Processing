"""
Physical shipping label extraction pipeline (Milestone 3).

Provides robust multi-cue physical label localization:
1. Spatial clustering & text-detection density (does not rely solely on OWLv2).
2. Barcode-anchored candidate regions (both decoded barcodes and visual barcode stripes).
3. OWLv2 zero-shot detection preserved as optional evidence.
4. Multi-evidence candidate fusion and deduplication.
5. Original, enhanced (CLAHE/sharpened), and rotated (0°, 90°, 180°, 270°, deskewed) crops.
6. Independent barcode detection on both full image and label crop variants.
7. Full text line, bounding box, and confidence extraction.
"""

from dataclasses import dataclass, field
import logging
from typing import Any, List, Optional, Tuple
import cv2
import numpy as np

from app.pipeline.interfaces import (
    BarcodeResult,
    DetectionBox,
    DetectionResult,
    IBarcodeDecoder,
    IOcrEngine,
    OcrResult,
    OcrTextBox,
)
from app.pipeline.spatial_clustering import SpatialCluster, get_box_bounds, compute_box_iou

logger = logging.getLogger("label_extractor")


@dataclass
class LabelCandidate:
    """A proposed physical shipping label region."""
    bbox_xyxy: list[int]
    source: str  # "barcode_anchor", "text_cluster", "owlv2_detector", "multi_evidence", "visual_barcode"
    score: float
    aspect_ratio: float = 1.0
    matched_barcode: Optional[BarcodeResult] = None
    matched_cluster: Optional[SpatialCluster] = None
    details: dict[str, Any] = field(default_factory=dict)


def compute_iou(box1: list[int], box2: list[int]) -> float:
    """Compute Intersection over Union between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])

    union_area = area1 + area2 - inter_area
    if union_area <= 0:
        return 0.0
    return inter_area / float(union_area)


def compute_containment(box1: list[int], box2: list[int]) -> float:
    """Compute containment ratio (intersection / min_area)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(1, (box1[2] - box1[0]) * (box1[3] - box1[1]))
    area2 = max(1, (box2[2] - box2[0]) * (box2[3] - box2[1]))

    return inter_area / float(min(area1, area2))


def clamp_box(box: list[int], max_w: int, max_h: int) -> list[int]:
    """Clamp [x1, y1, x2, y2] to image dimensions."""
    return [
        max(0, min(max_w - 1, int(box[0]))),
        max(0, min(max_h - 1, int(box[1]))),
        max(0, min(max_w, int(box[2]))),
        max(0, min(max_h, int(box[3]))),
    ]


# -----------------------------------------------------------------------------
# 1. Candidate Generation from Barcodes
# -----------------------------------------------------------------------------

def find_label_candidates_from_barcodes(
    barcodes: list[BarcodeResult],
    img_w: int,
    img_h: int,
) -> list[LabelCandidate]:
    """
    Expand decoded barcode bounding boxes into shipping label candidate regions.
    Shipping labels typically enclose a barcode with text printed above/below.
    """
    candidates = []
    for b in barcodes:
        if not b.bbox_xyxy:
            continue
        bx1, by1, bx2, by2 = b.bbox_xyxy
        bw = max(1, bx2 - bx1)
        bh = max(1, by2 - by1)

        # Shipping labels typically extend 1.5x - 2.5x barcode height and 0.3x width
        pad_y = int(bh * 2.0)
        pad_x = int(bw * 0.40)

        lx1 = max(0, bx1 - pad_x)
        ly1 = max(0, by1 - pad_y)
        lx2 = min(img_w, bx2 + pad_x)
        ly2 = min(img_h, by2 + pad_y)

        # Ensure minimal label size
        if (lx2 - lx1) < 80 or (ly2 - ly1) < 80:
            continue

        aspect = (lx2 - lx1) / float(max(1, ly2 - ly1))
        candidates.append(
            LabelCandidate(
                bbox_xyxy=[lx1, ly1, lx2, ly2],
                source="barcode_anchor",
                score=0.92,
                aspect_ratio=aspect,
                matched_barcode=b,
                details={"barcode_val": b.raw_value, "barcode_symbology": b.symbology},
            )
        )
    return candidates


def detect_visual_barcode_stripes(
    bgr_image: np.ndarray,
    min_area_ratio: float = 0.003,
    max_area_ratio: float = 0.25,
) -> list[LabelCandidate]:
    """
    Classical edge-gradient search for 1D barcode patterns (parallel black/white bars).
    Useful when a barcode is visible on the label but not yet decoded.
    """
    h, w = bgr_image.shape[:2]
    total_area = float(h * w)
    candidates = []

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY) if len(bgr_image.shape) == 3 else bgr_image
    # Scharr gradient in X direction to isolate vertical bars
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, -1)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, -1)
    gradient = cv2.subtract(grad_x, grad_y)
    gradient = cv2.convertScaleAbs(gradient)

    blurred = cv2.blur(gradient, (9, 9))
    _, thresh = cv2.threshold(blurred, 150, 255, cv2.THRESH_BINARY)

    # Morphological closing with rectangular kernel to form dense barcode blob
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    closed = cv2.erode(closed, None, iterations=2)
    closed = cv2.dilate(closed, None, iterations=2)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        box_area = cw * ch
        area_ratio = box_area / total_area
        if not (min_area_ratio <= area_ratio <= max_area_ratio):
            continue

        aspect = cw / float(max(1, ch))
        if 1.4 <= aspect <= 7.0:
            # Expand to include label text around the barcode
            pad_y = int(ch * 1.8)
            pad_x = int(cw * 0.3)
            lx1 = max(0, x - pad_x)
            ly1 = max(0, y - pad_y)
            lx2 = min(w, x + cw + pad_x)
            ly2 = min(h, y + ch + pad_y)

            candidates.append(
                LabelCandidate(
                    bbox_xyxy=[lx1, ly1, lx2, ly2],
                    source="visual_barcode",
                    score=0.70,
                    aspect_ratio=(lx2 - lx1) / float(max(1, ly2 - ly1)),
                    details={"contour_aspect": aspect},
                )
            )
    return candidates


def detect_visual_sticker_contours(
    bgr_image: np.ndarray,
    min_area_ratio: float = 0.003,
    max_area_ratio: float = 0.25,
) -> list[LabelCandidate]:
    """
    Classical edge and contrast contour search for white / high-contrast
    rectangular shipping label stickers affixed to parcels.
    """
    h, w = bgr_image.shape[:2]
    total_area = float(h * w)
    candidates: list[LabelCandidate] = []

    gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY) if len(bgr_image.shape) == 3 else bgr_image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        x, y, cw, ch = cv2.boundingRect(cnt)
        box_area = cw * ch
        area_ratio = box_area / total_area
        if not (min_area_ratio <= area_ratio <= max_area_ratio):
            continue

        aspect = cw / float(max(1, ch))
        if 0.35 <= aspect <= 3.0:
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.04 * peri, True)
            if len(approx) == 4:
                candidates.append(
                    LabelCandidate(
                        bbox_xyxy=[x, y, x + cw, y + ch],
                        source="visual_sticker",
                        score=0.62,
                        aspect_ratio=aspect,
                        details={"contour_aspect": aspect, "type": "rectangular_sticker"},
                    )
                )
    return candidates


# -----------------------------------------------------------------------------
# 2. Candidate Generation from Text Detections & Spatial Clusters
# -----------------------------------------------------------------------------

def find_label_candidates_from_text(
    clusters: list[SpatialCluster],
    img_w: int,
    img_h: int,
) -> list[LabelCandidate]:
    """
    Identify physical shipping labels from text-detection spatial clustering.
    A shipping label is a dense cluster of multiple text lines (typically >= 3 lines)
    situated in the parcel body (not camera/DWS overlay status bars).
    """
    candidates = []
    for cl in clusters:
        # Require multiple text lines
        if len(cl.text_lines) < 3:
            continue

        cx1, cy1, cx2, cy2 = cl.bbox_xyxy
        cw = max(1, cx2 - cx1)
        ch = max(1, cy2 - cy1)
        area_ratio = (cw * ch) / float(max(1, img_w * img_h))

        # Reject giant whole-screen clusters and tiny specks
        if area_ratio > 0.40 or area_ratio < 0.005:
            continue

        # Reject pure top status-bar overlay (e.g. green text at top 10%)
        if cy2 < (img_h * 0.15) and cl.region_label == "top_left":
            continue
        # Reject bottom panel overlay
        if cy1 > (img_h * 0.88) and cl.region_label == "bottom_panel":
            continue

        # Add 12% padding to capture label border and surrounding context
        pad_x = int(cw * 0.12)
        pad_y = int(ch * 0.12)
        lx1 = max(0, cx1 - pad_x)
        ly1 = max(0, cy1 - pad_y)
        lx2 = min(img_w, cx2 + pad_x)
        ly2 = min(img_h, cy2 + pad_y)

        aspect = (lx2 - lx1) / float(max(1, ly2 - ly1))
        # Confidence scales with number of text lines and OCR score
        line_factor = min(1.0, len(cl.text_lines) / 6.0)
        score = 0.75 + 0.15 * line_factor * cl.confidence

        candidates.append(
            LabelCandidate(
                bbox_xyxy=[lx1, ly1, lx2, ly2],
                source="text_cluster",
                score=score,
                aspect_ratio=aspect,
                matched_cluster=cl,
                details={"text_lines_count": len(cl.text_lines), "region_label": cl.region_label},
            )
        )
    return candidates


# -----------------------------------------------------------------------------
# 3. Candidate Generation from OWLv2 Detector (Optional Evidence)
# -----------------------------------------------------------------------------

def find_label_candidates_from_detector(
    det_result: DetectionResult,
) -> list[LabelCandidate]:
    """
    Extract candidate label regions from zero-shot detector (OWLv2) as optional evidence.
    """
    candidates = []
    for d in det_result.detections:
        if d.class_name == "shipping_label" and d.score >= 0.15:
            bw = max(1, d.bbox_xyxy[2] - d.bbox_xyxy[0])
            bh = max(1, d.bbox_xyxy[3] - d.bbox_xyxy[1])
            aspect = bw / float(bh)
            candidates.append(
                LabelCandidate(
                    bbox_xyxy=list(d.bbox_xyxy),
                    source="owlv2_detector",
                    score=float(d.score),
                    aspect_ratio=aspect,
                    details={"detector_model": det_result.model},
                )
            )
    return candidates


# -----------------------------------------------------------------------------
# 4. Multi-Evidence Fusion & Deduplication
# -----------------------------------------------------------------------------

def resolve_label_candidates(
    candidates: list[LabelCandidate],
    img_w: int,
    img_h: int,
    iou_thresh: float = 0.30,
) -> list[LabelCandidate]:
    """
    Fuse multi-cue label candidates, merging overlapping detections across sources
    (barcode + text cluster + OWLv2) into high-confidence multi-evidence candidates.
    """
    if not candidates:
        return []

    # Sort candidates by score descending
    sorted_cands = sorted(candidates, key=lambda c: c.score, reverse=True)
    fused: list[LabelCandidate] = []

    for cand in sorted_cands:
        matched = False
        for existing in fused:
            iou = compute_iou(cand.bbox_xyxy, existing.bbox_xyxy)
            cont = compute_containment(cand.bbox_xyxy, existing.bbox_xyxy)

            if iou > iou_thresh or cont > 0.55:
                # Merge into union bounding box
                ux1 = min(cand.bbox_xyxy[0], existing.bbox_xyxy[0])
                uy1 = min(cand.bbox_xyxy[1], existing.bbox_xyxy[1])
                ux2 = max(cand.bbox_xyxy[2], existing.bbox_xyxy[2])
                uy2 = max(cand.bbox_xyxy[3], existing.bbox_xyxy[3])
                existing.bbox_xyxy = clamp_box([ux1, uy1, ux2, uy2], img_w, img_h)

                # Upgrade to multi_evidence and boost confidence
                existing.source = "multi_evidence"
                existing.score = min(0.98, max(existing.score, cand.score) + 0.10)
                if cand.matched_barcode and not existing.matched_barcode:
                    existing.matched_barcode = cand.matched_barcode
                if cand.matched_cluster and not existing.matched_cluster:
                    existing.matched_cluster = cand.matched_cluster
                existing.details.update(cand.details)
                matched = True
                break

        if not matched:
            fused.append(cand)

    # Re-sort fused candidates by score descending
    fused.sort(key=lambda c: c.score, reverse=True)
    return fused


# -----------------------------------------------------------------------------
# 5. Crop Variants: Rotations and Enhancements
# -----------------------------------------------------------------------------

def detect_crop_skew_angle(crop_bgr: np.ndarray) -> float:
    """Detect dominant text/line orientation angle in degrees (-30° to +30°)."""
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    try:
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY) if len(crop_bgr.shape) == 3 else crop_bgr
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 70, minLineLength=40, maxLineGap=8)
        if lines is not None and len(lines) >= 4:
            angles = []
            for line in lines:
                x1, y1, x2, y2 = line[0]
                ang = np.degrees(np.arctan2(y2 - y1, x2 - x1))
                if -40.0 < ang < 40.0:
                    angles.append(ang)
            if angles:
                median_ang = float(np.median(angles))
                if abs(median_ang) > 1.0:
                    return median_ang
    except Exception:
        pass
    return 0.0


def rotate_image(image: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rotate image by arbitrary angle preserving content."""
    if abs(angle_deg) < 0.5:
        return image
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle_deg, 1.0)
    cos = abs(matrix[0, 0])
    sin = abs(matrix[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    matrix[0, 2] += (new_w / 2) - center[0]
    matrix[1, 2] += (new_h / 2) - center[1]
    return cv2.warpAffine(image, matrix, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def generate_label_crops(
    crop_bgr: np.ndarray,
    try_rotations: bool = True,
) -> dict[str, np.ndarray]:
    """
    Generate original, enhanced, and rotated label crops:
    - 0_deg: Original crop
    - 90_deg: Rotated 90° clockwise
    - 180_deg: Rotated 180° (upside down)
    - 270_deg: Rotated 270° clockwise (90° counter-clockwise)
    - clahe: Contrast enhanced (for faded thermal shipping labels)
    - deskewed: Skew-corrected if slight tilt detected
    """
    variants: dict[str, np.ndarray] = {"0_deg": crop_bgr}

    if crop_bgr is None or crop_bgr.size == 0:
        return variants

    # CLAHE contrast enhancement
    try:
        lab = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        enhanced_clahe = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)
        variants["clahe"] = enhanced_clahe
    except Exception:
        pass

    # Skew correction
    skew = detect_crop_skew_angle(crop_bgr)
    if abs(skew) > 1.5:
        deskewed = rotate_image(crop_bgr, -skew)
        variants["deskewed"] = deskewed

    # Orthogonal rotations
    if try_rotations:
        variants["90_deg"] = cv2.rotate(crop_bgr, cv2.ROTATE_90_CLOCKWISE)
        variants["180_deg"] = cv2.rotate(crop_bgr, cv2.ROTATE_180)
        variants["270_deg"] = cv2.rotate(crop_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)

    return variants


# -----------------------------------------------------------------------------
# 6. High-Level Label Extractor Workflow
# -----------------------------------------------------------------------------

class PhysicalLabelExtractor:
    """
    Orchestrates physical shipping label localization, barcode decoding,
    and multi-variant OCR.
    """

    def __init__(
        self,
        ocr_engine: IOcrEngine,
        barcode_decoder: IBarcodeDecoder,
    ):
        self.ocr_engine = ocr_engine
        self.barcode_decoder = barcode_decoder
        self.last_candidates: list[LabelCandidate] = []

    def extract_label_and_barcodes(
        self,
        bgr_image: np.ndarray,
        det_result: Optional[DetectionResult] = None,
        full_ocr_lines: Optional[list[OcrTextBox]] = None,
        spatial_clusters: Optional[list[SpatialCluster]] = None,
        overlay_bboxes: Optional[list[list[int]]] = None,
    ) -> Tuple[Optional[OcrResult], list[BarcodeResult], Optional[list[int]], Optional[np.ndarray]]:
        """
        Extract physical label using multi-cue localization, try original,
        enhanced and rotated label crops, decode barcodes independently on full
        image and label crop, and return all text lines, bounding boxes and confidence.
        Explicitly excludes machine metadata overlay boxes from physical-label candidates.
        """
        img_h, img_w = bgr_image.shape[:2]
        all_barcodes: list[BarcodeResult] = []
        seen_barcode_values: set[str] = set()

        # ---------------------------------------------------------------------
        # Step 1: Full-Image Independent Barcode Detection
        # ---------------------------------------------------------------------
        full_barcodes = self.barcode_decoder.decode(bgr_image)
        for b in full_barcodes:
            b.source_crop = "full_image"
            if b.raw_value not in seen_barcode_values:
                seen_barcode_values.add(b.raw_value)
                all_barcodes.append(b)
                logger.info(f"   ├─ Full-image barcode decoded: {b.symbology} = '{b.raw_value}' (box={b.bbox_xyxy})")

        # ---------------------------------------------------------------------
        # Step 2: Multi-Cue Candidate Label Localization
        # ---------------------------------------------------------------------
        candidates: list[LabelCandidate] = []

        # A: From decoded barcodes
        barcode_cands = find_label_candidates_from_barcodes(all_barcodes, img_w, img_h)
        candidates.extend(barcode_cands)

        # B: From text-detection boxes & spatial clusters
        if spatial_clusters:
            text_cands = find_label_candidates_from_text(spatial_clusters, img_w, img_h)
            candidates.extend(text_cands)

        # C: From OWLv2 object detector (optional evidence)
        if det_result:
            det_cands = find_label_candidates_from_detector(det_result)
            candidates.extend(det_cands)

        # D: From visual barcode stripes
        visual_barcode_cands = detect_visual_barcode_stripes(bgr_image)
        candidates.extend(visual_barcode_cands)

        # E: From visual sticker contours
        sticker_cands = detect_visual_sticker_contours(bgr_image)
        candidates.extend(sticker_cands)

        # Explicitly filter out any candidate overlapping with machine metadata overlay regions
        if overlay_bboxes:
            filtered_candidates = []
            for cand in candidates:
                is_overlay = False
                for obox in overlay_bboxes:
                    iou = compute_iou(cand.bbox_xyxy, obox)
                    cont = compute_containment(cand.bbox_xyxy, obox)
                    if iou > 0.15 or cont > 0.35:
                        is_overlay = True
                        logger.debug(f"Rejecting candidate {cand.bbox_xyxy} overlapping machine overlay {obox}")
                        break
                if not is_overlay:
                    filtered_candidates.append(cand)
            candidates = filtered_candidates

        # Fuse multi-evidence candidates and deduplicate
        fused_candidates = resolve_label_candidates(candidates, img_w, img_h)
        self.last_candidates = fused_candidates

        label_crop_bgr: Optional[np.ndarray] = None
        label_bbox: Optional[list[int]] = None

        if fused_candidates:
            best_candidate = fused_candidates[0]
            label_bbox = best_candidate.bbox_xyxy
            logger.info(
                f"   ├─ Selected physical label candidate [{best_candidate.source}] "
                f"bbox={label_bbox} score={best_candidate.score:.2f}"
            )
            x1, y1, x2, y2 = label_bbox
            label_crop_bgr = bgr_image[y1:y2, x1:x2].copy()
        else:
            logger.info("   ├─ No specific label candidate found; scanning full image")

        # ---------------------------------------------------------------------
        # Step 3: Barcode Decoding on Label Crop & Rotated Variants
        # ---------------------------------------------------------------------
        crop_variants: dict[str, np.ndarray] = {}
        if label_crop_bgr is not None and label_crop_bgr.size > 0:
            crop_variants = generate_label_crops(label_crop_bgr, try_rotations=True)

            for var_name, var_img in crop_variants.items():
                if var_img is None or var_img.size == 0:
                    continue
                var_barcodes = self.barcode_decoder.decode(var_img)
                for b in var_barcodes:
                    b.source_crop = f"label_crop_{var_name}"
                    if b.raw_value not in seen_barcode_values:
                        seen_barcode_values.add(b.raw_value)
                        all_barcodes.append(b)
                        logger.info(f"   ├─ Barcode decoded on label variant '{var_name}': {b.symbology} = '{b.raw_value}'")

        # ---------------------------------------------------------------------
        # Step 4: Multi-Variant Label OCR
        # ---------------------------------------------------------------------
        best_ocr_result: Optional[OcrResult] = None

        if label_crop_bgr is not None and label_crop_bgr.size > 0:
            # Try original label crop
            try:
                best_ocr_result = self.ocr_engine.extract_text(
                    label_crop_bgr,
                    variant_label="label_crop_0deg",
                    source_crop="shipping_label",
                )
                logger.info(
                    f"   ├─ Label OCR (0°): extracted {len(best_ocr_result.lines)} text line(s)"
                )
            except Exception as e:
                logger.warning(f"Label OCR (0°) failed: {e}")

            # If 0° OCR extracted very few lines (< 3 lines) or low confidence, try rotated & CLAHE variants
            should_try_variants = (
                best_ocr_result is None
                or len(best_ocr_result.lines) < 3
                or (best_ocr_result.lines and np.mean([l.score for l in best_ocr_result.lines]) < 0.70)
            )

            if should_try_variants:
                variant_priority = ["clahe", "90_deg", "270_deg", "180_deg", "deskewed"]
                for v_key in variant_priority:
                    if v_key not in crop_variants:
                        continue
                    v_img = crop_variants[v_key]
                    try:
                        v_res = self.ocr_engine.extract_text(
                            v_img,
                            variant_label=f"label_crop_{v_key}",
                            source_crop="shipping_label",
                        )
                        if best_ocr_result is None or len(v_res.lines) > len(best_ocr_result.lines):
                            logger.info(
                                f"   ├─ Better OCR from variant '{v_key}': "
                                f"{len(v_res.lines)} line(s) vs {len(best_ocr_result.lines) if best_ocr_result else 0}"
                            )
                            best_ocr_result = v_res
                    except Exception as ve:
                        logger.debug(f"Label variant OCR '{v_key}' error: {ve}")

        # Step 5: Incorporate full-resolution OCR text boxes situated inside label_bbox
        if label_bbox and full_ocr_lines:
            lx1, ly1, lx2, ly2 = label_bbox
            inside_boxes = []
            for b in full_ocr_lines:
                bx1, by1, bx2, by2 = get_box_bounds(b)
                # Check if box center falls inside label bbox
                bcx = (bx1 + bx2) / 2.0
                bcy = (by1 + by2) / 2.0
                if lx1 <= bcx <= lx2 and ly1 <= bcy <= ly2:
                    inside_boxes.append(b)

            if inside_boxes:
                if best_ocr_result is None or not best_ocr_result.lines:
                    best_ocr_result = OcrResult(
                        engine=self.ocr_engine.__class__.__name__,
                        model_version=getattr(self.ocr_engine, "MODEL_VERSION", "v1"),
                        lines=inside_boxes,
                        full_text="\n".join(b.text for b in inside_boxes),
                        source_crop="shipping_label",
                        preprocessing_variant="full_res_projected",
                    )
                else:
                    # Merge any text lines that weren't captured in crop OCR
                    crop_texts = {l.text.strip().lower() for l in best_ocr_result.lines if l.text.strip()}
                    for b in inside_boxes:
                        if b.text.strip().lower() not in crop_texts:
                            best_ocr_result.lines.append(b)
                    best_ocr_result.full_text = "\n".join(l.text for l in best_ocr_result.lines)

        return best_ocr_result, all_barcodes, label_bbox, label_crop_bgr
