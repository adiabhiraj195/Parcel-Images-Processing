"""Multi-format candidate overlay detector and preprocessor for warehouse scanner images.

Supports:
1. Green DWS overlay (top-left green text)
2. Yellow DWS overlay (top-left yellow text)
3. White-on-black scanner overlay (top-left monospaced text on black)
4. Ekart-style overlay (small black information box with white text)
5. Scanner information panel (grey/white panel near bottom of image)
6. Fallback region proposals (top-left ROI, bottom strip)

HSV masks and contour detectors act as candidate-region generators, feeding
contrast-enhanced and multi-scale crops to OCR engines.
"""

import logging
import re
from typing import Any

import cv2
import numpy as np

from app.config import settings
from app.pipeline.interfaces import OcrResult, OcrTextBox

logger = logging.getLogger("overlay_extractor")


class CandidateOverlay:
    """A detected candidate region containing machine metadata overlay."""

    def __init__(
        self,
        format_hint: str,
        bbox_xyxy: list[int],
        score: float,
        crop_bgr: np.ndarray,
        variants: dict[str, np.ndarray] | None = None,
    ):
        self.format_hint = format_hint
        self.bbox_xyxy = bbox_xyxy
        self.score = score
        self.crop_bgr = crop_bgr
        self.variants = variants or {}

    def __repr__(self) -> str:
        return f"<CandidateOverlay format={self.format_hint} bbox={self.bbox_xyxy} score={self.score:.2f}>"


# ---------- Backward Compatible Regex Patterns and OverlayData ----------

OVERLAY_FIELD_PATTERNS: dict[str, re.Pattern] = {
    "location": re.compile(r"Location[:\s.]+([^\n\r]+)", re.IGNORECASE),
    "awb_no": re.compile(r"AWB\s*(?:No|Number)?[:\s.]+([A-Z0-9-]{8,35})", re.IGNORECASE),
    "length": re.compile(r"Length[:\s.]+([\d]+(?:\.[\d]+)?)\s*(cm|mm|m|in)?", re.IGNORECASE),
    "width": re.compile(r"Width[:\s.]+([\d]+(?:\.[\d]+)?)\s*(cm|mm|m|in)?", re.IGNORECASE),
    "height": re.compile(r"Height[:\s.]+([\d]+(?:\.[\d]+)?)\s*(cm|mm|m|in)?", re.IGNORECASE),
    "volume": re.compile(
        r"R\.?\s*Vol\.?[:\s.]+([\d]+(?:\.[\d]+)?)\s*(cm|mm|m)?(?:\s*\(?\s*cubic\s*\)?)?",
        re.IGNORECASE,
    ),
    "weight": re.compile(
        r"Weight[:\s.]+([\d]+(?:\.[\d]+)?)\s*(gm|gms|g|9m|9ms|kg|kgs|lb|lbs|oz)?",
        re.IGNORECASE,
    ),
    "time": re.compile(r"Time[:\s.]+([^\n\r]+)", re.IGNORECASE),
}


class OverlayData:
    """Parsed fields from overlay text for backward compatibility."""

    def __init__(self):
        self.location: str | None = None
        self.awb_no: str | None = None
        self.length: float | None = None
        self.width: float | None = None
        self.height: float | None = None
        self.dimension_unit: str | None = "cm"
        self.volume: float | None = None
        self.weight_value: float | None = None
        self.weight_unit: str | None = "g"
        self.timestamp: str | None = None
        self.raw_lines: list[str] = []
        self.confidence: float = 0.0

    def is_populated(self) -> bool:
        return (
            self.awb_no is not None
            or self.weight_value is not None
            or (self.length is not None and self.width is not None)
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "location": self.location,
            "awb_no": self.awb_no,
            "length": self.length,
            "width": self.width,
            "height": self.height,
            "dimension_unit": self.dimension_unit,
            "volume": self.volume,
            "weight_value": self.weight_value,
            "weight_unit": self.weight_unit,
            "timestamp": self.timestamp,
            "raw_lines": self.raw_lines,
            "confidence": self.confidence,
        }


# ---------- Color Segmentation Ranges ----------

# Green saturated text (DWS Green Overlay)
_GREEN_LOWER = np.array([35, 50, 60])
_GREEN_UPPER = np.array([88, 255, 255])
_MIN_GREEN_PIXELS = 60

# Yellow saturated text (DWS Yellow Overlay)
_YELLOW_LOWER = np.array([15, 60, 120])
_YELLOW_UPPER = np.array([40, 255, 255])
_MIN_YELLOW_PIXELS = 80


def detect_yellow_overlay(bgr_image: np.ndarray) -> np.ndarray | None:
    """Return a binary mask of yellow overlay pixels, or None if absent."""
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _YELLOW_LOWER, _YELLOW_UPPER)
    if cv2.countNonZero(mask) < _MIN_YELLOW_PIXELS:
        return None
    return mask


def detect_green_overlay(bgr_image: np.ndarray) -> np.ndarray | None:
    """Return a binary mask of green overlay pixels, or None if absent."""
    hsv = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, _GREEN_LOWER, _GREEN_UPPER)
    if cv2.countNonZero(mask) < _MIN_GREEN_PIXELS:
        return None
    return mask


def evaluate_image_quality(img_bgr: np.ndarray) -> dict[str, Any]:
    """Measure image illumination, contrast, and dark pixel ratio."""
    if img_bgr is None or img_bgr.size == 0:
        return {"mean_brightness": 128.0, "contrast": 50.0, "dark_ratio": 0.0, "is_dark": False}
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    mean_val = float(np.mean(gray))
    std_val = float(np.std(gray))
    dark_ratio = float(np.mean(gray < 60))
    bright_ratio = float(np.mean(gray > 200))
    is_dark = (mean_val < 85.0) or (dark_ratio > 0.40)
    return {
        "mean_brightness": round(mean_val, 2),
        "contrast": round(std_val, 2),
        "dark_ratio": round(dark_ratio, 2),
        "bright_ratio": round(bright_ratio, 2),
        "is_dark": is_dark,
    }


def generate_controlled_variants(
    img_bgr: np.ndarray,
    select_based_on_quality: bool = True,
) -> dict[str, np.ndarray]:
    """
    Generate controlled image variants:
    - Original
    - Grayscale
    - CLAHE
    - Gamma-corrected
    - Adaptive-thresholded
    Selects variants based on image quality instead of running every variant blindly.
    """
    if img_bgr is None or img_bgr.size == 0:
        return {}

    variants: dict[str, np.ndarray] = {"original": img_bgr}
    quality = evaluate_image_quality(img_bgr)
    is_dark = quality["is_dark"]
    low_contrast = quality["contrast"] < 35.0

    # 1. Grayscale (converted to 3-channel BGR for model input)
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY) if len(img_bgr.shape) == 3 else img_bgr
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    try:
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        l_clahe = clahe.apply(l)
        enhanced_clahe = cv2.cvtColor(cv2.merge((l_clahe, a, b)), cv2.COLOR_LAB2BGR)
    except Exception:
        enhanced_clahe = gray_bgr

    # 3. Gamma correction (lift shadows in dark images)
    gamma = 1.8
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    gamma_corrected = cv2.LUT(img_bgr, table)

    # 4. Adaptive thresholded (binarized high contrast)
    adapt_bin = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 3
    )
    if np.mean(adapt_bin) < 127:
        adapt_bin = cv2.bitwise_not(adapt_bin)
    adapt_bgr = cv2.cvtColor(adapt_bin, cv2.COLOR_GRAY2BGR)

    if not select_based_on_quality:
        variants["grayscale"] = gray_bgr
        variants["clahe"] = enhanced_clahe
        variants["gamma_corrected"] = gamma_corrected
        variants["adaptive_thresholded"] = adapt_bgr
        variants["binarized"] = adapt_bgr
        return variants

    # Select intelligently based on quality metrics
    if is_dark:
        variants["gamma_corrected"] = gamma_corrected
        variants["clahe"] = enhanced_clahe
        if quality["dark_ratio"] > 0.60:
            variants["adaptive_thresholded"] = adapt_bgr
            variants["binarized"] = adapt_bgr
    elif low_contrast:
        variants["clahe"] = enhanced_clahe
        variants["grayscale"] = gray_bgr
    else:
        # Standard well-lit image
        variants["clahe"] = enhanced_clahe

    return variants


def compute_overlay_iou(b1: list[int], b2: list[int]) -> tuple[float, float]:
    """Compute IoU and Containment ratio between two [x1, y1, x2, y2] boxes."""
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)
    inter = iw * ih
    a1 = max(1, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(1, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = a1 + a2 - inter
    iou = inter / float(union) if union > 0 else 0.0
    containment = inter / float(min(a1, a2))
    return iou, containment


def deduplicate_candidate_overlays(
    candidates: list[CandidateOverlay],
    iou_thresh: float = 0.35,
    containment_thresh: float = 0.60,
) -> list[CandidateOverlay]:
    """Deduplicate overlapping candidate overlay regions, retaining highest-confidence and most specific."""
    if not candidates:
        return []
    sorted_cands = sorted(candidates, key=lambda c: c.score, reverse=True)
    kept: list[CandidateOverlay] = []
    for cand in sorted_cands:
        suppressed = False
        for chosen in kept:
            iou, cont = compute_overlay_iou(cand.bbox_xyxy, chosen.bbox_xyxy)
            if iou > iou_thresh or cont > containment_thresh:
                suppressed = True
                break
        if not suppressed:
            kept.append(cand)
    return kept


class MultiFormatOverlayDetector:
    """
    Identifies candidate overlay regions across multiple visual formats:
    - Green saturated text
    - Yellow saturated text
    - White text on dark/black background
    - Dark text on light/grey bottom panel
    - Fallback full-frame / regional proposals
    """

    def __init__(self, upscale_factor: float = 2.0):
        self.upscale_factor = upscale_factor

    def enhance_crop(self, crop_bgr: np.ndarray) -> dict[str, np.ndarray]:
        """
        Produce controlled enhancement variants of a candidate crop:
        original, grayscale, clahe, gamma_corrected, adaptive_thresholded,
        plus upscaled if small. Selects variants intelligently based on quality.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            return {"original": crop_bgr}

        variants = generate_controlled_variants(crop_bgr, select_based_on_quality=True)

        h, w = crop_bgr.shape[:2]
        # Include upscaled variant if small text
        if self.upscale_factor > 1.0 and (w < 800 or h < 300):
            scale = self.upscale_factor
            target_w = int(w * scale)
            target_h = int(h * scale)
            upscaled = cv2.resize(crop_bgr, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
            variants["upscaled"] = upscaled

        return variants

    def filter_giant_regions(
        self,
        candidates: list[CandidateOverlay],
        img_w: int,
        img_h: int,
        max_area_ratio: float = 0.30,
    ) -> list[CandidateOverlay]:
        """
        Reject giant candidate regions that do not represent precise metadata crops.
        """
        valid: list[CandidateOverlay] = []
        total_area = float(max(1, img_w * img_h))

        for cand in candidates:
            bw = cand.bbox_xyxy[2] - cand.bbox_xyxy[0]
            bh = cand.bbox_xyxy[3] - cand.bbox_xyxy[1]
            box_area = bw * bh
            area_ratio = box_area / total_area

            # Do not submit giant candidate regions
            if area_ratio > max_area_ratio:
                logger.info(
                    f"Rejecting giant candidate overlay '{cand.format_hint}' {cand.bbox_xyxy} "
                    f"({area_ratio:.1%} of image area)."
                )
                continue
            valid.append(cand)
        return valid

    def _extract_bounding_box_from_mask(
        self,
        mask: np.ndarray,
        img_w: int,
        img_h: int,
        pad_px: int = 15,
        min_pixels: int = 50,
    ) -> list[int] | None:
        """Compute clamped bounding box enclosing non-zero pixels in mask."""
        coords = cv2.findNonZero(mask)
        if coords is None or len(coords) < min_pixels:
            return None
        gx, gy, gw, gh = cv2.boundingRect(coords)
        x1 = max(0, gx - pad_px)
        y1 = max(0, gy - pad_px)
        x2 = min(img_w, gx + gw + pad_px)
        y2 = min(img_h, gy + gh + pad_px)
        if (x2 - x1) < 20 or (y2 - y1) < 15:
            return None
        return [x1, y1, x2, y2]

    def detect_green_candidates(self, bgr_image: np.ndarray) -> list[CandidateOverlay]:
        """Candidate strategy 1: Green saturated text (DWS Green overlay)."""
        h, w = bgr_image.shape[:2]
        # Focus on top 50% for DWS overlay text
        roi_h = int(h * 0.55)
        top_half = bgr_image[:roi_h, :]

        mask = detect_green_overlay(top_half)
        if mask is None:
            return []

        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[:roi_h, :] = mask

        bbox = self._extract_bounding_box_from_mask(full_mask, w, h, pad_px=20, min_pixels=_MIN_GREEN_PIXELS)
        if not bbox:
            return []

        crop = bgr_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        count = cv2.countNonZero(mask)
        conf = min(0.95, 0.5 + (count / 5000.0) * 0.45)

        variants = self.enhance_crop(crop)
        return [CandidateOverlay("green_dws", bbox, conf, crop, variants)]

    def detect_yellow_candidates(self, bgr_image: np.ndarray) -> list[CandidateOverlay]:
        """Candidate strategy 2: Yellow saturated text (DWS Yellow overlay)."""
        h, w = bgr_image.shape[:2]
        roi_h = int(h * 0.55)
        top_half = bgr_image[:roi_h, :]

        mask = detect_yellow_overlay(top_half)
        if mask is None:
            return []

        full_mask = np.zeros((h, w), dtype=np.uint8)
        full_mask[:roi_h, :] = mask

        bbox = self._extract_bounding_box_from_mask(full_mask, w, h, pad_px=20, min_pixels=_MIN_YELLOW_PIXELS)
        if not bbox:
            return []

        crop = bgr_image[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        count = cv2.countNonZero(mask)
        conf = min(0.95, 0.5 + (count / 5000.0) * 0.45)

        variants = self.enhance_crop(crop)
        return [CandidateOverlay("yellow_dws", bbox, conf, crop, variants)]

    def detect_white_on_dark_candidates(self, bgr_image: np.ndarray) -> list[CandidateOverlay]:
        """
        Candidate strategy 3: White text on dark/black background.
        Covers:
        - Top-left white monospaced scanner overlay
        - Ekart small black information box with white text
        """
        h, w = bgr_image.shape[:2]
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        candidates: list[CandidateOverlay] = []

        # A. Top-left dark rectangle check (white_text_scanner)
        tl_w = int(w * 0.5)
        tl_h = int(h * 0.4)
        tl_gray = gray[:tl_h, :tl_w]

        # In a white-on-black overlay, the background is very dark (< 50) and text is bright (> 150)
        dark_pixels = np.sum(tl_gray < 50)
        bright_pixels = np.sum(tl_gray > 160)
        if dark_pixels > (tl_w * tl_h * 0.15) and bright_pixels > 200:
            # Contrast is present in top-left
            # Find bounds of high contrast text inside dark region
            edges = cv2.Canny(tl_gray, 50, 150)
            coords = cv2.findNonZero(edges)
            if coords is not None and len(coords) > 100:
                gx, gy, gw, gh = cv2.boundingRect(coords)
                x1 = max(0, gx - 15)
                y1 = max(0, gy - 15)
                x2 = min(w, gx + gw + 20)
                y2 = min(h, gy + gh + 20)
                if (x2 - x1) > 60 and (y2 - y1) > 30:
                    crop = bgr_image[y1:y2, x1:x2]
                    candidates.append(
                        CandidateOverlay(
                            "white_text_scanner",
                            [x1, y1, x2, y2],
                            0.75,
                            crop,
                            self.enhance_crop(crop),
                        )
                    )

        # B. Ekart-style small black box detection across image
        # Morphological search for compact dark rectangles
        _, dark_mask = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
        closed_dark = cv2.morphologyEx(dark_mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed_dark, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for cnt in contours:
            x, y, cw, ch = cv2.boundingRect(cnt)
            area = cw * ch
            # Ekart box is typically 80x40 to 600x300
            if 3000 < area < (w * h * 0.15) and (cw / max(1, ch)) > 1.2:
                sub_gray = gray[y:y+ch, x:x+cw]
                # Verify presence of bright text inside
                if np.sum(sub_gray > 160) > 80:
                    pad = 10
                    bx1 = max(0, x - pad)
                    by1 = max(0, y - pad)
                    bx2 = min(w, x + cw + pad)
                    by2 = min(h, y + ch + pad)
                    crop = bgr_image[by1:by2, bx1:bx2]
                    candidates.append(
                        CandidateOverlay(
                            "ekart",
                            [bx1, by1, bx2, by2],
                            0.70,
                            crop,
                            self.enhance_crop(crop),
                        )
                    )
                    if len(candidates) >= 3:
                        break

        return candidates

    def detect_bottom_panel_candidates(self, bgr_image: np.ndarray) -> list[CandidateOverlay]:
        """
        Candidate strategy 4: Grey/white information panel near bottom of the image.
        Typically contains: L, W, H, A.W, V.W, AWB.
        """
        h, w = bgr_image.shape[:2]
        # Look at the bottom 30% of the image
        bottom_y = int(h * 0.70)
        bot_bgr = bgr_image[bottom_y:, :]
        gray = cv2.cvtColor(bot_bgr, cv2.COLOR_BGR2GRAY)

        # Bottom panels are wide horizontal bands (grey or white) with dark text
        # Look for horizontal projection of text edges
        edges = cv2.Canny(gray, 40, 140)
        row_edges = np.sum(edges > 0, axis=1)

        # Find rows with text activity
        active_rows = np.where(row_edges > (w * 0.05))[0]
        if len(active_rows) > 10:
            min_r = int(active_rows[0])
            max_r = int(active_rows[-1])
            y1 = max(0, bottom_y + min_r - 15)
            y2 = min(h, bottom_y + max_r + 20)
            x1 = 0
            x2 = w
            if (y2 - y1) > 25:
                crop = bgr_image[y1:y2, x1:x2]
                return [
                    CandidateOverlay(
                        "bottom_panel",
                        [x1, y1, x2, y2],
                        0.72,
                        crop,
                        self.enhance_crop(crop),
                    )
                ]
        return []

    def detect_fallback_proposals(self, bgr_image: np.ndarray) -> list[CandidateOverlay]:
        """Region proposals / fallback when color segmentation yields no strong candidate."""
        h, w = bgr_image.shape[:2]
        proposals: list[CandidateOverlay] = []

        # Fallback 1: Standard top-left scanner overlay region
        tl_x2 = int(w * 0.45)
        tl_y2 = int(h * 0.35)
        tl_crop = bgr_image[0:tl_y2, 0:tl_x2]
        proposals.append(
            CandidateOverlay(
                "generic_overlay",
                [0, 0, tl_x2, tl_y2],
                0.35,
                tl_crop,
                self.enhance_crop(tl_crop),
            )
        )

        # Fallback 2: Standard bottom panel strip
        bot_y1 = int(h * 0.80)
        bot_crop = bgr_image[bot_y1:h, 0:w]
        proposals.append(
            CandidateOverlay(
                "bottom_panel",
                [0, bot_y1, w, h],
                0.30,
                bot_crop,
                self.enhance_crop(bot_crop),
            )
        )

        return proposals

    def get_candidate_regions(self, bgr_image: np.ndarray) -> list[CandidateOverlay]:
        """
        Run all candidate-region strategies, filter giant regions, deduplicate
        overlapping regions, and return clean candidate regions ordered by priority.
        """
        h, w = bgr_image.shape[:2]
        raw_candidates: list[CandidateOverlay] = []

        # 1. Color saturated text (Green then Yellow)
        green = self.detect_green_candidates(bgr_image)
        raw_candidates.extend(green)

        yellow = self.detect_yellow_candidates(bgr_image)
        raw_candidates.extend(yellow)

        # 2. White text on dark background
        white_dark = self.detect_white_on_dark_candidates(bgr_image)
        raw_candidates.extend(white_dark)

        # 3. Bottom panel
        bottom = self.detect_bottom_panel_candidates(bgr_image)
        raw_candidates.extend(bottom)

        # Filter out giant vague regions
        filtered = self.filter_giant_regions(raw_candidates, w, h, max_area_ratio=0.30)

        # 4. If no high-confidence candidates found, include fallback proposals only if edge activity
        if not filtered:
            fallbacks = self.detect_fallback_proposals(bgr_image)
            filtered = self.filter_giant_regions(fallbacks, w, h, max_area_ratio=0.30)

        # Deduplicate overlapping candidate overlay regions
        deduped = deduplicate_candidate_overlays(filtered, iou_thresh=0.35, containment_thresh=0.60)

        # Sort by candidate score descending
        deduped.sort(key=lambda c: c.score, reverse=True)
        return deduped


# ---------- Backward Compatible Line Segmentation and OCR Helpers ----------

def _segment_overlay_lines(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Use horizontal-projection profile to find text line bands."""
    row_sums = np.sum(mask > 0, axis=1)
    in_line = False
    bands: list[tuple[int, int]] = []
    line_start = 0

    for r in range(len(row_sums)):
        if row_sums[r] > 3:
            if not in_line:
                line_start = r
                in_line = True
        else:
            if in_line:
                bands.append((line_start, r))
                in_line = False
    if in_line:
        bands.append((line_start, len(row_sums)))

    merged: list[tuple[int, int]] = []
    for start, end in bands:
        if merged and start - merged[-1][1] < 5:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))

    results: list[tuple[int, int, int, int]] = []
    for rs, re_ in merged:
        band = mask[rs:re_, :]
        col_has = np.any(band > 0, axis=0)
        cols = np.where(col_has)[0]
        if len(cols) < 3:
            continue
        cs, ce = int(cols[0]), int(cols[-1]) + 1
        if (ce - cs) < 15 or (re_ - rs) < 4:
            continue
        results.append((rs, re_, cs, ce))

    return results


def extract_overlay_line_images(bgr_image: np.ndarray, mask: np.ndarray) -> list[np.ndarray]:
    """Crop each overlay text line as a clean binary image."""
    coords = cv2.findNonZero(mask)
    if coords is None:
        return []

    gx, gy, gw, gh = cv2.boundingRect(coords)
    h, w = bgr_image.shape[:2]
    pad = 5
    oy1 = max(0, gy - pad)
    oy2 = min(h, gy + gh + pad)
    ox1 = max(0, gx - pad)
    ox2 = min(w, gx + gw + pad)
    crop_mask = mask[oy1:oy2, ox1:ox2]

    line_bands = _segment_overlay_lines(crop_mask)

    images: list[np.ndarray] = []
    for rs, re_, cs, ce in line_bands:
        lpad = 4
        r0 = max(0, rs - lpad)
        r1 = min(crop_mask.shape[0], re_ + lpad)
        c0 = max(0, cs - lpad)
        c1 = min(crop_mask.shape[1], ce + lpad)

        line_mask = crop_mask[r0:r1, c0:c1]
        binary = cv2.bitwise_not(line_mask)

        target_h = 48
        cur_h = binary.shape[0]
        if cur_h < target_h:
            scale = target_h / float(cur_h)
            binary = cv2.resize(
                binary, None, fx=scale, fy=scale,
                interpolation=cv2.INTER_NEAREST,
            )

        images.append(binary)

    return images


def clean_ocr_line(line: str) -> str:
    """Normalize common OCR misrecognitions on fixed-width scanner text."""
    line = re.sub(r"(?<=\d)[-_](?=\d)", ".", line)
    line = re.sub(r"(?<=\d)\s*9m\b", " gm", line, flags=re.IGNORECASE)
    line = re.sub(r"(?<=\d)/(?=\.)", "7", line)
    return line


def ocr_overlay_lines(line_images: list[np.ndarray], ocr_engine) -> list[str]:
    """Run single-line OCR on each line image and return text strings."""
    texts: list[str] = []
    for img in line_images:
        try:
            result = ocr_engine.extract_text(img)
            line_text = result.full_text.strip() if result.full_text else ""
            cleaned = clean_ocr_line(line_text)
            texts.append(cleaned)
        except Exception as e:
            logger.debug(f"OCR failed on overlay line: {e}")
            texts.append("")
    return texts


def parse_overlay_fields(lines: list[str]) -> OverlayData:
    """Parse structured key-value fields from overlay text lines."""
    data = OverlayData()
    cleaned_lines = [clean_ocr_line(l) for l in lines if l]
    data.raw_lines = cleaned_lines
    full_text = "\n".join(cleaned_lines)

    matched_fields = 0

    m = OVERLAY_FIELD_PATTERNS["location"].search(full_text)
    if m:
        data.location = m.group(1).strip()
        matched_fields += 1

    m = OVERLAY_FIELD_PATTERNS["awb_no"].search(full_text)
    if m:
        raw = m.group(1).strip()
        clean = re.sub(r"[^A-Za-z0-9]", "", raw)
        if len(clean) >= 8:
            data.awb_no = clean
            matched_fields += 1

    dim_unit = "cm"
    for field in ("length", "width", "height"):
        m = OVERLAY_FIELD_PATTERNS[field].search(full_text)
        if m:
            try:
                val = float(m.group(1))
                setattr(data, field, val)
                if m.group(2):
                    dim_unit = m.group(2).lower()
                matched_fields += 1
            except (ValueError, TypeError):
                pass
    data.dimension_unit = dim_unit

    m = OVERLAY_FIELD_PATTERNS["volume"].search(full_text)
    if not m:
        m = re.search(r"R\.?\s*VOL\b[^0-9]*([\d]+(?:\.[\d]+)?)", full_text, re.IGNORECASE)
    if m:
        try:
            data.volume = float(m.group(1))
        except (ValueError, TypeError):
            pass

    m = OVERLAY_FIELD_PATTERNS["weight"].search(full_text)
    if m:
        try:
            val = float(m.group(1))
            raw_unit = (m.group(2) or "gm").lower()
            if raw_unit in ("gm", "gms", "g", "9m", "9ms", "grams"):
                data.weight_unit = "g"
                data.weight_value = val
            elif raw_unit in ("kg", "kgs"):
                data.weight_unit = "kg"
                data.weight_value = val
            elif raw_unit in ("lb", "lbs"):
                data.weight_unit = "lb"
                data.weight_value = val
            else:
                data.weight_unit = raw_unit
                data.weight_value = val
            matched_fields += 1
        except (ValueError, TypeError):
            pass

    m = OVERLAY_FIELD_PATTERNS["time"].search(full_text)
    if m:
        data.timestamp = m.group(1).strip()

    data.confidence = round(min(1.0, matched_fields / 6.0), 2)
    return data


def build_ocr_result_from_overlay(overlay_data: OverlayData) -> OcrResult:
    """Synthesize an OcrResult from overlay data."""
    parts: list[str] = []
    if overlay_data.awb_no:
        parts.append(f"AWB No: {overlay_data.awb_no}")
    if overlay_data.weight_value is not None:
        parts.append(f"Weight: {overlay_data.weight_value} {overlay_data.weight_unit or 'g'}")
    if (
        overlay_data.length is not None
        and overlay_data.width is not None
        and overlay_data.height is not None
    ):
        u = overlay_data.dimension_unit or "cm"
        parts.append(f"Length: {overlay_data.length} {u}")
        parts.append(f"Width: {overlay_data.width} {u}")
        parts.append(f"Height: {overlay_data.height} {u}")
        parts.append(
            f"DIMS: {overlay_data.length} x {overlay_data.width} x {overlay_data.height} {u}"
        )
    if overlay_data.volume is not None:
        parts.append(f"R.Vol.: {overlay_data.volume} cm (cubic)")
    if overlay_data.location:
        parts.append(f"Location: {overlay_data.location}")
    if overlay_data.timestamp:
        parts.append(f"Time: {overlay_data.timestamp}")

    full_text = "\n".join(parts)

    line_box = OcrTextBox(
        text=full_text,
        score=overlay_data.confidence,
        box=[[0, 0], [1, 0], [1, 1], [0, 1]],
        source_crop="scanner_overlay",
        preprocessing_variant="segmented_lines",
    )

    return OcrResult(
        engine="scanner-overlay-segmentation-v1",
        lines=[line_box] if full_text else [],
        full_text=full_text,
        source_crop="scanner_overlay",
        preprocessing_variant="segmented_lines",
    )


overlay_detector = MultiFormatOverlayDetector(upscale_factor=settings.OVERLAY_UPSCALE_FACTOR)
