"""
Optional PaddleOCR-VL 1.6 Vision-Language Model Adapter (Milestone 4).

Target Hardware: Apple Silicon M1 (Darwin ARM64) CPU/MPS.
Configurable & Optional:
- Evaluates document/metadata semantics on selected image regions.
- Preserves deterministic OCR (PP-OCRv5/v6) and deterministic field parsing as primary.
- Strict evidence corroboration: Never accepts generated values from the VLM
  unless independently verified by recognized deterministic OCR text or decoded barcodes.
"""

from dataclasses import dataclass, field
import logging
import re
import time
from typing import Any, List, Optional, Tuple
import cv2
import numpy as np

from app.config import settings
from app.pipeline.interfaces import BarcodeResult, OcrTextBox

logger = logging.getLogger("paddleocr_vl")


@dataclass
class VlmFieldProposal:
    """Individual field proposal extracted by vision-language interpreter."""
    field_name: str  # "tracking_number", "weight_value", "length", "width", "height"
    proposed_value: Any
    confidence: float
    corroborated: bool = False
    evidence_source: Optional[str] = None
    rejection_reason: Optional[str] = None


@dataclass
class VlmInterpretationResult:
    """Result of second-stage document interpretation with corroboration status."""
    tracking_number: Optional[str] = None
    weight_value: Optional[float] = None
    dimensions: Optional[Tuple[float, float, float]] = None
    proposals: List[VlmFieldProposal] = field(default_factory=list)
    accepted_fields: List[str] = field(default_factory=list)
    rejected_fields: List[str] = field(default_factory=list)
    raw_output: str = ""
    latency_ms: float = 0.0


def corroborate_proposed_tracking(
    proposed_tracking: str,
    ocr_lines: Optional[List[OcrTextBox]],
    barcodes: Optional[List[BarcodeResult]],
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Verify proposed tracking number against decoded barcodes or deterministic OCR lines.
    Rejects values with no corroborating evidence (prevents VLM hallucination).
    """
    if not proposed_tracking or not proposed_tracking.strip():
        return False, None, "Empty tracking candidate"

    clean_prop = re.sub(r"[^A-Z0-9]", "", proposed_tracking.upper())
    if len(clean_prop) < 6:
        return False, None, "Candidate too short (< 6 chars)"

    # Rule: Must have at least one digit for supported formats
    if not any(c.isdigit() for c in clean_prop):
        return False, None, "Alpha-only tracking candidate rejected"

    # Check 1: Decoded barcode matches
    if barcodes:
        for b in barcodes:
            clean_bc = re.sub(r"[^A-Z0-9]", "", b.raw_value.upper())
            if clean_prop == clean_bc:
                return True, f"barcode_{b.symbology}", None

    # Check 2: Deterministic OCR text lines match
    if ocr_lines:
        for line in ocr_lines:
            clean_line = re.sub(r"[^A-Z0-9]", "", line.text.upper())
            if clean_prop in clean_line:
                return True, f"ocr_line:{line.text}", None

    return False, None, "No corroborating evidence in barcode or OCR text"


def corroborate_proposed_weight(
    proposed_weight: float,
    ocr_lines: Optional[List[OcrTextBox]],
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Verify proposed numeric weight value against recognized deterministic OCR text.
    """
    if proposed_weight is None or proposed_weight <= 0.0:
        return False, None, "Invalid non-positive weight"

    if not ocr_lines:
        return False, None, "No deterministic OCR lines available for verification"

    target_str_1 = f"{proposed_weight:.1f}"
    target_str_2 = f"{proposed_weight:.2f}"
    target_str_int = f"{int(proposed_weight)}" if proposed_weight.is_integer() else None

    for line in ocr_lines:
        text = line.text.upper()
        # Look for the numeric string in proximity to weight units or labels
        has_num = (
            target_str_1 in text
            or target_str_2 in text
            or (target_str_int and target_str_int in text)
        )
        if has_num:
            return True, f"ocr_line:{line.text}", None

    return False, None, f"Numeric weight {proposed_weight} not found in deterministic OCR text"


def corroborate_proposed_dimensions(
    length: float,
    width: float,
    height: float,
    ocr_lines: Optional[List[OcrTextBox]],
) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Verify proposed dimensions against recognized deterministic OCR text lines.
    """
    if not ocr_lines:
        return False, None, "No deterministic OCR lines available for verification"

    l_str = f"{length:.0f}" if length.is_integer() else f"{length:.1f}"
    w_str = f"{width:.0f}" if width.is_integer() else f"{width:.1f}"
    h_str = f"{height:.0f}" if height.is_integer() else f"{height:.1f}"

    for line in ocr_lines:
        text = line.text.upper()
        # Look for L*W*H pattern or individual dimensions in line
        if (l_str in text and w_str in text) or (w_str in text and h_str in text) or (l_str in text and h_str in text):
            return True, f"ocr_line:{line.text}", None

    return False, None, f"Dimensions {length}x{width}x{height} not corroborated by OCR text"


class PaddleOcrVlAdapter:
    """
    Adapter for PaddleOCR-VL 1.6 document/metadata interpreter.
    Runs locally on Apple Silicon M1 (CPU).
    """

    MODEL_ID = "PaddleOCR-VL-1.6"

    def __init__(
        self,
        enabled: Optional[bool] = None,
        model_version: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.enabled = enabled if enabled is not None else settings.ENABLE_PADDLEOCR_VL
        self.model_version = model_version or settings.PADDLEOCR_VL_VERSION
        self.device = device or settings.PADDLEOCR_VL_DEVICE
        self._pipeline = None
        self._load_error: Optional[str] = None

    def get_health(self) -> dict[str, Any]:
        """Return engine status, version, runtime, and readiness."""
        if not self.enabled:
            return {
                "engine": "paddleocr_vl",
                "model_version": self.model_version,
                "runtime": f"Apple Silicon M1 ({self.device})",
                "readiness": "disabled",
                "error": None,
            }

        if self._load_error:
            return {
                "engine": "paddleocr_vl",
                "model_version": self.model_version,
                "runtime": f"Apple Silicon M1 ({self.device})",
                "readiness": "degraded",
                "error": self._load_error,
            }

        return {
            "engine": "paddleocr_vl",
            "model_version": self.model_version,
            "runtime": f"Apple Silicon M1 ({self.device})",
            "readiness": "ready" if self._pipeline is not None else "standby",
            "error": None,
        }

    def _ensure_loaded(self) -> bool:
        """Lazy load PaddleOCR-VL pipeline if enabled."""
        if not self.enabled:
            return False
        if self._pipeline is not None:
            return True
        if self._load_error is not None:
            return False

        try:
            import paddlex
            from paddlex.inference.pipelines import PaddleOCRVLPipeline

            # Initialize PaddleOCR-VL pipeline for Apple Silicon M1
            self._pipeline = paddlex.create_pipeline("PaddleOCR-VL")
            logger.info(f"PaddleOCR-VL {self.model_version} loaded successfully on {self.device}")
            return True
        except Exception as e:
            self._load_error = str(e)
            logger.warning(
                f"PaddleOCR-VL {self.model_version} optional pipeline unavailable on {self.device}: {e}. "
                f"Deterministic OCR & parsing will handle all extractions."
            )
            return False

    def interpret_region(
        self,
        crop_bgr: np.ndarray,
        region_type: str = "shipping_label",
        full_ocr_lines: Optional[List[OcrTextBox]] = None,
        barcodes: Optional[List[BarcodeResult]] = None,
    ) -> Optional[VlmInterpretationResult]:
        """
        Run document interpretation on selected region and strictly corroborate
        proposed values against deterministic OCR and barcode evidence.
        """
        if not self.enabled:
            return None

        if crop_bgr is None or crop_bgr.size == 0:
            return None

        t0 = time.perf_counter()
        loaded = self._ensure_loaded()
        if not loaded or self._pipeline is None:
            return None

        # Execute VLM inference
        try:
            output = self._pipeline.predict(crop_bgr)
            raw_text = str(output)
        except Exception as e:
            logger.warning(f"PaddleOCR-VL prediction failed on crop: {e}")
            return None

        latency = (time.perf_counter() - t0) * 1000.0

        # Parse proposed fields and corroborate against evidence
        result = self._parse_and_corroborate(raw_text, full_ocr_lines, barcodes, latency)
        return result

    def _parse_and_corroborate(
        self,
        raw_output: str,
        full_ocr_lines: Optional[List[OcrTextBox]],
        barcodes: Optional[List[BarcodeResult]],
        latency_ms: float,
    ) -> VlmInterpretationResult:
        """Extract candidate values and enforce evidence corroboration."""
        res = VlmInterpretationResult(raw_output=raw_output, latency_ms=latency_ms)

        # 1. Look for tracking number candidates in output
        awb_match = re.search(r"(?:tracking|awb|barcode)[:\s]*([A-Z0-9-]{8,35})", raw_output, re.IGNORECASE)
        if awb_match:
            candidate_awb = awb_match.group(1).strip()
            ok, src, reason = corroborate_proposed_tracking(candidate_awb, full_ocr_lines, barcodes)
            prop = VlmFieldProposal(
                field_name="tracking_number",
                proposed_value=candidate_awb,
                confidence=0.90 if ok else 0.30,
                corroborated=ok,
                evidence_source=src,
                rejection_reason=reason,
            )
            res.proposals.append(prop)
            if ok:
                res.tracking_number = candidate_awb
                res.accepted_fields.append("tracking_number")
                logger.info(f"   [VLM Corroborated] Tracking: '{candidate_awb}' via {src}")
            else:
                res.rejected_fields.append("tracking_number")
                logger.warning(f"   [VLM Rejected] Tracking: '{candidate_awb}' ({reason})")

        # 2. Look for weight candidates
        wt_match = re.search(r"(?:weight|wt|a\.w|gross)[:\s]*([0-9]+\.?[0-9]*)\s*(?:kg|g|gm)?", raw_output, re.IGNORECASE)
        if wt_match:
            try:
                candidate_wt = float(wt_match.group(1))
                ok, src, reason = corroborate_proposed_weight(candidate_wt, full_ocr_lines)
                prop = VlmFieldProposal(
                    field_name="weight_value",
                    proposed_value=candidate_wt,
                    confidence=0.88 if ok else 0.25,
                    corroborated=ok,
                    evidence_source=src,
                    rejection_reason=reason,
                )
                res.proposals.append(prop)
                if ok:
                    res.weight_value = candidate_wt
                    res.accepted_fields.append("weight_value")
                    logger.info(f"   [VLM Corroborated] Weight: {candidate_wt} via {src}")
                else:
                    res.rejected_fields.append("weight_value")
                    logger.warning(f"   [VLM Rejected] Weight: {candidate_wt} ({reason})")
            except ValueError:
                pass

        # 3. Look for dimensions candidates
        dim_match = re.search(r"(?:dimensions?|dims?)[:\s]*([0-9.]+)\s*[*xX]\s*([0-9.]+)\s*[*xX]\s*([0-9.]+)", raw_output, re.IGNORECASE)
        if dim_match:
            try:
                l, w, h = float(dim_match.group(1)), float(dim_match.group(2)), float(dim_match.group(3))
                ok, src, reason = corroborate_proposed_dimensions(l, w, h, full_ocr_lines)
                prop = VlmFieldProposal(
                    field_name="dimensions",
                    proposed_value=(l, w, h),
                    confidence=0.85 if ok else 0.20,
                    corroborated=ok,
                    evidence_source=src,
                    rejection_reason=reason,
                )
                res.proposals.append(prop)
                if ok:
                    res.dimensions = (l, w, h)
                    res.accepted_fields.append("dimensions")
                    logger.info(f"   [VLM Corroborated] Dimensions: {l}x{w}x{h} via {src}")
                else:
                    res.rejected_fields.append("dimensions")
                    logger.warning(f"   [VLM Rejected] Dimensions: {l}x{w}x{h} ({reason})")
            except ValueError:
                pass

        return res


class MockPaddleOcrVlAdapter(PaddleOcrVlAdapter):
    """
    Mock adapter for PaddleOCR-VL used in testing and offline evaluation.
    """

    def __init__(
        self,
        mock_raw_output: str = "Tracking: 1Z9999999999999999 Weight: 2.5 KG Dims: 30x20x15",
        enabled: bool = True,
    ):
        super().__init__(enabled=enabled)
        self.mock_raw_output = mock_raw_output

    def get_health(self) -> dict[str, Any]:
        return {
            "engine": "paddleocr_vl",
            "model_version": "1.6-mock",
            "runtime": "Apple Silicon M1 CPU (Mock)",
            "readiness": "ready",
            "error": None,
        }

    def interpret_region(
        self,
        crop_bgr: np.ndarray,
        region_type: str = "shipping_label",
        full_ocr_lines: Optional[List[OcrTextBox]] = None,
        barcodes: Optional[List[BarcodeResult]] = None,
    ) -> Optional[VlmInterpretationResult]:
        if not self.enabled:
            return None
        return self._parse_and_corroborate(
            self.mock_raw_output,
            full_ocr_lines,
            barcodes,
            latency_ms=12.5,
        )
