"""Format-specific parser registry for DWS and scanner metadata overlays.

Supported Profiles:
1. green_dws (top-left green DWS text)
2. yellow_dws (top-left yellow DWS text)
3. white_text_scanner (top-left white monospaced text on black)
4. ekart (small black box with white text)
5. bottom_panel (grey/white panel near bottom with L, W, H, A.W, V.W, AWB)
6. generic_overlay (carrier-agnostic fallback)

Rules:
- Arbitrary dimension label order (e.g. L0214 X H0061 X W0154)
- Ordered 3-number dimension triplets without labels (0208 X 0148 X 0163)
- Never guess units from numeric magnitude; use verified scanner profile or explicit unit
- Separate A.W (actual weight) and V.W (volumetric weight)
"""

import logging
import re
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from pydantic import BaseModel, Field

logger = logging.getLogger("parser_registry")


def is_valid_tracking(cand: str | None) -> bool:
    """
    Validate tracking number candidate according to Milestone 6 rules:
    - Minimum 8 chars, maximum 35 chars.
    - Reject alphabetic-only tracking values (e.g. INVENTIMENT).
    - Require at least one digit for supported formats.
    - Support alphanumeric IDs like PROC0000316077, 1Z..., FMPC..., etc.
    """
    if not cand:
        return False
    clean = re.sub(r"[^A-Za-z0-9]", "", cand)
    if len(clean) < 8 or len(clean) > 35:
        return False
    # Reject alphabetic-only tracking values
    if not any(c.isdigit() for c in clean):
        return False
    return True


class ParsedOverlayMetadata(BaseModel):
    """Structured extraction output from a format profile."""

    profile_name: str
    tracking_number: str | None = None
    raw_tracking_text: str = ""

    length: float | None = None
    width: float | None = None
    height: float | None = None
    dimension_unit: str | None = None
    raw_dimension_text: str = ""

    # Separate actual weight and volumetric weight
    weight_value: float | None = None
    weight_unit: str | None = None
    actual_weight: float | None = None
    actual_weight_unit: str | None = None
    volumetric_weight: float | None = None
    volumetric_weight_unit: str | None = None
    raw_weight_text: str = ""

    volume: float | None = None
    raw_volume_text: str = ""

    location: str | None = None
    timestamp: str | None = None
    machine_id: str | None = None
    sorter_id: str | None = None

    raw_lines: list[str] = Field(default_factory=list)
    confidence: float = 0.0
    field_rules: dict[str, str] = Field(default_factory=dict)

    def is_populated(self) -> bool:
        return (
            self.tracking_number is not None
            or self.weight_value is not None
            or self.actual_weight is not None
            or (self.length is not None and self.width is not None)
        )


# ============================================================================
# Dimension and Weight Parsing Helpers
# ============================================================================

def parse_arbitrary_labelled_dimensions(text: str) -> tuple[float, float, float, str | None, str] | None:
    """
    Parse dimensions with arbitrary label order, e.g.:
    - 'Dimension: L0214 X H0061 X W0154'
    - 'Dims: L: 30 H: 10 W: 20 cm'
    Returns (length, width, height, unit, raw_matched_str)
    """
    # Pattern for L..., W..., H... in any order separated by X, *, comma or space
    # e.g., L0214 X H0061 X W0154
    labeled_order_pattern = re.compile(
        r"(?:DIMENSIONS?|DIMS?)?[:\s]*([LWH])[:\s]*(\d+(?:\.\d+)?)\s*(?:[xX*×\s,]+)\s*([LWH])[:\s]*(\d+(?:\.\d+)?)\s*(?:[xX*×\s,]+)\s*([LWH])[:\s]*(\d+(?:\.\d+)?)\s*(CM|MM|IN|M)?\b",
        re.IGNORECASE,
    )
    m = labeled_order_pattern.search(text)
    if m:
        labels = [m.group(1).upper(), m.group(3).upper(), m.group(5).upper()]
        values = [float(m.group(2)), float(m.group(4)), float(m.group(6))]
        raw_unit = m.group(7).lower() if m.group(7) else None

        dims = {}
        for lbl, val in zip(labels, values):
            if lbl == "L":
                dims["length"] = val
            elif lbl == "W":
                dims["width"] = val
            elif lbl == "H":
                dims["height"] = val

        if "length" in dims and "width" in dims and "height" in dims:
            return dims["length"], dims["width"], dims["height"], raw_unit, m.group(0).strip()

    return None


def parse_discrete_dimensions(text: str) -> tuple[float, float, float, str | None, str] | None:
    """
    Parse discrete dimension lines:
    'Length: 75.300 cm'
    'Width: 49.200 cm'
    'Height: 44.400 cm'
    """
    m_l = re.search(r"\bLength[:\s.]+([\d]+(?:\.[\d]+)?)\s*(cm|mm|m|in)?\b", text, re.IGNORECASE)
    m_w = re.search(r"\bWidth[:\s.]+([\d]+(?:\.[\d]+)?)\s*(cm|mm|m|in)?\b", text, re.IGNORECASE)
    m_h = re.search(r"\bHeight[:\s.]+([\d]+(?:\.[\d]+)?)\s*(cm|mm|m|in)?\b", text, re.IGNORECASE)

    if m_l and m_w and m_h:
        try:
            l = float(m_l.group(1))
            w = float(m_w.group(1))
            h = float(m_h.group(1))
            unit = m_l.group(2).lower() if m_l.group(2) else None
            raw = f"{m_l.group(0)}; {m_w.group(0)}; {m_h.group(0)}"
            return l, w, h, unit, raw
        except (ValueError, TypeError):
            pass
    return None


def parse_triplet_dimensions(
    text: str,
    default_order: tuple[str, str, str] = ("length", "width", "height"),
    profile_unit: str | None = None,
) -> tuple[float, float, float, str | None, str] | None:
    """
    Parse three-number dimension triplets without L/W/H labels:
    - 'Dimension: 0208 X 0148 X 0163'
    - 'L * W * H: 0330 * 0369 * 0040'
    - '30 x 20 x 15 cm'
    """
    triplet_pattern = re.compile(
        r"(?:DIMENSIONS?|DIMS?|L\s*\*\s*W\s*\*\s*H|MEAS)?[:\s]*(\d+(?:\.\d+)?)\s*(?:[xX*×])\s*(\d+(?:\.\d+)?)\s*(?:[xX*×])\s*(\d+(?:\.\d+)?)\s*(CM|MM|IN|M)?\b",
        re.IGNORECASE,
    )
    for m in triplet_pattern.finditer(text):
        try:
            v1 = float(m.group(1))
            v2 = float(m.group(2))
            v3 = float(m.group(3))
            explicit_unit = m.group(4).lower() if m.group(4) else None
            unit = explicit_unit or profile_unit  # Never guess; only use explicit or verified profile unit

            dims = {default_order[0]: v1, default_order[1]: v2, default_order[2]: v3}
            return dims["length"], dims["width"], dims["height"], unit, m.group(0).strip()
        except (ValueError, TypeError, KeyError):
            continue

    return None


def parse_weight_with_rules(
    text: str,
    profile_unit: str | None = None,
) -> tuple[float | None, str | None, float | None, str | None, float | None, str | None, str]:
    """
    Parse weight fields distinguishing Actual Weight (A.W) and Volumetric Weight (V.W).
    Returns (actual_val, actual_unit, vol_val, vol_unit, primary_val, primary_unit, raw_text).
    """
    act_val = None
    act_unit = None
    vol_val = None
    vol_unit = None
    raw_parts = []

    # 1. Check explicit Actual Weight (A.W or Actual Weight)
    m_aw = re.search(r"(?:A\.W|ACTUAL\s*WT|ACTUAL\s*WEIGHT)[:\s]*([\d]+(?:\.[\d]+)?)\s*(KG|KGS|G|GM|GMS|9M|LB|LBS)?\b", text, re.IGNORECASE)
    if m_aw:
        try:
            act_val = float(m_aw.group(1))
            u = m_aw.group(2)
            act_unit = normalize_weight_unit(u) if u else profile_unit
            raw_parts.append(m_aw.group(0).strip())
        except ValueError:
            pass

    # 2. Check explicit Volumetric Weight (V.W or Vol Weight)
    m_vw = re.search(r"(?:V\.W|VOL\.?\s*WT|VOLUMETRIC\s*WEIGHT)[:\s]*([\d]+(?:\.[\d]+)?)\s*(KG|KGS|G|GM|GMS|9M|LB|LBS)?\b", text, re.IGNORECASE)
    if m_vw:
        try:
            vol_val = float(m_vw.group(1))
            u = m_vw.group(2)
            vol_unit = normalize_weight_unit(u) if u else profile_unit
            raw_parts.append(m_vw.group(0).strip())
        except ValueError:
            pass

    # 3. Check generic Weight if A.W not specified
    prim_val = act_val
    prim_unit = act_unit
    if prim_val is None:
        m_w = re.search(r"(?:WEIGHT|WT|GROSS)[:\s]*([\d]+(?:\.[\d]+)?)\s*(KG|KGS|G|GM|GMS|9M|GRAMS?|LB|LBS|OZS?)?\b", text, re.IGNORECASE)
        if m_w:
            try:
                prim_val = float(m_w.group(1))
                u = m_w.group(2)
                prim_unit = normalize_weight_unit(u) if u else profile_unit
                raw_parts.append(m_w.group(0).strip())
            except ValueError:
                pass

    return act_val, act_unit, vol_val, vol_unit, prim_val, prim_unit, "; ".join(raw_parts)


def normalize_weight_unit(raw_unit: str | None) -> str | None:
    """Normalize weight unit string."""
    if not raw_unit:
        return None
    u = raw_unit.lower().strip()
    if u in ("kg", "kgs", "kilogram", "kilograms"):
        return "kg"
    elif u in ("g", "gm", "gms", "9m", "9ms", "gram", "grams"):
        return "g"
    elif u in ("lb", "lbs", "pound", "pounds"):
        return "lb"
    elif u in ("oz", "ozs", "ounce", "ounces"):
        return "oz"
    return u


# ============================================================================
# Base Overlay Profile Interface
# ============================================================================

class BaseOverlayProfile(ABC):
    """Abstract base class for format-specific overlay parsing profiles."""

    name: str = "base"
    verified_dim_unit: str | None = None
    verified_weight_unit: str | None = None
    triplet_order: tuple[str, str, str] = ("length", "width", "height")

    @abstractmethod
    def can_handle(self, text: str, format_hint: str | None = None) -> float:
        """Score 0.0 - 1.0 indicating confidence this text belongs to this profile."""

    @abstractmethod
    def parse(self, text: str, lines: list[str]) -> ParsedOverlayMetadata:
        """Parse text lines into structured metadata."""


# ============================================================================
# Profile 1: Green DWS Overlay
# ============================================================================

class GreenDwsProfile(BaseOverlayProfile):
    """
    Top-left green text:
    Location, AWB, dimensions, real volume, weight, timestamp, machine ID.
    Verified units: cm, g (or kg when declared).
    """

    name = "green_dws"
    verified_dim_unit = "cm"
    verified_weight_unit = "g"
    triplet_order = ("length", "width", "height")

    def can_handle(self, text: str, format_hint: str | None = None) -> float:
        score = 0.0
        if re.search(r"Machine\s*ID|MachineNo", text, re.IGNORECASE):
            score += 0.35
        if re.search(r"Real\s*Vol|R\.?\s*Vol", text, re.IGNORECASE):
            score += 0.30
        if re.search(r"AWB(?:\s*(?:No|Number))?", text, re.IGNORECASE):
            score += 0.20
        if format_hint == "green_dws":
            score += 0.15
        return min(1.0, score)

    def parse(self, text: str, lines: list[str]) -> ParsedOverlayMetadata:
        meta = ParsedOverlayMetadata(profile_name=self.name, raw_lines=lines)
        matched = 0

        # Location
        m = re.search(r"Location[:\s.]+([^\n\r]+)", text, re.IGNORECASE)
        if m:
            meta.location = m.group(1).strip()
            meta.field_rules["location"] = "green_dws.location_kv"
            matched += 1

        # AWB
        m = re.search(r"AWB(?:\s*(?:No|Number))?[:\s.]+([A-Z0-9-]{8,35})", text, re.IGNORECASE)
        if m:
            cand = re.sub(r"[^A-Za-z0-9]", "", m.group(1))
            if is_valid_tracking(cand):
                meta.tracking_number = cand
                meta.raw_tracking_text = m.group(0).strip()
                meta.field_rules["tracking_number"] = "green_dws.awb_kv"
                matched += 1

        # Dimensions: try arbitrary labeled first, then discrete, then triplet
        dim_res = (
            parse_arbitrary_labelled_dimensions(text)
            or parse_discrete_dimensions(text)
            or parse_triplet_dimensions(text, self.triplet_order, self.verified_dim_unit)
        )
        if dim_res:
            l, w, h, u, raw = dim_res
            meta.length = l
            meta.width = w
            meta.height = h
            meta.dimension_unit = u or self.verified_dim_unit
            meta.raw_dimension_text = raw
            meta.field_rules["dimensions"] = "green_dws.dimension"
            matched += 1

        # Real Volume
        m = re.search(r"(?:Real\s*Vol|R\.?\s*Vol|Volume)[:\s.]+([\d]+(?:\.[\d]+)?)", text, re.IGNORECASE)
        if m:
            try:
                meta.volume = float(m.group(1))
                meta.raw_volume_text = m.group(0).strip()
                meta.field_rules["volume"] = "green_dws.volume"
            except ValueError:
                pass

        # Weight
        act_val, act_u, vol_val, vol_u, p_val, p_u, raw_w = parse_weight_with_rules(text, self.verified_weight_unit)
        if p_val is not None:
            meta.weight_value = p_val
            meta.weight_unit = p_u
            meta.raw_weight_text = raw_w
            meta.field_rules["weight"] = "green_dws.weight"
            matched += 1

        # Timestamp
        m = re.search(r"(?:Time|Timestamp)[:\s.]+([^\n\r]+)", text, re.IGNORECASE)
        if m:
            meta.timestamp = m.group(1).strip()
            meta.field_rules["timestamp"] = "green_dws.timestamp"

        # Machine ID
        m = re.search(r"(?:Machine\s*ID|MachineNo)[:\s.]+([^\n\r]+)", text, re.IGNORECASE)
        if m:
            meta.machine_id = m.group(1).strip()
            meta.field_rules["machine_id"] = "green_dws.machine_id"

        meta.confidence = round(min(1.0, matched / 5.0), 2)
        return meta


# ============================================================================
# Profile 2: Yellow DWS Overlay
# ============================================================================

class YellowDwsProfile(BaseOverlayProfile):
    """
    Top-left yellow text:
    Location, AWB, length, width, height, volume, weight, timestamp.
    Verified units: cm, g.
    """

    name = "yellow_dws"
    verified_dim_unit = "cm"
    verified_weight_unit = "g"
    triplet_order = ("length", "width", "height")

    def can_handle(self, text: str, format_hint: str | None = None) -> float:
        score = 0.0
        if re.search(r"R\.?\s*Vol", text, re.IGNORECASE):
            score += 0.35
        if re.search(r"AWB\s*No", text, re.IGNORECASE):
            score += 0.30
        if re.search(r"Length", text, re.IGNORECASE) and re.search(r"Width", text, re.IGNORECASE):
            score += 0.25
        if format_hint == "yellow_dws":
            score += 0.15
        return min(1.0, score)

    def parse(self, text: str, lines: list[str]) -> ParsedOverlayMetadata:
        meta = ParsedOverlayMetadata(profile_name=self.name, raw_lines=lines)
        matched = 0

        # Location
        m = re.search(r"Location[:\s.]+([^\n\r]+)", text, re.IGNORECASE)
        if m:
            meta.location = m.group(1).strip()
            meta.field_rules["location"] = "yellow_dws.location_kv"
            matched += 1

        # AWB No
        m = re.search(r"AWB\s*(?:No|Number)?[:\s.]+([A-Z0-9-]{8,35})", text, re.IGNORECASE)
        if m:
            cand = re.sub(r"[^A-Za-z0-9]", "", m.group(1))
            if is_valid_tracking(cand):
                meta.tracking_number = cand
                meta.raw_tracking_text = m.group(0).strip()
                meta.field_rules["tracking_number"] = "yellow_dws.awb_no"
                matched += 1
            meta.field_rules["tracking_number"] = "yellow_dws.awb_no"
            matched += 1

        # Dimensions: discrete length, width, height preferred
        dim_res = (
            parse_discrete_dimensions(text)
            or parse_arbitrary_labelled_dimensions(text)
            or parse_triplet_dimensions(text, self.triplet_order, self.verified_dim_unit)
        )
        if dim_res:
            l, w, h, u, raw = dim_res
            meta.length = l
            meta.width = w
            meta.height = h
            meta.dimension_unit = u or self.verified_dim_unit
            meta.raw_dimension_text = raw
            meta.field_rules["dimensions"] = "yellow_dws.discrete"
            matched += 1

        # Volume
        m = re.search(r"R\.?\s*Vol\.?[:\s.]+([\d]+(?:\.[\d]+)?)", text, re.IGNORECASE)
        if m:
            try:
                meta.volume = float(m.group(1))
                meta.raw_volume_text = m.group(0).strip()
                meta.field_rules["volume"] = "yellow_dws.r_vol"
            except ValueError:
                pass

        # Weight
        act_val, act_u, vol_val, vol_u, p_val, p_u, raw_w = parse_weight_with_rules(text, self.verified_weight_unit)
        if p_val is not None:
            meta.weight_value = p_val
            meta.weight_unit = p_u
            meta.raw_weight_text = raw_w
            meta.field_rules["weight"] = "yellow_dws.weight"
            matched += 1

        # Timestamp
        m = re.search(r"Time[:\s.]+([^\n\r]+)", text, re.IGNORECASE)
        if m:
            meta.timestamp = m.group(1).strip()
            meta.field_rules["timestamp"] = "yellow_dws.time"

        meta.confidence = round(min(1.0, matched / 5.0), 2)
        return meta


# ============================================================================
# Profile 3: White-on-Black Scanner Overlay
# ============================================================================

class WhiteTextScannerProfile(BaseOverlayProfile):
    """
    Top-left white monospaced text on black:
    Location, barcode, dimension triplet, weight, time, sorter ID.
    Supports triplet: 'Dimension: 0208 X 0148 X 0163'
    """

    name = "white_text_scanner"
    verified_dim_unit = None  # Triplet format does not declare unit in image; do not guess
    verified_weight_unit = "kg"
    triplet_order = ("length", "width", "height")

    def can_handle(self, text: str, format_hint: str | None = None) -> float:
        score = 0.0
        if re.search(r"Sorter\s*ID|SorterNo", text, re.IGNORECASE):
            score += 0.35
        if re.search(r"Barcode[:\s]+[A-Z0-9]+", text, re.IGNORECASE):
            score += 0.30
        if re.search(r"Dimension[:\s]+\d+\s*[xX]\s*\d+\s*[xX]\s*\d+", text, re.IGNORECASE):
            score += 0.35
        if format_hint == "white_text_scanner":
            score += 0.15
        return min(1.0, score)

    def parse(self, text: str, lines: list[str]) -> ParsedOverlayMetadata:
        meta = ParsedOverlayMetadata(profile_name=self.name, raw_lines=lines)
        matched = 0

        # Barcode or AWB
        m = re.search(r"(?:Barcode|AWB)[:\s.]+([A-Z0-9-]{8,35})", text, re.IGNORECASE)
        if m:
            cand = re.sub(r"[^A-Za-z0-9]", "", m.group(1))
            if is_valid_tracking(cand):
                meta.tracking_number = cand
                meta.raw_tracking_text = m.group(0).strip()
                meta.field_rules["tracking_number"] = "white_text_scanner.barcode"
                matched += 1

        # Dimension triplet (e.g. 'Dimension: 0208 X 0148 X 0163')
        dim_res = (
            parse_triplet_dimensions(text, self.triplet_order, self.verified_dim_unit)
            or parse_arbitrary_labelled_dimensions(text)
            or parse_discrete_dimensions(text)
        )
        if dim_res:
            l, w, h, u, raw = dim_res
            meta.length = l
            meta.width = w
            meta.height = h
            meta.dimension_unit = u  # May be None if unit is undeclared (never guessed)
            meta.raw_dimension_text = raw
            meta.field_rules["dimensions"] = "white_text_scanner.triplet"
            matched += 1

        # Weight
        act_val, act_u, vol_val, vol_u, p_val, p_u, raw_w = parse_weight_with_rules(text, self.verified_weight_unit)
        if p_val is not None:
            meta.weight_value = p_val
            meta.weight_unit = p_u
            meta.raw_weight_text = raw_w
            meta.field_rules["weight"] = "white_text_scanner.weight"
            matched += 1

        # Location
        m = re.search(r"Location[:\s.]+([^\n\r]+)", text, re.IGNORECASE)
        if m:
            meta.location = m.group(1).strip()
            meta.field_rules["location"] = "white_text_scanner.location"

        # Time
        m = re.search(r"Time[:\s.]+([^\n\r]+)", text, re.IGNORECASE)
        if m:
            meta.timestamp = m.group(1).strip()
            meta.field_rules["timestamp"] = "white_text_scanner.time"

        # Sorter ID
        m = re.search(r"(?:Sorter\s*ID|SorterNo)[:\s.]+([^\n\r]+)", text, re.IGNORECASE)
        if m:
            meta.sorter_id = m.group(1).strip()
            meta.field_rules["sorter_id"] = "white_text_scanner.sorter_id"

        meta.confidence = round(min(1.0, matched / 4.0), 2)
        return meta


# ============================================================================
# Profile 4: Ekart-style Overlay
# ============================================================================

class EkartProfile(BaseOverlayProfile):
    """
    Small black information box with white text:
    Barcode, weight, L*W*H (e.g., 'L * W * H: 0330 * 0369 * 0040'), scan time.
    """

    name = "ekart"
    verified_dim_unit = None  # Ekart mm/cm not guessed unless profile or text declares
    verified_weight_unit = "kg"
    triplet_order = ("length", "width", "height")

    def can_handle(self, text: str, format_hint: str | None = None) -> float:
        score = 0.0
        if re.search(r"L\s*\*\s*W\s*\*\s*H", text, re.IGNORECASE):
            score += 0.45
        if re.search(r"Scan\s*Time", text, re.IGNORECASE):
            score += 0.30
        if re.search(r"\b(?:FMP[CP]\d+|EKART)\b", text, re.IGNORECASE):
            score += 0.30
        if format_hint == "ekart":
            score += 0.15
        return min(1.0, score)

    def parse(self, text: str, lines: list[str]) -> ParsedOverlayMetadata:
        meta = ParsedOverlayMetadata(profile_name=self.name, raw_lines=lines)
        matched = 0

        # Barcode - require explicit label or recognized prefix with digits
        m = re.search(r"(?:Barcode|AWB|Tracking)[:\s.]+([A-Z0-9-]{8,35})", text, re.IGNORECASE)
        if not m:
            m = re.search(r"\b(FMP[CP]\d{8,14}|PROC\d{10})\b", text, re.IGNORECASE)
        if m:
            cand = re.sub(r"[^A-Za-z0-9]", "", m.group(1))
            if is_valid_tracking(cand):
                meta.tracking_number = cand
                meta.raw_tracking_text = m.group(0).strip()
                meta.field_rules["tracking_number"] = "ekart.barcode"
                matched += 1

        # L * W * H
        dim_res = (
            parse_triplet_dimensions(text, self.triplet_order, self.verified_dim_unit)
            or parse_arbitrary_labelled_dimensions(text)
        )
        if dim_res:
            l, w, h, u, raw = dim_res
            meta.length = l
            meta.width = w
            meta.height = h
            meta.dimension_unit = u
            meta.raw_dimension_text = raw
            meta.field_rules["dimensions"] = "ekart.l_w_h_triplet"
            matched += 1

        # Weight
        act_val, act_u, vol_val, vol_u, p_val, p_u, raw_w = parse_weight_with_rules(text, self.verified_weight_unit)
        if p_val is not None:
            meta.weight_value = p_val
            meta.weight_unit = p_u
            meta.raw_weight_text = raw_w
            meta.field_rules["weight"] = "ekart.weight"
            matched += 1

        # Scan Time
        m = re.search(r"(?:Scan\s*Time|Time)[:\s.]+([^\n\r]+)", text, re.IGNORECASE)
        if m:
            meta.timestamp = m.group(1).strip()
            meta.field_rules["timestamp"] = "ekart.scan_time"

        meta.confidence = round(min(1.0, matched / 3.0), 2)
        return meta


# ============================================================================
# Profile 5: Bottom Scanner Information Panel
# ============================================================================

class BottomPanelProfile(BaseOverlayProfile):
    """
    Grey/white panel near the bottom of the image:
    L, W, H, A.W, V.W, AWB.
    Crucial: retains A.W and V.W as distinct fields.
    """

    name = "bottom_panel"
    verified_dim_unit = "cm"
    verified_weight_unit = "kg"
    triplet_order = ("length", "width", "height")

    def can_handle(self, text: str, format_hint: str | None = None) -> float:
        score = 0.0
        if re.search(r"A\.W\b", text, re.IGNORECASE):
            score += 0.40
        if re.search(r"V\.W\b", text, re.IGNORECASE):
            score += 0.30
        if re.search(r"\bL[:\s]+\d+.*?\bW[:\s]+\d+.*?\bH[:\s]+\d+", text, re.IGNORECASE):
            score += 0.30
        if format_hint == "bottom_panel":
            score += 0.15
        return min(1.0, score)

    def parse(self, text: str, lines: list[str]) -> ParsedOverlayMetadata:
        meta = ParsedOverlayMetadata(profile_name=self.name, raw_lines=lines)
        matched = 0

        # AWB
        m = re.search(r"(?:AWB(?:\s*No)?|Barcode|Tracking)[:\s.]+([A-Z0-9-]{8,35})", text, re.IGNORECASE)
        if m:
            cand = re.sub(r"[^A-Za-z0-9]", "", m.group(1))
            if is_valid_tracking(cand):
                meta.tracking_number = cand
                meta.raw_tracking_text = m.group(0).strip()
                meta.field_rules["tracking_number"] = "bottom_panel.awb"
                matched += 1

        # Dimensions: L, W, H in bottom panel
        dim_res = (
            parse_arbitrary_labelled_dimensions(text)
            or parse_discrete_dimensions(text)
            or parse_triplet_dimensions(text, self.triplet_order, self.verified_dim_unit)
        )
        if dim_res:
            l, w, h, u, raw = dim_res
            meta.length = l
            meta.width = w
            meta.height = h
            meta.dimension_unit = u or self.verified_dim_unit
            meta.raw_dimension_text = raw
            meta.field_rules["dimensions"] = "bottom_panel.l_w_h"
            matched += 1

        # Weight: Separate A.W (Actual Weight) and V.W (Volumetric Weight)
        act_val, act_u, vol_val, vol_u, p_val, p_u, raw_w = parse_weight_with_rules(text, self.verified_weight_unit)
        if act_val is not None:
            meta.actual_weight = act_val
            meta.actual_weight_unit = act_u or self.verified_weight_unit
            meta.field_rules["actual_weight"] = "bottom_panel.actual_weight"
            matched += 1
        if vol_val is not None:
            meta.volumetric_weight = vol_val
            meta.volumetric_weight_unit = vol_u or self.verified_weight_unit
            meta.field_rules["volumetric_weight"] = "bottom_panel.volumetric_weight"

        # Primary weight value is actual weight
        meta.weight_value = act_val if act_val is not None else p_val
        meta.weight_unit = act_u if act_u is not None else p_u
        meta.raw_weight_text = raw_w
        if meta.weight_value is not None:
            meta.field_rules["weight"] = "bottom_panel.weight_reconciled"

        meta.confidence = round(min(1.0, matched / 4.0), 2)
        return meta


# ============================================================================
# Profile 6: Generic Overlay Fallback
# ============================================================================

class GenericOverlayProfile(BaseOverlayProfile):
    """
    Fallback profile for general carrier/scanner overlays.
    """

    name = "generic_overlay"
    verified_dim_unit = None
    verified_weight_unit = None
    triplet_order = ("length", "width", "height")

    def can_handle(self, text: str, format_hint: str | None = None) -> float:
        return 0.10  # Always can act as lowest-priority fallback

    def parse(self, text: str, lines: list[str]) -> ParsedOverlayMetadata:
        meta = ParsedOverlayMetadata(profile_name=self.name, raw_lines=lines)
        matched = 0

        # AWB / Tracking
        m = re.search(r"(?:AWB(?:\s*No)?|Barcode|Tracking(?:\s*No)?|TRK#?)[:\s.]+([A-Z0-9-]{8,35})", text, re.IGNORECASE)
        if not m:
            m = re.search(r"\b(1Z[0-9A-Z]{16}|PROC\d{10}|FMP[CP]\d{8,14}|\d{10,22})\b", text, re.IGNORECASE)
        if m:
            cand = re.sub(r"[^A-Za-z0-9]", "", m.group(1))
            if is_valid_tracking(cand):
                meta.tracking_number = cand
                meta.raw_tracking_text = m.group(0).strip()
                meta.field_rules["tracking_number"] = "generic_overlay.tracking"
                matched += 1

        # Dimensions: arbitrary labeled -> discrete -> triplet
        dim_res = (
            parse_arbitrary_labelled_dimensions(text)
            or parse_discrete_dimensions(text)
            or parse_triplet_dimensions(text, self.triplet_order, self.verified_dim_unit)
        )
        if dim_res:
            l, w, h, u, raw = dim_res
            meta.length = l
            meta.width = w
            meta.height = h
            meta.dimension_unit = u
            meta.raw_dimension_text = raw
            meta.field_rules["dimensions"] = "generic_overlay.dimensions"
            matched += 1

        # Weight
        act_val, act_u, vol_val, vol_u, p_val, p_u, raw_w = parse_weight_with_rules(text, self.verified_weight_unit)
        meta.actual_weight = act_val
        meta.actual_weight_unit = act_u
        meta.volumetric_weight = vol_val
        meta.volumetric_weight_unit = vol_u
        meta.weight_value = p_val
        meta.weight_unit = p_u
        meta.raw_weight_text = raw_w
        if meta.weight_value is not None:
            meta.field_rules["weight"] = "generic_overlay.weight"
            matched += 1

        # Volume
        m = re.search(r"(?:R\.?\s*Vol|Volume)[:\s.]+([\d]+(?:\.[\d]+)?)", text, re.IGNORECASE)
        if m:
            try:
                meta.volume = float(m.group(1))
                meta.raw_volume_text = m.group(0).strip()
            except ValueError:
                pass

        # Location
        m = re.search(r"Location[:\s.]+([^\n\r]+)", text, re.IGNORECASE)
        if m:
            meta.location = m.group(1).strip()

        # Time
        m = re.search(r"(?:Time|Timestamp)[:\s.]+([^\n\r]+)", text, re.IGNORECASE)
        if m:
            meta.timestamp = m.group(1).strip()

        meta.confidence = round(min(1.0, matched / 3.0), 2)
        return meta


# ============================================================================
# Profile Registry
# ============================================================================

class OverlayProfileRegistry:
    """Registry coordinating format-specific overlay profiles."""

    def __init__(self):
        self.profiles: list[BaseOverlayProfile] = [
            GreenDwsProfile(),
            YellowDwsProfile(),
            WhiteTextScannerProfile(),
            EkartProfile(),
            BottomPanelProfile(),
            GenericOverlayProfile(),
        ]
        self._profiles_by_name = {p.name: p for p in self.profiles}

    def get_profile(self, name: str) -> BaseOverlayProfile | None:
        return self._profiles_by_name.get(name)

    def select_profile(self, text: str, format_hint: str | None = None) -> BaseOverlayProfile:
        """
        Select profile based on extracted field evidence rather than visual ROI classification alone.
        Requires anchor coverage (AWB, Weight, Dimensions, Time/Location).
        Never selects Ekart if only INVENTIMENT is extracted with no other expected fields.
        """
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        best_profile: BaseOverlayProfile = self._profiles_by_name["generic_overlay"]
        best_evidence_score = 0.0

        for profile in self.profiles:
            if profile.name == "generic_overlay":
                continue

            meta = profile.parse(text, lines)

            # Check anchor coverage of extracted fields
            has_tracking = bool(meta.tracking_number and is_valid_tracking(meta.tracking_number))
            has_weight = bool(meta.weight_value is not None or meta.actual_weight is not None)
            has_dims = bool(meta.length is not None and meta.width is not None and meta.height is not None)
            has_meta = bool(
                meta.location is not None
                or meta.timestamp is not None
                or meta.sorter_id is not None
                or meta.machine_id is not None
            )

            anchor_count = sum([has_tracking, has_weight, has_dims, has_meta])

            # Textual cue score
            cue_score = profile.can_handle(text, format_hint=format_hint)

            # Profile-specific evidence rules
            if profile.name == "ekart":
                # Never select Ekart if tracking is invalid/INVENTIMENT and weight/dims are null
                if not has_dims and not has_weight and not (has_meta and has_tracking):
                    continue
                # Ekart requires explicit L*W*H or Scan Time or (Barcode with valid digits)
                has_ekart_lwh = bool(re.search(r"L\s*\*\s*W\s*\*\s*H", text, re.IGNORECASE))
                has_ekart_time = bool(re.search(r"Scan\s*Time", text, re.IGNORECASE))
                has_ekart_code = bool(re.search(r"\b(?:FMP[CP]\d+|EKART)\b", text, re.IGNORECASE))
                if not (has_ekart_lwh or has_ekart_time or has_ekart_code):
                    continue

            elif profile.name == "bottom_panel":
                has_aw_vw = bool(re.search(r"\b[AV]\.W\b", text, re.IGNORECASE))
                has_bottom_dims = bool(re.search(r"\bL[:\s]+\d+.*?\bW[:\s]+\d+.*?\bH[:\s]+\d+", text, re.IGNORECASE))
                if not (has_aw_vw or has_bottom_dims):
                    continue

            elif profile.name == "green_dws":
                has_dws_cue = bool(re.search(r"Machine\s*ID|Real\s*Vol|R\.?\s*Vol", text, re.IGNORECASE))
                if not has_dws_cue:
                    continue

            elif profile.name == "yellow_dws":
                has_yellow_cue = bool(re.search(r"R\.?\s*Vol|AWB\s*No", text, re.IGNORECASE)) and bool(re.search(r"Length|Width", text, re.IGNORECASE))
                if not has_yellow_cue:
                    continue

            elif profile.name == "white_text_scanner":
                has_wts_cue = bool(re.search(r"Sorter\s*ID|Dimension[:\s]+\d+\s*[xX]\s*\d+\s*[xX]\s*\d+", text, re.IGNORECASE))
                if not has_wts_cue:
                    continue

            # Weight evidence
            format_boost = 0.10 if (format_hint and format_hint == profile.name) else 0.0
            evidence_score = (anchor_count / 4.0) * 0.65 + cue_score * 0.25 + format_boost

            # Must satisfy evidence threshold: at least 2 anchors (or 1 major anchor + high cue)
            meets_threshold = (anchor_count >= 2) or (anchor_count >= 1 and cue_score >= 0.50)

            if meets_threshold and evidence_score > best_evidence_score:
                best_evidence_score = evidence_score
                best_profile = profile

        return best_profile

    def parse(self, text: str, format_hint: str | None = None) -> ParsedOverlayMetadata:
        """
        Select profile based on evidence and parse text.
        If no profile meets the evidence threshold, uses generic parsing and marks profile as unknown.
        """
        profile = self.select_profile(text, format_hint)
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        meta = profile.parse(text, lines)

        # If generic overlay fallback was used or evidence threshold was not met, mark profile as unknown
        if profile.name == "generic_overlay":
            valid_fields = sum(
                1 for f in (
                    meta.tracking_number if is_valid_tracking(meta.tracking_number) else None,
                    meta.weight_value,
                    meta.length,
                ) if f is not None
            )
            if valid_fields < 2:
                meta.profile_name = "unknown"

        return meta

    def audit_profiles(self, text: str, format_hint: str | None = None) -> list[dict[str, Any]]:
        """
        Produce a complete auditable report of all evaluated profiles,
        their anchor coverage, evidence scores, and selection / rejection reasons.
        """
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        selected = self.select_profile(text, format_hint)
        audit = []

        for profile in self.profiles:
            meta = profile.parse(text, lines)
            has_tracking = bool(meta.tracking_number and is_valid_tracking(meta.tracking_number))
            has_weight = bool(meta.weight_value is not None or meta.actual_weight is not None)
            has_dims = bool(meta.length is not None and meta.width is not None and meta.height is not None)
            has_meta = bool(
                meta.location is not None
                or meta.timestamp is not None
                or meta.sorter_id is not None
                or meta.machine_id is not None
            )
            anchor_count = sum([has_tracking, has_weight, has_dims, has_meta])
            cue_score = profile.can_handle(text, format_hint=format_hint)

            status = "rejected"
            reason = None

            if profile.name == selected.name:
                status = "selected"
            elif profile.name == "ekart":
                if not has_dims and not has_weight and not (has_meta and has_tracking):
                    reason = "Disqualified: tracking invalid or null with null weight and dimensions"
                else:
                    reason = "Ekart specific cues (L*W*H or Scan Time) not satisfied"
            elif profile.name == "bottom_panel":
                reason = "Bottom panel anchors (A.W / V.W / bottom dimensions) not satisfied"
            elif profile.name in ("green_dws", "yellow_dws"):
                reason = "DWS anchors (Machine ID / Real Vol / AWB No) not satisfied"
            elif profile.name == "white_text_scanner":
                reason = "White text scanner anchors (Sorter ID / triplet dimensions) not satisfied"
            else:
                reason = "Evidence threshold not met"

            format_boost = 0.10 if (format_hint and format_hint == profile.name) else 0.0
            evidence_score = (anchor_count / 4.0) * 0.65 + cue_score * 0.25 + format_boost

            audit.append({
                "profile_name": profile.name,
                "status": status,
                "evidence_score": round(evidence_score, 3),
                "cue_score": round(cue_score, 3),
                "anchor_coverage": {
                    "tracking": has_tracking,
                    "weight": has_weight,
                    "dimensions": has_dims,
                    "metadata": has_meta,
                },
                "anchor_count": anchor_count,
                "disqualification_reason": reason if status != "selected" else None,
            })

        return audit


# Global registry singleton
parser_registry = OverlayProfileRegistry()
