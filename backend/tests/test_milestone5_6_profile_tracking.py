"""
Unit tests for Milestone 5 (Profile Selection) & Milestone 6 (Tracking Validation).
Verifies:
- Profile selection based on extracted field evidence and anchor coverage.
- Rejection of Ekart selection when only 'INVENTIMENT' is extracted.
- Fallback to unknown profile when evidence threshold is unmet.
- Rejection of alphabetic-only tracking values (e.g., 'INVENTIMENT').
- Acceptance of alphanumeric barcode IDs (e.g., 'PROC0000316077').
- Preservation of raw OCR candidates in provenance audit.
"""

import pytest

from app.pipeline.interfaces import BarcodeResult, OcrResult, OcrTextBox, DetectionResult
from app.pipeline.parser import DeterministicFieldParser
from app.pipeline.parser_registry import parser_registry, is_valid_tracking


@pytest.fixture
def parser():
    return DeterministicFieldParser()


# ============================================================================
# Milestone 5: Profile Selection & Anchor Coverage Tests
# ============================================================================

def test_inventiment_never_selects_ekart():
    """
    Never select Ekart profile if parser extracted only INVENTIMENT and no
    other expected fields. Profile must fall back to unknown.
    """
    text = "Location: INMHAPVP\nINVENTIMENT"
    # Even if format_hint is passed as 'ekart' from visual ROI
    res = parser_registry.parse(text, format_hint="ekart")

    assert res.profile_name != "ekart"
    assert res.profile_name == "unknown"
    assert res.tracking_number is None  # INVENTIMENT rejected
    assert res.weight_value is None
    assert res.length is None


def test_ekart_requires_lwh_weight_or_barcode():
    """Ekart profile is selected when legitimate field evidence is present."""
    text = (
        "Barcode: FMPP0012345678\n"
        "Weight: 1.250 kg\n"
        "L * W * H: 0250 * 0180 * 0120\n"
        "Scan Time: 2025-07-04 14:20:00"
    )
    res = parser_registry.parse(text, format_hint="ekart")
    assert res.profile_name == "ekart"
    assert res.tracking_number == "FMPP0012345678"
    assert res.weight_value == 1.25
    assert res.length == 250.0
    assert res.width == 180.0
    assert res.height == 120.0


def test_unsupported_evidence_falls_back_to_unknown():
    """When text contains insufficient anchor evidence, profile is marked unknown."""
    text = "SOME RANDOM HEADER\nJUST RANDOM WORDS WITHOUT ANCHORS"
    res = parser_registry.parse(text, format_hint="green_dws")
    assert res.profile_name == "unknown"
    assert res.tracking_number is None


def test_bottom_panel_requires_aw_vw_or_dimensions():
    """Bottom panel profile requires actual A.W/V.W or panel dimension anchors."""
    # Text with genuine bottom panel evidence
    valid_text = (
        "L: 30 W: 20 H: 10\n"
        "A.W: 2.100 kg\n"
        "V.W: 1.500 kg\n"
        "AWB: 998877665544"
    )
    res = parser_registry.parse(valid_text, format_hint="bottom_panel")
    assert res.profile_name == "bottom_panel"
    assert res.actual_weight == 2.1
    assert res.volumetric_weight == 1.5
    assert res.tracking_number == "998877665544"

    # Text without bottom panel evidence should not be bottom_panel
    invalid_text = "AWB: 998877665544\nHELLO WORLD"
    res_invalid = parser_registry.parse(invalid_text, format_hint="bottom_panel")
    assert res_invalid.profile_name != "bottom_panel"


# ============================================================================
# Milestone 6: Tracking Validation Tests
# ============================================================================

def test_is_valid_tracking_rule():
    """Verify is_valid_tracking strictly enforces alphanumeric and digit presence."""
    # Alphabetic-only candidates MUST be rejected
    assert is_valid_tracking("INVENTIMENT") is False
    assert is_valid_tracking("INMHAPVP") is False
    assert is_valid_tracking("PACKAGELABEL") is False
    assert is_valid_tracking("ABCDEFGHIJKLMN") is False

    # Valid candidates containing at least one digit MUST be accepted
    assert is_valid_tracking("PROC0000316077") is True
    assert is_valid_tracking("FMPP1234567890") is True
    assert is_valid_tracking("1Z999AA10123456784") is True
    assert is_valid_tracking("19108888777766") is True
    assert is_valid_tracking("940011189922334455") is True

    # Too short (< 8 chars) MUST be rejected
    assert is_valid_tracking("1234567") is False
    assert is_valid_tracking("A12345") is False


def test_parser_rejects_inventiment_and_preserves_candidates(parser):
    """Parser must reject INVENTIMENT as tracking number and preserve candidates in audit."""
    text = "Location: INMHAPVP\nINVENTIMENT"
    ocr = OcrResult(engine="test", lines=[OcrTextBox(text=text, score=0.95)], full_text=text)
    detection = DetectionResult(model="test", detections=[])

    res = parser.parse(ocr, [], detection)
    # INVENTIMENT rejected
    assert res.tracking_number is None

    # Provenance audit must contain raw observed candidates
    track_prov = res.provenance.get("tracking_number")
    assert track_prov is not None
    assert track_prov.value is None
    assert len(track_prov.raw_candidates) >= 1
    assert any("INVENTIMENT" in c for c in track_prov.raw_candidates)


def test_parser_accepts_alphanumeric_proc_barcode(parser):
    """Parser must accept alphanumeric barcode IDs such as PROC0000316077."""
    barcodes = [BarcodeResult(raw_value="PROC0000316077", symbology="Code128")]
    text = "SCANNER BED\nPROC0000316077\nWT: 2.5 KG"
    ocr = OcrResult(engine="test", lines=[OcrTextBox(text=text, score=0.95)], full_text=text)
    detection = DetectionResult(model="test", detections=[])

    res = parser.parse(ocr, barcodes, detection)
    assert res.tracking_number == "PROC0000316077"
    assert "barcode" in res.provenance["tracking_number"].source
    assert res.provenance["tracking_number"].conflict_status == "AGREED"
