"""Unit tests for format-specific overlay profiles in parser_registry."""

from app.pipeline.parser_registry import (
    parse_arbitrary_labelled_dimensions,
    parser_registry,
)


def test_green_dws_profile():
    """Verify green_dws profile parsing."""
    text = (
        "Location: HYDERABAD_HUB_DWS\n"
        "AWB No: 19108888777766\n"
        "Dimension: 45.0 X 30.0 X 20.0 cm\n"
        "Real Vol: 27000.0 cm\n"
        "Weight: 3500.0 gm\n"
        "Time: 2025-07-04 10:15:30\n"
        "Machine ID: DWS-M04"
    )
    res = parser_registry.parse(text, format_hint="green_dws")

    assert res.profile_name == "green_dws"
    assert res.tracking_number == "19108888777766"
    assert res.length == 45.0
    assert res.width == 30.0
    assert res.height == 20.0
    assert res.dimension_unit == "cm"
    assert res.volume == 27000.0
    assert res.weight_value == 3500.0
    assert res.weight_unit == "g"
    assert res.machine_id == "DWS-M04"
    assert res.location == "HYDERABAD_HUB_DWS"


def test_yellow_dws_profile():
    """Verify yellow_dws profile parsing."""
    text = (
        "Location: Bangalore_Hoskote_GW (Karnataka)\n"
        "AWB No: 19106721621035\n"
        "Length: 75.300 cm\n"
        "Width: 49.200 cm\n"
        "Height: 44.400 cm\n"
        "R.Vol.: 134230.000 cm (cubic)\n"
        "Weight: 14060.000 gm\n"
        "Time: 7/4/2025 12:43:07 PM"
    )
    res = parser_registry.parse(text, format_hint="yellow_dws")

    assert res.profile_name == "yellow_dws"
    assert res.tracking_number == "19106721621035"
    assert res.length == 75.3
    assert res.width == 49.2
    assert res.height == 44.4
    assert res.dimension_unit == "cm"
    assert res.volume == 134230.0
    assert res.weight_value == 14060.0
    assert res.weight_unit == "g"


def test_white_text_scanner_triplet():
    """Verify white_text_scanner profile with three-number dimension triplet."""
    text = (
        "Location: DEL_AIRPORT_SORT\n"
        "Barcode: 940011189922334455\n"
        "Dimension: 0208 X 0148 X 0163\n"
        "Weight: 2.350 KG\n"
        "Time: 14:32:00\n"
        "Sorter ID: S-09"
    )
    res = parser_registry.parse(text, format_hint="white_text_scanner")

    assert res.profile_name == "white_text_scanner"
    assert res.tracking_number == "940011189922334455"
    assert res.length == 208.0
    assert res.width == 148.0
    assert res.height == 163.0
    # Unit was not declared in image text and profile has no verified default unit -> None (never guess!)
    assert res.dimension_unit is None
    assert "0208 X 0148 X 0163" in res.raw_dimension_text
    assert res.weight_value == 2.35
    assert res.weight_unit == "kg"
    assert res.sorter_id == "S-09"


def test_arbitrary_labelled_dimension_order():
    """Verify arbitrary labeled dimension order: L0214 X H0061 X W0154."""
    text = "Dimension: L0214 X H0061 X W0154 cm"
    parsed = parse_arbitrary_labelled_dimensions(text)
    assert parsed is not None
    l, w, h, u, raw = parsed

    assert l == 214.0
    assert h == 61.0
    assert w == 154.0
    assert u == "cm"


def test_ekart_profile():
    """Verify ekart profile parsing with L * W * H."""
    text = (
        "Barcode: FMPP1234567890\n"
        "Weight: 0.850 kg\n"
        "L * W * H: 0330 * 0369 * 0040\n"
        "Scan Time: 2025-07-04 15:30:00"
    )
    res = parser_registry.parse(text, format_hint="ekart")

    assert res.profile_name == "ekart"
    assert res.tracking_number == "FMPP1234567890"
    assert res.length == 330.0
    assert res.width == 369.0
    assert res.height == 40.0
    assert res.weight_value == 0.85
    assert res.weight_unit == "kg"


def test_bottom_panel_profile_separate_weights():
    """Verify bottom_panel profile maintains A.W and V.W separately."""
    text = (
        "L: 32.5  W: 24.0  H: 18.0 cm\n"
        "A.W: 1.450 kg\n"
        "V.W: 2.808 kg\n"
        "AWB: 888777666555"
    )
    res = parser_registry.parse(text, format_hint="bottom_panel")

    assert res.profile_name == "bottom_panel"
    assert res.tracking_number == "888777666555"
    assert res.length == 32.5
    assert res.width == 24.0
    assert res.height == 18.0
    assert res.dimension_unit == "cm"
    assert res.actual_weight == 1.45
    assert res.actual_weight_unit == "kg"
    assert res.volumetric_weight == 2.808
    assert res.volumetric_weight_unit == "kg"
    assert res.weight_value == 1.45  # Primary weight should be actual weight


def test_never_guess_units_from_magnitude():
    """Verify that dimensions with no units and unknown profile unit remain None."""
    text = "Dimension: 250 x 180 x 120"
    res = parser_registry.parse(text, format_hint="generic_overlay")
    assert res.length == 250.0
    assert res.width == 180.0
    assert res.height == 120.0
    # Must NOT guess mm even though numbers are large!
    assert res.dimension_unit is None
