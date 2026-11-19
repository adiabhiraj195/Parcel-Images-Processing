"""Local 1D and 2D barcode/QR decoder adapter using ZXing-C++."""

from typing import Any

import numpy as np
from PIL import Image

from app.pipeline.interfaces import BarcodeResult, IBarcodeDecoder


class ZXingBarcodeDecoder(IBarcodeDecoder):
    """
    High-performance, standalone barcode and QR code decoder.
    Uses C++ compiled wheels with native Apple Silicon arm64 support.
    """

    def decode(self, image_input: Any) -> list[BarcodeResult]:
        """
        Decode 1D barcodes (Code 128, Code 39, EAN, UPC) and 2D barcodes (QR, DataMatrix).
        Accepts numpy ndarray (BGR or Grayscale), PIL Image, or zxingcpp.Image.
        """
        import zxingcpp

        if isinstance(image_input, Image.Image):
            image_array = np.ascontiguousarray(np.array(image_input), dtype=np.uint8)
        elif isinstance(image_input, np.ndarray):
            image_array = np.ascontiguousarray(image_input, dtype=np.uint8)
        else:
            image_array = image_input

        try:
            results = zxingcpp.read_barcodes(image_array)
        except Exception:
            return []

        decoded_items: list[BarcodeResult] = []

        for item in results:
            if not item.text:
                continue

            # Normalize symbology name e.g. "BarcodeFormat.Code 128" -> "Code128"
            symbology_str = (
                str(item.format)
                .replace("BarcodeFormat.", "")
                .replace(" ", "")
            )

            bbox: list[int] | None = None
            if hasattr(item, "position"):
                pos = item.position
                xs = [pos.top_left.x, pos.top_right.x, pos.bottom_right.x, pos.bottom_left.x]
                ys = [pos.top_left.y, pos.top_right.y, pos.bottom_right.y, pos.bottom_left.y]
                bbox = [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))]

            decoded_items.append(
                BarcodeResult(
                    raw_value=item.text.strip(),
                    symbology=symbology_str,
                    bbox_xyxy=bbox,
                )
            )

        return decoded_items


class MockBarcodeDecoder(IBarcodeDecoder):
    """Deterministic mock barcode decoder for tests."""

    def __init__(self, predefined_results: list[BarcodeResult] | None = None):
        self.predefined_results = predefined_results

    def decode(self, image_input: Any) -> list[BarcodeResult]:
        if self.predefined_results is not None:
            return self.predefined_results
        return [
            BarcodeResult(
                raw_value="1Z9999999999999999",
                symbology="Code128",
                bbox_xyxy=[50, 100, 350, 180],
            )
        ]
