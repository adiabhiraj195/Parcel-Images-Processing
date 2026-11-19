"""Local image validation, orientation correction, quality analysis, and safe cropping."""

import os

import cv2
import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel

from app.pipeline.interfaces import IImagePreprocessor


class ImageValidationError(Exception):
    """Raised when an image fails format, corruption, or dimension limits."""


class ImageQualityMetrics(BaseModel):
    """Quality indicators computed via OpenCV heuristics."""

    width: int
    height: int
    channels: int
    blur_score: float        # Laplacian variance: higher = sharper
    is_blurry: bool          # Heuristic flag (blur_score < threshold)
    mean_brightness: float   # 0 to 255
    contrast_std: float      # Standard deviation of pixel intensities
    exif_rotated: bool


class PreprocessedImage:
    """Container holding loaded image data and quality indicators."""

    def __init__(
        self,
        bgr_image: np.ndarray,
        pil_image: Image.Image,
        metrics: ImageQualityMetrics,
    ):
        self.bgr_image = bgr_image
        self.pil_image = pil_image
        self.metrics = metrics

    @property
    def rgb_image(self) -> np.ndarray:
        return cv2.cvtColor(self.bgr_image, cv2.COLOR_BGR2RGB)


class OpenCVPreprocessor(IImagePreprocessor):
    """Pillow and OpenCV image preprocessor with Apple Silicon memory safety limits."""

    # Memory guardrails for M1: max 30 megapixels (~5477 x 5477)
    MAX_PIXELS = 30_000_000
    MAX_FILE_BYTES = 25 * 1024 * 1024  # 25 MB
    BLUR_THRESHOLD = 80.0               # Calibrated Laplacian variance threshold

    def validate_and_load(self, file_path: str) -> PreprocessedImage:
        """
        Validate file content, check for corruption, apply EXIF orientation,
        and compute OpenCV quality heuristics.
        """
        if not os.path.exists(file_path):
            raise ImageValidationError(f"File not found: {file_path}")

        file_size = os.path.getsize(file_path)
        if file_size > self.MAX_FILE_BYTES:
            raise ImageValidationError(
                f"File size ({file_size} bytes) exceeds limit of {self.MAX_FILE_BYTES} bytes."
            )

        try:
            # 1. Open with Pillow to verify image headers and structure
            with Image.open(file_path) as img:
                img.verify()

            # Re-open after verify() (Pillow requirement)
            with Image.open(file_path) as raw_img:
                # 2. Check pixel count guardrail
                w, h = raw_img.size
                if w * h > self.MAX_PIXELS:
                    raise ImageValidationError(
                        f"Image resolution ({w}x{h} = {w*h} px) exceeds safe M1 limit of {self.MAX_PIXELS} px."
                    )

                # 3. Apply EXIF orientation
                exif_rotated = False
                try:
                    transposed = ImageOps.exif_transpose(raw_img)
                    if transposed is not None and transposed.size != raw_img.size:
                        exif_rotated = True
                        pil_img = transposed.convert("RGB")
                    else:
                        pil_img = raw_img.convert("RGB")
                except Exception:
                    pil_img = raw_img.convert("RGB")

        except (UnidentifiedImageError, OSError, SyntaxError) as e:
            raise ImageValidationError(f"Corrupt or unsupported image file: {e!s}")

        # 4. Convert to OpenCV BGR ndarray
        rgb_array = np.array(pil_img)
        bgr_image = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)

        # 5. Compute quality heuristics using OpenCV
        gray = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)
        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_brightness = float(np.mean(gray))
        contrast_std = float(np.std(gray))

        metrics = ImageQualityMetrics(
            width=pil_img.width,
            height=pil_img.height,
            channels=3,
            blur_score=round(blur_score, 2),
            is_blurry=blur_score < self.BLUR_THRESHOLD,
            mean_brightness=round(mean_brightness, 2),
            contrast_std=round(contrast_std, 2),
            exif_rotated=exif_rotated,
        )

        return PreprocessedImage(bgr_image=bgr_image, pil_image=pil_img, metrics=metrics)

    def safe_crop(
        self,
        image: np.ndarray,
        bbox_xyxy: list[int],
        padding_ratio: float = 0.05,
    ) -> tuple[np.ndarray, list[int]]:
        """
        Safely crop image bounding box with bounds clamping and optional padding.
        Returns (cropped_bgr, clamped_bbox_xyxy).
        """
        h, w = image.shape[:2]
        x1, y1, x2, y2 = bbox_xyxy

        # Ensure valid ordering
        x_min, x_max = min(x1, x2), max(x1, x2)
        y_min, y_max = min(y1, y2), max(y1, y2)

        # Proportional padding
        box_w = x_max - x_min
        box_h = y_max - y_min
        pad_x = int(box_w * padding_ratio)
        pad_y = int(box_h * padding_ratio)

        # Clamp to image boundaries
        clamped_x1 = max(0, x_min - pad_x)
        clamped_y1 = max(0, y_min - pad_y)
        clamped_x2 = min(w, x_max + pad_x)
        clamped_y2 = min(h, y_max + pad_y)

        # Guard against zero-area crops
        if clamped_x2 <= clamped_x1 or clamped_y2 <= clamped_y1:
            # Fallback to 1x1 region
            clamped_x1, clamped_y1 = 0, 0
            clamped_x2, clamped_y2 = min(w, 10), min(h, 10)

        cropped = image[clamped_y1:clamped_y2, clamped_x1:clamped_x2]
        return cropped, [clamped_x1, clamped_y1, clamped_x2, clamped_y2]

    def resize_for_inference(
        self,
        image: np.ndarray,
        max_dim: int = 960,
    ) -> tuple[np.ndarray, float]:
        """
        Downscale image if larger than max_dim to keep M1 inference rapid.
        Returns (resized_image, scale_factor).
        """
        h, w = image.shape[:2]
        if max(h, w) <= max_dim:
            return image, 1.0

        scale = max_dim / float(max(h, w))
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
        return resized, scale


preprocessor = OpenCVPreprocessor()
