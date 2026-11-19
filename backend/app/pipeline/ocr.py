"""Local OCR adapters for text detection and recognition on Apple Silicon CPU.

Provides:
- Pretrained PaddleOCR engine (CPU-first, lightweight/mobile model with seamless CPU fallback)
- LocalCpuOcrEngine (OpenCV text-line segmentation + local TrOCR on CPU)
- Tesseract adapter for high-contrast metadata overlays
- Mock OCR engine for offline deterministic testing
- Factory helper to dynamically select the configured engine
"""

import logging
import os
import shutil
from typing import Any, Optional

import cv2
import numpy as np
from PIL import Image

from app.config import settings
from app.pipeline.interfaces import IOcrEngine, OcrResult, OcrTextBox

logger = logging.getLogger("ocr")


class LocalCpuOcrEngine(IOcrEngine):
    """
    Local CPU-first OCR engine using OpenCV text-line segmentation and
    pretrained Hugging Face TrOCR (microsoft/trocr-small-printed) running
    locally on Apple Silicon CPU.
    """

    MODEL_VERSION = "microsoft/trocr-small-printed-cpu"

    def __init__(self, cache_dir: str | None = None):
        self.cache_dir = cache_dir or getattr(settings, "resolved_model_cache_dir", "./model_cache")
        self._processor = None
        self._model = None

    def get_health(self) -> dict[str, Any]:
        return {
            "engine": "local_cpu",
            "model_version": self.MODEL_VERSION,
            "runtime": "Apple Silicon CPU (TrOCR)",
            "readiness": "ready",
            "error": None,
        }

    def _ensure_loaded(self):
        """Lazy-load TrOCR tokenizer, processor, and vision-encoder-decoder weights."""
        if self._model is None or self._processor is None:
            import transformers
            from transformers import TrOCRProcessor, VisionEncoderDecoderModel

            transformers.logging.set_verbosity_error()

            cache_path = getattr(settings, "resolved_model_cache_dir", self.cache_dir)
            logger.info(f"Loading local TrOCR model on CPU from {cache_path}")
            try:
                self._processor = TrOCRProcessor.from_pretrained(
                    "microsoft/trocr-small-printed",
                    cache_dir=cache_path,
                    local_files_only=True,
                )
                self._model = VisionEncoderDecoderModel.from_pretrained(
                    "microsoft/trocr-small-printed",
                    cache_dir=cache_path,
                    local_files_only=True,
                )
            except Exception as e:
                logger.debug(f"TrOCR local_files_only notice ({e}), loading with standard cache...")
                self._processor = TrOCRProcessor.from_pretrained(
                    "microsoft/trocr-small-printed",
                    cache_dir=cache_path,
                )
                self._model = VisionEncoderDecoderModel.from_pretrained(
                    "microsoft/trocr-small-printed",
                    cache_dir=cache_path,
                )
            self._model.to("cpu")
            self._model.eval()

    def extract_text(
        self,
        image_input: Any,
        variant_label: str = "original",
        source_crop: str | None = None,
    ) -> OcrResult:
        """
        Run text line detection via OpenCV and recognition via TrOCR on Apple Silicon CPU.
        Returns recognized lines, bounding quad coordinates, and confidence scores.
        """
        if isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
            bgr_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                bgr_img = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.shape[2] == 4:
                bgr_img = cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
            else:
                bgr_img = image_input
            pil_img = Image.fromarray(cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB))
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        if bgr_img.size == 0 or bgr_img.shape[0] < 6 or bgr_img.shape[1] < 6:
            return OcrResult(
                engine="local_cpu",
                model_version=self.MODEL_VERSION,
                lines=[],
                full_text="",
                source_crop=source_crop,
                preprocessing_variant=variant_label,
            )

        self._ensure_loaded()
        import torch

        h, w = bgr_img.shape[:2]
        gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)

        # Contrast adaptive binarization for text lines
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
        )
        kernel_w = max(12, int(w * 0.06))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 3))
        dilated = cv2.dilate(thresh, kernel, iterations=1)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        line_boxes: list[tuple[int, int, int, int]] = []
        min_line_w = max(12, int(w * 0.04))
        min_line_h = max(5, int(h * 0.02))

        for cnt in contours:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            if bw >= min_line_w and bh >= min_line_h:
                line_boxes.append((bx, by, bw, bh))

        # Sort lines vertically
        line_boxes.sort(key=lambda b: b[1])

        # If no distinct line boxes or if crop is small, process entire crop as single line
        if not line_boxes:
            line_boxes = [(0, 0, w, h)]
        else:
            line_boxes = line_boxes[:10]  # Cap to maintain bounded CPU processing

        boxes: list[OcrTextBox] = []
        full_text_lines: list[str] = []

        for bx, by, bw, bh in line_boxes:
            pad_y = max(1, int(bh * 0.08))
            pad_x = max(1, int(bw * 0.04))
            y1 = max(0, by - pad_y)
            y2 = min(h, by + bh + pad_y)
            x1 = max(0, bx - pad_x)
            x2 = min(w, bx + bw + pad_x)

            line_crop = bgr_img[y1:y2, x1:x2]
            if line_crop.size == 0 or line_crop.shape[0] < 4 or line_crop.shape[1] < 4:
                continue

            pil_patch = Image.fromarray(cv2.cvtColor(line_crop, cv2.COLOR_BGR2RGB))
            try:
                pixel_values = self._processor(images=pil_patch, return_tensors="pt").pixel_values
                with torch.no_grad():
                    out = self._model.generate(
                        pixel_values,
                        return_dict_in_generate=True,
                        output_scores=True,
                        max_new_tokens=48,
                    )
                line_text = self._processor.batch_decode(out.sequences, skip_special_tokens=True)[0].strip()
                if not line_text:
                    continue

                # Compute genuine OCR confidence from token generation logits
                confs = []
                if hasattr(out, "scores") and out.scores:
                    for s in out.scores:
                        prob = torch.softmax(s, dim=-1).max().item()
                        confs.append(prob)
                raw_conf = float(np.mean(confs)) if confs else 0.85

                quad = [[bx, by], [bx + bw, by], [bx + bw, by + bh], [bx, by + bh]]
                boxes.append(
                    OcrTextBox(
                        text=line_text,
                        score=round(raw_conf, 4),
                        box=quad,
                        source_crop=source_crop,
                        preprocessing_variant=variant_label,
                    )
                )
                full_text_lines.append(line_text)
            except Exception as line_err:
                logger.debug(f"TrOCR line recognition error: {line_err}")
                continue

        full_text = "\n".join(full_text_lines)
        if full_text_lines:
            logger.debug(
                f"OCR [LocalCpu] crop '{source_crop}' ({variant_label}): {len(boxes)} line(s) recognized: "
                f"{' | '.join(full_text_lines[:3])}{'...' if len(full_text_lines) > 3 else ''}"
            )
        return OcrResult(
            engine="local_cpu",
            model_version=self.MODEL_VERSION,
            lines=boxes,
            full_text=full_text,
            source_crop=source_crop,
            preprocessing_variant=variant_label,
        )


class PaddleOcrEngine(IOcrEngine):
    """
    Local OCR engine using pretrained PaddleOCR text detection and recognition.
    Runs CPU-first on Apple Silicon M1 using PP-OCRv5 or PP-OCRv6 pretrained models.
    Does NOT silently fall back to TrOCR. Marks health degraded if unavailable.
    """

    _shared_instances: dict[str, Any] = {}

    def __init__(
        self,
        lang: str = "en",
        use_angle_cls: bool = False,
        ocr_version: str | None = None,
    ):
        self.lang = lang
        self.use_angle_cls = use_angle_cls
        self.ocr_version = ocr_version or getattr(settings, "PADDLE_OCR_VERSION", "PP-OCRv5")
        self.MODEL_VERSION = f"PaddleOCR-{self.ocr_version}"
        self._ocr = None
        self._is_ready = False
        self._error: Optional[str] = None
        self._initialized = False

    def check_readiness(self) -> tuple[bool, str | None]:
        """Verify PaddleOCR import and inference readiness without silent fallback."""
        if not self._initialized:
            self._ensure_loaded()
        return self._is_ready, self._error

    def get_health(self) -> dict[str, Any]:
        """Return engine operational status and readiness metadata."""
        ready, err = self.check_readiness()
        return {
            "engine": "paddleocr",
            "model_version": self.ocr_version,
            "runtime": "Apple Silicon CPU (paddlepaddle)",
            "readiness": "ready" if ready else "degraded",
            "error": err,
        }

    def _ensure_loaded(self):
        """Load and verify PaddleOCR once into memory without silent TrOCR fallback."""
        if self._initialized:
            return
        self._initialized = True

        # Reuse existing singleton instance if available
        if self.ocr_version in PaddleOcrEngine._shared_instances:
            self._ocr = PaddleOcrEngine._shared_instances[self.ocr_version]
            self._is_ready = True
            self._error = None
            return

        # Ensure environment variables are configured for sandbox and local cache
        settings.init_environment_variables()

        try:
            import paddle  # Check if paddlepaddle is installed
            from paddleocr import PaddleOCR

            # Initialize lightweight/mobile OCR pipeline on CPU with configured version
            # First attempt PaddleOCR 2.x standard parameters (use_angle_cls, use_gpu=False)
            try:
                self._ocr = PaddleOCR(
                    lang=self.lang,
                    use_angle_cls=self.use_angle_cls,
                    use_gpu=False,
                    show_log=False,
                )
            except (TypeError, ValueError):
                # PaddleOCR 3.x parameter signature fallback
                self._ocr = PaddleOCR(
                    lang=self.lang,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=self.use_angle_cls,
                    ocr_version=self.ocr_version,
                )

            # Test inference on a tiny blank image to verify execution
            dummy = np.ones((64, 64, 3), dtype=np.uint8) * 255
            _ = self._ocr.ocr(dummy)
            PaddleOcrEngine._shared_instances[self.ocr_version] = self._ocr
            self._is_ready = True
            self._error = None
            logger.info(
                f"PaddleOCR ({self.ocr_version}) initialized and verified successfully on CPU."
            )
        except Exception as e:
            self._is_ready = False
            self._error = str(e)
            logger.error(
                f"PaddleOCR is unavailable on this environment ({e}). "
                "Marking OCR engine as degraded. Silent TrOCR fallback is disabled."
            )
            # If tar corruption happened (e.g. unexpected end of data / zlib error), clean up partial cache
            if "unexpected end of data" in str(e).lower() or "zlib" in str(e).lower():
                import shutil
                paddle_home = os.path.expanduser("~/.paddleocr")
                if os.path.exists(paddle_home):
                    shutil.rmtree(paddle_home, ignore_errors=True)

    def extract_text(
        self,
        image_input: Any,
        variant_label: str = "original",
        source_crop: str | None = None,
    ) -> OcrResult:
        """
        Run text detection and recognition on input image.
        Returns recognized text lines, bounding boxes, and model confidence scores.
        Prevents silent fallback to unintended engines.
        """
        self._ensure_loaded()
        if not self._is_ready:
            raise RuntimeError(
                f"PaddleOCR is degraded and unavailable: {self._error}. "
                "Silent processing with an unintended engine is prevented."
            )

        if isinstance(image_input, Image.Image):
            rgb_arr = np.array(image_input.convert("RGB"))
            bgr_image = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)
        elif isinstance(image_input, np.ndarray):
            if len(image_input.shape) == 2:
                bgr_image = cv2.cvtColor(image_input, cv2.COLOR_GRAY2BGR)
            elif image_input.shape[2] == 4:
                bgr_image = cv2.cvtColor(image_input, cv2.COLOR_BGRA2BGR)
            else:
                bgr_image = image_input
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        orig_h, orig_w = bgr_image.shape[:2]
        max_dim = 800
        scale = 1.0
        infer_image = bgr_image
        if max(orig_h, orig_w) > max_dim:
            scale = max_dim / float(max(orig_h, orig_w))
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))
            infer_image = cv2.resize(bgr_image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        try:
            # PaddleOCR accepts BGR numpy arrays
            raw_output = self._ocr.ocr(infer_image)
        except Exception as e:
            self._is_ready = False
            self._error = str(e)
            logger.error(f"PaddleOCR inference failed ({e}). Silent fallback is disabled.")
            raise RuntimeError(
                f"PaddleOCR inference failed: {e}. "
                "Silent processing with an unintended engine is prevented."
            )

        boxes: list[OcrTextBox] = []
        full_text_lines: list[str] = []

        if raw_output and isinstance(raw_output, list):
            # Normal output: list of pages/results, e.g. [ [ [box, (text, score)], ... ] ]
            page_results = raw_output[0] if (len(raw_output) > 0 and isinstance(raw_output[0], list)) else raw_output
            for item in page_results:
                if not item:
                    continue
                # Standard format: [ [[x1,y1],[x2,y2],[x3,y3],[x4,y4]], (text, score) ]
                if isinstance(item, (list, tuple)) and len(item) == 2:
                    poly_box, text_info = item
                    if isinstance(text_info, (list, tuple)) and len(text_info) == 2:
                        txt = str(text_info[0]).strip()
                        raw_conf = float(text_info[1])
                    else:
                        txt = str(text_info).strip()
                        raw_conf = 0.80

                    # Convert box coordinates back to original image space
                    clean_quad: list[list[int]] = []
                    if isinstance(poly_box, (list, tuple)):
                        for pt in poly_box:
                            ox = int(round(pt[0] / scale))
                            oy = int(round(pt[1] / scale))
                            # Clamp to original image bounds
                            ox = max(0, min(orig_w, ox))
                            oy = max(0, min(orig_h, oy))
                            clean_quad.append([ox, oy])

                    if txt:
                        boxes.append(
                            OcrTextBox(
                                text=txt,
                                score=round(raw_conf, 4),
                                box=clean_quad,
                                source_crop=source_crop,
                                preprocessing_variant=variant_label,
                            )
                        )
                        full_text_lines.append(txt)

                # Dict format from paddlex pipeline
                elif isinstance(item, dict):
                    rec_texts = item.get("rec_texts", [])
                    rec_scores = item.get("rec_scores", [])
                    dt_polys = item.get("dt_polys", [])
                    rec_boxes = item.get("rec_boxes", [])
                    for i, txt in enumerate(rec_texts):
                        txt_str = str(txt).strip()
                        score = float(rec_scores[i]) if i < len(rec_scores) else 0.80
                        box_quad: list[list[int]] = []
                        if i < len(dt_polys) and dt_polys[i] is not None:
                            poly = dt_polys[i]
                            if hasattr(poly, "tolist"):
                                poly = poly.tolist()
                            box_quad = [[int(round(pt[0])), int(round(pt[1]))] for pt in poly]
                        elif i < len(rec_boxes) and rec_boxes[i] is not None:
                            rb = rec_boxes[i]
                            if hasattr(rb, "tolist"):
                                rb = rb.tolist()
                            if len(rb) == 4:
                                x1, y1, x2, y2 = rb
                                box_quad = [
                                    [int(x1), int(y1)],
                                    [int(x2), int(y1)],
                                    [int(x2), int(y2)],
                                    [int(x1), int(y2)],
                                ]
                        if txt_str:
                            boxes.append(
                                OcrTextBox(
                                    text=txt_str,
                                    score=round(score, 4),
                                    box=box_quad,
                                    source_crop=source_crop,
                                    preprocessing_variant=variant_label,
                                )
                            )
                            full_text_lines.append(txt_str)

        full_text = "\n".join(full_text_lines)

        return OcrResult(
            engine="paddleocr",
            model_version=self.MODEL_VERSION,
            lines=boxes,
            full_text=full_text,
            source_crop=source_crop,
            preprocessing_variant=variant_label,
        )


class TesseractOcrEngine(IOcrEngine):
    """
    Local secondary OCR adapter using Tesseract for clean machine overlays.
    Extracts text and bounding boxes when Tesseract is available.
    """

    MODEL_VERSION = "tesseract-ocr-v5"

    def __init__(self, psm: int = 6):
        self.psm = psm
        self._available: bool | None = None

    def get_health(self) -> dict[str, Any]:
        avail = self.is_available()
        return {
            "engine": "tesseract",
            "model_version": self.MODEL_VERSION,
            "runtime": "system binary",
            "readiness": "ready" if avail else "degraded",
            "error": None if avail else "Tesseract binary not found",
        }

    def is_available(self) -> bool:
        """Check if tesseract binary or pytesseract library is accessible."""
        if self._available is None:
            # Check for tesseract binary in PATH or typical locations
            bin_path = shutil.which("tesseract")
            if bin_path:
                self._available = True
            else:
                for cand in ("/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract", "/usr/bin/tesseract"):
                    if os.path.exists(cand):
                        self._available = True
                        break
                if self._available is None:
                    self._available = False
        return self._available

    def extract_text(
        self,
        image_input: Any,
        variant_label: str = "original",
        source_crop: str | None = None,
    ) -> OcrResult:
        """
        Run Tesseract on high-contrast crop. Returns structured lines and confidences.
        """
        if isinstance(image_input, np.ndarray):
            pil_img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
        elif isinstance(image_input, Image.Image):
            pil_img = image_input.convert("RGB")
        else:
            raise ValueError(f"Unsupported image input type: {type(image_input)}")

        if not self.is_available():
            logger.debug("Tesseract binary not found on system; returning empty result.")
            return OcrResult(
                engine="tesseract",
                model_version=self.MODEL_VERSION,
                lines=[],
                full_text="",
                source_crop=source_crop,
                preprocessing_variant=variant_label,
            )

        try:
            import pytesseract
            data = pytesseract.image_to_data(
                pil_img,
                config=f"--psm {self.psm}",
                output_type=pytesseract.Output.DICT,
            )

            boxes: list[OcrTextBox] = []
            full_lines: list[str] = []
            current_line_words: list[str] = []
            current_line_confs: list[float] = []
            current_line_boxes: list[list[int]] = []

            n_boxes = len(data["text"])
            for i in range(n_boxes):
                w_text = data["text"][i].strip()
                conf = float(data["conf"][i])
                if conf > 0 and w_text:
                    current_line_words.append(w_text)
                    current_line_confs.append(conf / 100.0)
                    bx, by, bw, bh = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
                    current_line_boxes.append([bx, by, bx + bw, by + bh])

                # Check line breaks (block_num, par_num, line_num transition or end)
                is_last = (i == n_boxes - 1)
                next_line = (i < n_boxes - 1 and data["line_num"][i+1] != data["line_num"][i])
                if (next_line or is_last) and current_line_words:
                    line_str = " ".join(current_line_words)
                    avg_conf = sum(current_line_confs) / float(len(current_line_confs))
                    min_x = min(b[0] for b in current_line_boxes)
                    min_y = min(b[1] for b in current_line_boxes)
                    max_x = max(b[2] for b in current_line_boxes)
                    max_y = max(b[3] for b in current_line_boxes)

                    quad = [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]]
                    boxes.append(
                        OcrTextBox(
                            text=line_str,
                            score=round(avg_conf, 4),
                            box=quad,
                            source_crop=source_crop,
                            preprocessing_variant=variant_label,
                        )
                    )
                    full_lines.append(line_str)
                    current_line_words = []
                    current_line_confs = []
                    current_line_boxes = []

            return OcrResult(
                engine="tesseract",
                model_version=self.MODEL_VERSION,
                lines=boxes,
                full_text="\n".join(full_lines),
                source_crop=source_crop,
                preprocessing_variant=variant_label,
            )

        except Exception as e:
            logger.warning(f"Tesseract OCR failed: {e}")
            return OcrResult(
                engine="tesseract",
                model_version=self.MODEL_VERSION,
                lines=[],
                full_text="",
                source_crop=source_crop,
                preprocessing_variant=variant_label,
            )


class MockOcrEngine(IOcrEngine):
    """Deterministic OCR engine for unit and integration testing."""

    MODEL_VERSION = "mock-ocr-v2"

    def get_health(self) -> dict[str, Any]:
        return {
            "engine": "mock-ocr",
            "model_version": self.MODEL_VERSION,
            "runtime": "mock",
            "readiness": "ready",
            "error": None,
        }

    def __init__(self, predefined_lines: list[OcrTextBox] | None = None):
        self.predefined_lines = predefined_lines

    def extract_text(
        self,
        image_input: Any,
        variant_label: str = "original",
        source_crop: str | None = None,
    ) -> OcrResult:
        if self.predefined_lines is not None:
            full = "\n".join(line.text for line in self.predefined_lines)
            return OcrResult(
                engine="mock-ocr",
                model_version=self.MODEL_VERSION,
                lines=self.predefined_lines,
                full_text=full,
                source_crop=source_crop,
                preprocessing_variant=variant_label,
            )

        sample_lines = [
            OcrTextBox(
                text="AWB NO: 19106721621035",
                score=0.9782,
                box=[[10, 10], [250, 10], [250, 30], [10, 30]],
                source_crop=source_crop,
                preprocessing_variant=variant_label,
            ),
            OcrTextBox(
                text="LENGTH: 75.300 CM",
                score=0.9614,
                box=[[10, 40], [200, 40], [200, 60], [10, 60]],
                source_crop=source_crop,
                preprocessing_variant=variant_label,
            ),
            OcrTextBox(
                text="WIDTH: 49.200 CM",
                score=0.9589,
                box=[[10, 70], [190, 70], [190, 90], [10, 90]],
                source_crop=source_crop,
                preprocessing_variant=variant_label,
            ),
            OcrTextBox(
                text="HEIGHT: 44.400 CM",
                score=0.9521,
                box=[[10, 100], [195, 100], [195, 120], [10, 120]],
                source_crop=source_crop,
                preprocessing_variant=variant_label,
            ),
            OcrTextBox(
                text="WEIGHT: 14060.000 GM",
                score=0.9437,
                box=[[10, 130], [210, 130], [210, 150], [10, 150]],
                source_crop=source_crop,
                preprocessing_variant=variant_label,
            ),
        ]
        full_text = "\n".join(l.text for l in sample_lines)
        return OcrResult(
            engine="mock-ocr",
            model_version=self.MODEL_VERSION,
            lines=sample_lines,
            full_text=full_text,
            source_crop=source_crop,
            preprocessing_variant=variant_label,
        )


def get_ocr_engine(engine_name: str | None = None) -> IOcrEngine:
    """Factory helper to obtain the configured OCR engine."""
    choice = (engine_name or settings.OCR_ENGINE).lower()
    if settings.MOCK_INFERENCE or choice == "mock":
        return MockOcrEngine()
    elif choice in ("local_cpu", "trocr"):
        return LocalCpuOcrEngine()
    elif choice == "tesseract":
        return TesseractOcrEngine()
    elif choice in ("paddleocr", "auto", "default"):
        return PaddleOcrEngine()
    else:
        logger.warning(f"Unknown OCR engine '{choice}', defaulting to paddleocr")
        return PaddleOcrEngine()


# Backward compatibility aliases
TrOcrLocalEngine = LocalCpuOcrEngine
