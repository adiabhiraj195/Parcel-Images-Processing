"""Pretrained zero-shot object detector adapter using OWLv2 on Apple Silicon CPU."""

import torch  # Ensure PyTorch is imported before Paddle to prevent macOS tensor collision
from typing import Any
import numpy as np
import cv2
from PIL import Image

from app.config import settings
from app.pipeline.interfaces import DetectionBox, DetectionResult, IObjectDetector


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
    """Compute Intersection over Minimum area (containment ratio)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    min_area = min(area1, area2)
    if min_area <= 0:
        return 0.0
    return inter_area / float(min_area)


def non_max_suppression(
    boxes: list[DetectionBox],
    iou_threshold: float = 0.5,
    containment_threshold: float = 0.70,
) -> list[DetectionBox]:
    """
    Deduplicate overlapping or contained detections of the same canonical class,
    retaining the highest confidence score.
    """
    if not boxes:
        return []

    # Sort boxes descending by score
    sorted_boxes = sorted(boxes, key=lambda b: b.score, reverse=True)
    kept: list[DetectionBox] = []

    for current in sorted_boxes:
        overlap = False
        for chosen in kept:
            # Only suppress boxes belonging to the same category
            if current.class_name == chosen.class_name:
                iou = compute_iou(current.bbox_xyxy, chosen.bbox_xyxy)
                containment = compute_containment(current.bbox_xyxy, chosen.bbox_xyxy)
                if iou > iou_threshold or containment > containment_threshold:
                    overlap = True
                    break
        if not overlap:
            kept.append(current)

    return kept


def check_edge_proximity(
    bbox: list[int],
    image_width: int,
    image_height: int,
    margin_px: int = 15,
) -> tuple[bool, list[str]]:
    """
    Check if a parcel box is close to or touches image borders.
    Returns (is_near_edge, touched_edges_list).
    Note: Edge proximity is treated as evidence, not automatic proof of partial parcel.
    """
    x1, y1, x2, y2 = bbox
    touched = []
    if x1 <= margin_px:
        touched.append("left")
    if y1 <= margin_px:
        touched.append("top")
    if x2 >= (image_width - margin_px):
        touched.append("right")
    if y2 >= (image_height - margin_px):
        touched.append("bottom")
    return bool(touched), touched


def filter_parcels_in_roi(
    detections: list[DetectionBox],
    scanner_bed_roi: list[int] | None = None,
    min_overlap_ratio: float = 0.20,
) -> tuple[list[DetectionBox], list[DetectionBox]]:
    """
    Filter parcel detections into active scan ROI vs background parcels.
    Returns (active_parcels, background_parcels).
    """
    if not scanner_bed_roi:
        parcels = [d for d in detections if d.class_name == "parcel"]
        return parcels, []

    rx1, ry1, rx2, ry2 = scanner_bed_roi
    active: list[DetectionBox] = []
    background: list[DetectionBox] = []

    for d in detections:
        if d.class_name != "parcel":
            continue
        bx1, by1, bx2, by2 = d.bbox_xyxy
        inter_x1 = max(rx1, bx1)
        inter_y1 = max(ry1, by1)
        inter_x2 = min(rx2, bx2)
        inter_y2 = min(ry2, by2)

        if inter_x2 > inter_x1 and inter_y2 > inter_y1:
            inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
            box_area = max(1, (bx2 - bx1) * (by2 - by1))
            overlap_ratio = inter_area / float(box_area)
            if overlap_ratio >= min_overlap_ratio:
                active.append(d)
                continue
        background.append(d)

    return active, background


class OwlV2ZeroShotDetector(IObjectDetector):
    """
    Zero-shot parcel and shipping-label detector using Hugging Face OWLv2.
    Loads once on CPU to protect Apple Silicon memory.
    """

    MODEL_ID = "google/owlv2-base-patch16-ensemble"

    # Candidate text prompts grouped by canonical class
    PROMPT_MAPPINGS: dict[str, str] = {
        "parcel": "parcel",
        "cardboard box": "parcel",
        "shipping package": "parcel",
        "courier parcel": "parcel",
        "plastic mailer": "parcel",
        "courier bag": "parcel",
        "shipping bag": "parcel",
        "poly mailer": "parcel",
        "shipping box": "parcel",
        "carton box": "parcel",
        "shipping label": "shipping_label",
        "shipping sticker": "shipping_label",
        "waybill label": "shipping_label",
        "barcode label": "shipping_label",
        "address label": "shipping_label",
        "parcel label": "shipping_label",
        "white label sticker": "shipping_label",
    }

    def __init__(self, score_threshold: float = 0.18, iou_threshold: float = 0.45):
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold
        self._processor = None
        self._model = None
        self._text_prompts = list(self.PROMPT_MAPPINGS.keys())

    def get_health(self) -> dict[str, Any]:
        return {
            "engine": "owlv2",
            "model_version": self.MODEL_ID,
            "runtime": "Apple Silicon CPU (PyTorch)",
            "readiness": "ready",
            "error": None,
        }

    def _ensure_loaded(self):
        """Lazy-load weights once into CPU memory."""
        if self._model is None or self._processor is None:
            import transformers
            from transformers import Owlv2ForObjectDetection, Owlv2Processor

            transformers.logging.set_verbosity_error()

            cache_path = getattr(settings, "resolved_model_cache_dir", settings.MODEL_CACHE_DIR)
            try:
                self._processor = Owlv2Processor.from_pretrained(
                    self.MODEL_ID,
                    cache_dir=cache_path,
                    local_files_only=True,
                )
                self._model = Owlv2ForObjectDetection.from_pretrained(
                    self.MODEL_ID,
                    cache_dir=cache_path,
                    local_files_only=True,
                )
            except Exception:
                self._processor = Owlv2Processor.from_pretrained(
                    self.MODEL_ID,
                    cache_dir=cache_path,
                )
                self._model = Owlv2ForObjectDetection.from_pretrained(
                    self.MODEL_ID,
                    cache_dir=cache_path,
                )
            self._model.to("cpu")
            self._model.eval()

    def detect(self, pil_image: Image.Image) -> DetectionResult:
        """
        Run zero-shot inference on input image and return normalized detections.
        Applies aspect-ratio preserving downscaling and adaptive contrast boost
        for dark scanner images while keeping coordinates exact in original pixel space.
        """
        self._ensure_loaded()
        orig_w, orig_h = pil_image.size

        # 1. Bounded downscale preserving aspect ratio (max dimension 1024)
        max_dim = 1024
        scale = min(1.0, max_dim / float(max(orig_w, orig_h)))
        infer_w = int(orig_w * scale)
        infer_h = int(orig_h * scale)

        resized_pil = pil_image.resize((infer_w, infer_h), Image.Resampling.BILINEAR)

        # 2. Adaptive contrast enhancement for dark warehouse scanner images
        gray = np.array(resized_pil.convert("L"))
        mean_bright = float(np.mean(gray))

        if mean_bright < 35.0:
            inv_gamma = 0.5
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
            img_np = np.array(resized_pil)
            if len(img_np.shape) == 3:
                lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
                l_ch, a_ch, b_ch = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
                l_ch = clahe.apply(cv2.LUT(l_ch, table))
                img_np = cv2.cvtColor(cv2.merge((l_ch, a_ch, b_ch)), cv2.COLOR_LAB2RGB)
            else:
                img_np = cv2.LUT(img_np, table)
            infer_pil = Image.fromarray(img_np)
        else:
            infer_pil = resized_pil

        # Format input for OWLv2
        text_queries = [self._text_prompts]
        inputs = self._processor(
            text=text_queries,
            images=infer_pil,
            return_tensors="pt",
        )

        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process outputs to infer_w, infer_h dimensions
        target_sizes = torch.tensor([[infer_h, infer_w]])
        results = self._processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.score_threshold,
            text_labels=text_queries,
        )[0]

        raw_detections: list[DetectionBox] = []
        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        text_labels = results.get("text_labels")
        labels = results["labels"].cpu().numpy() if "labels" in results else None

        for idx, (box, score) in enumerate(zip(boxes, scores)):
            if text_labels and idx < len(text_labels) and text_labels[idx] is not None:
                prompt_text = text_labels[idx]
            elif labels is not None and idx < len(labels):
                prompt_text = self._text_prompts[labels[idx]]
            else:
                prompt_text = "unknown"

            canonical_class = self.PROMPT_MAPPINGS.get(prompt_text, "unknown")

            x1, y1, x2, y2 = box.tolist()
            # Scale coordinates back to original image space
            orig_x1 = max(0, min(orig_w, int(round(x1 / scale))))
            orig_y1 = max(0, min(orig_h, int(round(y1 / scale))))
            orig_x2 = max(0, min(orig_w, int(round(x2 / scale))))
            orig_y2 = max(0, min(orig_h, int(round(y2 / scale))))

            box_w = orig_x2 - orig_x1
            box_h = orig_y2 - orig_y1
            area = box_w * box_h
            total_area = orig_w * orig_h

            # Filter degenerate zero-area boxes
            if box_w <= 1 or box_h <= 1:
                continue

            # Filter false-positive edge slivers / tiny speckles
            if canonical_class == "parcel":
                if area < (0.005 * total_area):
                    continue
                aspect = box_w / float(max(1, box_h))
                # Reject extreme aspect ratio horizontal/vertical slivers with low height
                if (aspect > 6.0 or aspect < 0.16) and (box_h < 0.08 * orig_h or box_w < 0.08 * orig_w):
                    continue
            elif canonical_class == "shipping_label":
                if area < (0.0008 * total_area):
                    continue

            raw_detections.append(
                DetectionBox(
                    class_name=canonical_class,
                    score=round(float(score), 4),
                    bbox_xyxy=[orig_x1, orig_y1, orig_x2, orig_y2],
                )
            )

        # Apply NMS deduplication
        deduped = non_max_suppression(raw_detections, self.iou_threshold)

        return DetectionResult(
            model=self.MODEL_ID,
            detections=deduped,
        )


class MockObjectDetector(IObjectDetector):
    """Deterministic mock detector for unit tests and offline testing."""

    def __init__(self, predefined_detections: list[DetectionBox] | None = None):
        self.predefined_detections = predefined_detections

    def get_health(self) -> dict[str, Any]:
        return {
            "engine": "mock-detector",
            "model_version": "mock-detector-v1",
            "runtime": "mock",
            "readiness": "ready",
            "error": None,
        }

    def detect(self, pil_image: Image.Image) -> DetectionResult:
        w, h = pil_image.size
        if self.predefined_detections is not None:
            return DetectionResult(model="mock-detector", detections=self.predefined_detections)

        # Default standard single parcel with centered label
        default_boxes = [
            DetectionBox(
                class_name="parcel",
                score=0.92,
                bbox_xyxy=[int(w * 0.1), int(h * 0.1), int(w * 0.9), int(h * 0.9)],
            ),
            DetectionBox(
                class_name="shipping_label",
                score=0.88,
                bbox_xyxy=[int(w * 0.3), int(h * 0.3), int(w * 0.7), int(h * 0.7)],
            ),
        ]
        return DetectionResult(model="mock-detector-v1", detections=default_boxes)
