"""Abstract Base Classes and Data Schemas for the modular CV, OCR, and parsing pipeline."""

from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel, Field


class DetectionBox(BaseModel):
    """Normalized bounding box in pixel coordinates."""

    class_name: str
    score: float
    bbox_xyxy: list[int]  # [x_min, y_min, x_max, y_max]


class DetectionResult(BaseModel):
    """Normalized detection model output."""

    model: str
    detections: list[DetectionBox]


class OcrTextBox(BaseModel):
    """Normalized OCR detected text line."""

    text: str
    score: float
    box: list[list[int]] = Field(default_factory=list)  # Quad corners [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
    source_crop: str | None = None  # e.g., "full_image", "overlay_top_left", "label_crop"
    preprocessing_variant: str | None = None  # e.g., "original", "upscaled_2x", "clahe", "threshold"


class OcrResult(BaseModel):
    """Full raw OCR output."""

    engine: str
    model_version: str | None = None
    lines: list[OcrTextBox] = Field(default_factory=list)
    full_text: str = ""
    source_crop: str | None = None
    preprocessing_variant: str | None = None


class BarcodeResult(BaseModel):
    """Barcode or QR code decoding output."""

    raw_value: str
    symbology: str  # e.g. CODE_128, QR_CODE
    bbox_xyxy: list[int] | None = None
    source_crop: str | None = None  # "full_image" or "label_crop"


class FieldProvenance(BaseModel):
    """Audit provenance for an individual extracted logistics field."""

    value: Any = None
    source: str = ""  # "scanner_metadata", "printed_barcode", "label_ocr"
    raw_text: str = ""
    ocr_confidence: float = 0.0
    bounding_box: list[int] | None = None
    evidence_reference: str | None = None
    parser_rule: str = ""
    unit: str | None = None
    normalized_value: Any = None
    conflict_status: str | None = None  # None, "AGREED", "CONFLICT"
    raw_candidates: list[str] = Field(default_factory=list)


# Extraction status taxonomy
STATUS_SUCCESS = "SUCCESS"
STATUS_PARTIAL_EXTRACTION = "PARTIAL_EXTRACTION"
STATUS_LOW_CONFIDENCE = "LOW_CONFIDENCE"
STATUS_SOURCE_CONFLICT = "SOURCE_CONFLICT"
STATUS_PROCESSING_FAILED = "PROCESSING_FAILED"

EXTRACTION_STATUSES = {
    STATUS_SUCCESS,
    STATUS_PARTIAL_EXTRACTION,
    STATUS_LOW_CONFIDENCE,
    STATUS_SOURCE_CONFLICT,
    STATUS_PROCESSING_FAILED,
}

# Parcel and Physical Label Detection Outcome Taxonomy
OUTCOME_NO_PARCEL_DETECTED = "NO_PARCEL_DETECTED"
OUTCOME_PARCEL_DETECTED_LABEL_NOT_FOUND = "PARCEL_DETECTED_LABEL_NOT_FOUND"
OUTCOME_LABEL_DETECTED_UNREADABLE = "LABEL_DETECTED_UNREADABLE"
OUTCOME_LABEL_FOUND = "LABEL_FOUND"
OUTCOME_MULTIPLE_PARCELS = "MULTIPLE_PARCELS"
OUTCOME_LOW_CONFIDENCE_REVIEW = "LOW_CONFIDENCE_REVIEW"

PARCEL_LABEL_OUTCOMES = {
    OUTCOME_NO_PARCEL_DETECTED,
    OUTCOME_PARCEL_DETECTED_LABEL_NOT_FOUND,
    OUTCOME_LABEL_DETECTED_UNREADABLE,
    OUTCOME_LABEL_FOUND,
    OUTCOME_MULTIPLE_PARCELS,
    OUTCOME_LOW_CONFIDENCE_REVIEW,
}

# Scene flags (independent of extraction status)
SCENE_FLAG_NO_PARCEL = "NO_PARCEL"
SCENE_FLAG_MULTIPLE_PARCELS = "MULTIPLE_PARCELS"
SCENE_FLAG_PARTIAL_PARCEL = "PARTIAL_PARCEL"
SCENE_FLAG_LABEL_NOT_FOUND = "LABEL_NOT_FOUND"
SCENE_FLAG_LABEL_OCCLUDED = "LABEL_OCCLUDED"
SCENE_FLAG_LABEL_UNREADABLE = "LABEL_UNREADABLE"

SCENE_FLAGS = {
    SCENE_FLAG_NO_PARCEL,
    SCENE_FLAG_MULTIPLE_PARCELS,
    SCENE_FLAG_PARTIAL_PARCEL,
    SCENE_FLAG_LABEL_NOT_FOUND,
    SCENE_FLAG_LABEL_OCCLUDED,
    SCENE_FLAG_LABEL_UNREADABLE,
}


class ExtractedLogisticsData(BaseModel):
    """Logistics fields parsed from OCR & barcode evidence with provenance."""

    tracking_number: str | None = None
    weight_value: float | None = None
    weight_unit: str | None = None
    actual_weight: float | None = None
    actual_weight_unit: str | None = None
    volumetric_weight: float | None = None
    volumetric_weight_unit: str | None = None
    length: float | None = None
    width: float | None = None
    height: float | None = None
    dimension_unit: str | None = None
    volume: float | None = None
    location: str | None = None
    timestamp: str | None = None
    machine_id: str | None = None
    sorter_id: str | None = None
    parcel_count: int | None = None
    status: str = STATUS_PROCESSING_FAILED
    parcel_label_outcome: str | None = None
    parcel_boxes: list[DetectionBox] = Field(default_factory=list)
    label_boxes: list[DetectionBox] = Field(default_factory=list)
    scene_flags: list[str] = Field(default_factory=list)
    provenance: dict[str, FieldProvenance] = Field(default_factory=dict)
    field_confidence: dict[str, float] = Field(default_factory=dict)
    evidence: dict[str, Any] = Field(default_factory=dict)


class IImagePreprocessor(ABC):
    """Interface for image validation, orientation correction, and cropping."""

    @abstractmethod
    def validate_and_load(self, file_path: str) -> Any:
        pass

    @abstractmethod
    def safe_crop(
        self,
        image: Any,
        bbox_xyxy: list[int],
        padding_ratio: float = 0.05,
    ) -> tuple[Any, list[int]]:
        pass


class IObjectDetector(ABC):
    """Interface for local zero-shot parcel and label detector."""

    @abstractmethod
    def detect(self, image_input: Any) -> DetectionResult:
        pass


class IOcrEngine(ABC):
    """Interface for local text detection and recognition."""

    @abstractmethod
    def extract_text(self, image_input: Any, variant_label: str = "original") -> OcrResult:
        pass


class IBarcodeDecoder(ABC):
    """Interface for local barcode and QR decoding."""

    @abstractmethod
    def decode(self, image_input: Any) -> list[BarcodeResult]:
        pass


class IFieldParser(ABC):
    """Interface for deterministic regex parsing of logistics fields."""

    @abstractmethod
    def parse(
        self,
        ocr_result: OcrResult,
        barcode_results: list[BarcodeResult],
        detection_result: DetectionResult,
    ) -> ExtractedLogisticsData:
        pass
