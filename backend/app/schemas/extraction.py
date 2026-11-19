"""Pydantic schemas for extraction records, status taxonomy, and reviews."""

import json
from datetime import datetime
from typing import Optional, Any, Dict, List
from pydantic import BaseModel, Field, ConfigDict, model_validator

# Extraction status taxonomy
VALID_EXTRACTION_STATUSES = {
    "SUCCESS",
    "PARTIAL_EXTRACTION",
    "LOW_CONFIDENCE",
    "SOURCE_CONFLICT",
    "PROCESSING_FAILED",
}

# Scene flags (independent of extraction status)
VALID_SCENE_FLAGS = {
    "NO_PARCEL",
    "MULTIPLE_PARCELS",
    "PARTIAL_PARCEL",
    "LABEL_NOT_FOUND",
    "LABEL_OCCLUDED",
    "LABEL_UNREADABLE",
}

# Combined valid statuses for backwards compatibility
VALID_STATUSES = VALID_EXTRACTION_STATUSES | VALID_SCENE_FLAGS

# Explicit parcel and shipping-label detection outcome taxonomy
VALID_PARCEL_LABEL_OUTCOMES = {
    "NO_PARCEL_DETECTED",
    "PARCEL_DETECTED_LABEL_NOT_FOUND",
    "LABEL_DETECTED_UNREADABLE",
    "LABEL_FOUND",
    "MULTIPLE_PARCELS",
    "LOW_CONFIDENCE_REVIEW",
}


class ExtractionResponse(BaseModel):
    """Extraction record returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    image_id: str
    filename: Optional[str] = None
    tracking_number: Optional[str] = None
    weight_value: Optional[float] = None
    weight_unit: Optional[str] = None
    actual_weight: Optional[float] = None
    volumetric_weight: Optional[float] = None
    length: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    dimension_unit: Optional[str] = None
    parcel_count: Optional[int] = None
    status: str
    parcel_label_outcome: Optional[str] = None
    scene_flags: List[str] = Field(default_factory=list)
    raw_ocr_json: Optional[str] = None
    barcode_json: Optional[str] = None
    detection_json: Optional[str] = None
    field_confidence_json: Optional[str] = None
    evidence_json: Optional[str] = None
    reviewed: bool
    reviewed_at: Optional[datetime] = None
    processing_status: Optional[str] = "COMPLETED"
    current_stage: Optional[str] = None
    stage_message: Optional[str] = None
    stage_progress: Optional[int] = None
    stage_started_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
    preview_url: Optional[str] = None

    @model_validator(mode="before")
    @classmethod
    def populate_evidence_fields(cls, data: Any) -> Any:
        if isinstance(data, dict):
            ev_str = data.get("evidence_json")
            if ev_str:
                try:
                    ev = json.loads(ev_str) if isinstance(ev_str, str) else ev_str
                    if isinstance(ev, dict):
                        if not data.get("scene_flags") and "scene_flags" in ev:
                            data["scene_flags"] = ev["scene_flags"]
                        if not data.get("parcel_label_outcome") and "parcel_label_outcome" in ev:
                            data["parcel_label_outcome"] = ev["parcel_label_outcome"]
                        if data.get("actual_weight") is None and "actual_weight" in ev:
                            data["actual_weight"] = ev["actual_weight"]
                        elif data.get("actual_weight") is None and "weight_provenance" in ev:
                            data["actual_weight"] = ev["weight_provenance"].get("value")
                        if data.get("volumetric_weight") is None and "volumetric_weight" in ev:
                            data["volumetric_weight"] = ev["volumetric_weight"]
                        elif data.get("volumetric_weight") is None and "volumetric_weight_provenance" in ev:
                            data["volumetric_weight"] = ev["volumetric_weight_provenance"].get("value")
                except Exception:
                    pass
        return data


class ExtractionReviewUpdateRequest(BaseModel):
    """Operator manual correction payload."""

    tracking_number: Optional[str] = None
    weight_value: Optional[float] = None
    weight_unit: Optional[str] = None
    actual_weight: Optional[float] = None
    volumetric_weight: Optional[float] = None
    length: Optional[float] = None
    width: Optional[float] = None
    height: Optional[float] = None
    dimension_unit: Optional[str] = None
    parcel_count: Optional[int] = None
    status: Optional[str] = None
    parcel_label_outcome: Optional[str] = None
    scene_flags: Optional[List[str]] = None
    reviewed: bool = True
