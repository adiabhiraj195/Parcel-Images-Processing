"""Pydantic schemas for batch creation, retrieval, and status."""

from datetime import datetime
from typing import List, Optional
from pydantic import BaseModel, Field, ConfigDict

from app.schemas.image import FileUploadItem, PresignedUploadItem
from app.schemas.extraction import ExtractionResponse


class BatchCreateRequest(BaseModel):
    """Payload to create a new batch and request upload targets."""

    name: str = Field(..., min_length=1, max_length=255)
    files: List[FileUploadItem] = Field(..., min_length=1, max_length=500)


class BatchCreateResponse(BaseModel):
    """Response containing batch ID and presigned upload targets."""

    batch_id: str
    name: str
    total_images: int
    uploads: List[PresignedUploadItem]


class BatchResponse(BaseModel):
    """Batch summary details and processing progress."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    name: str
    total_images: int
    uploaded_images: int
    processed_images: int
    failed_images: int
    created_at: datetime
    updated_at: datetime


class ActiveProcessingItem(BaseModel):
    """Details of the currently actively processing image in the batch."""

    model_config = ConfigDict(from_attributes=True)

    image_id: str
    filename: str
    stage: str
    stage_name: str
    stage_message: str
    stage_progress: int
    stage_index: int
    total_stages: int = 6
    stage_started_at: Optional[datetime] = None
    preview_url: Optional[str] = None


class BatchProgressItem(BaseModel):
    """Per-image live status item in progress endpoint."""

    model_config = ConfigDict(from_attributes=True)

    image_id: str
    filename: str
    processing_status: str
    current_stage: Optional[str] = None
    stage_message: Optional[str] = None
    stage_progress: Optional[int] = None
    stage_started_at: Optional[datetime] = None
    preview_url: Optional[str] = None


class BatchProgressResponse(BaseModel):
    """Lightweight real-time processing progress snapshot."""

    batch: BatchResponse
    is_processing: bool
    active_image: Optional[ActiveProcessingItem] = None
    images: List[BatchProgressItem]


class BatchResultsResponse(BaseModel):
    """Batch results summary and rows."""

    batch: BatchResponse
    results: List[ExtractionResponse]
    total_results: int
    active_processing: Optional[ActiveProcessingItem] = None
