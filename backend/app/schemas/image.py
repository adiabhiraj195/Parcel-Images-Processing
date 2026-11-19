"""Pydantic schemas for images and upload parameters."""

from datetime import datetime
from typing import Optional, Dict
from pydantic import BaseModel, Field, field_validator, ConfigDict


class FileUploadItem(BaseModel):
    """File metadata supplied by the frontend before upload."""

    filename: str = Field(..., min_length=1, max_length=255)
    content_type: str = Field(..., min_length=3, max_length=100)
    file_size: int = Field(..., gt=0, le=25 * 1024 * 1024)  # Maximum 25MB

    @field_validator("content_type")
    @classmethod
    def validate_content_type(cls, v: str) -> str:
        clean = v.lower().strip()
        if clean not in ("image/jpeg", "image/jpg", "image/png"):
            raise ValueError("Only JPG and PNG images are supported.")
        return clean

    @field_validator("filename")
    @classmethod
    def sanitize_filename(cls, v: str) -> str:
        import os
        # Strip directories to prevent path traversal
        clean = os.path.basename(v).strip()
        if not clean:
            raise ValueError("Filename cannot be empty.")
        return clean


class PresignedUploadItem(BaseModel):
    """Presigned upload coordinates returned to the browser."""

    image_id: str
    filename: str
    content_type: str
    file_size: int
    upload_url: str
    required_headers: Dict[str, str] = Field(default_factory=dict)
    s3_key: str


class ImageResponse(BaseModel):
    """Image representation returned by the API."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    batch_id: str
    filename: str
    content_type: str
    file_size: int
    s3_bucket: str
    s3_key: str
    upload_status: str
    processing_status: str
    current_stage: Optional[str] = None
    stage_message: Optional[str] = None
    stage_progress: Optional[int] = None
    stage_started_at: Optional[datetime] = None
    created_at: datetime
    updated_at: datetime
