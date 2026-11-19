"""Image SQLAlchemy ORM model."""

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, Integer, DateTime, ForeignKey
from sqlalchemy.orm import relationship

from app.database import Base


def utc_now():
    return datetime.now(timezone.utc)


class Image(Base):
    __tablename__ = "images"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    batch_id = Column(String(36), ForeignKey("batches.id", ondelete="CASCADE"), nullable=False, index=True)
    filename = Column(String(255), nullable=False)
    content_type = Column(String(100), nullable=False)
    file_size = Column(Integer, nullable=False)
    s3_bucket = Column(String(255), nullable=False)
    s3_key = Column(String(1024), nullable=False)

    # Status tracking
    upload_status = Column(String(50), default="PENDING_UPLOAD", nullable=False)  # PENDING_UPLOAD, UPLOADED, FAILED
    processing_status = Column(String(50), default="PENDING", nullable=False)    # PENDING, QUEUED, PROCESSING, COMPLETED, FAILED
    current_stage = Column(String(50), default="PENDING", nullable=False)        # QUEUED, INGESTING, PREPROCESSING, SCENE_DETECTION, METADATA_EXTRACTION, LABEL_EXTRACTION, RECONCILIATION, COMPLETED, FAILED
    stage_message = Column(String(255), default="Waiting in queue", nullable=False)
    stage_progress = Column(Integer, default=0, nullable=False)
    stage_started_at = Column(DateTime(timezone=True), default=utc_now, nullable=True)

    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    # Relationships
    batch = relationship("Batch", back_populates="images")
    extraction = relationship("Extraction", back_populates="image", uselist=False, cascade="all, delete-orphan")
