"""Extraction SQLAlchemy ORM model enforcing one-result-per-image invariant."""

import uuid
from datetime import datetime, timezone
from sqlalchemy import (
    Column,
    String,
    Float,
    Integer,
    Boolean,
    DateTime,
    ForeignKey,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.database import Base


def utc_now():
    return datetime.now(timezone.utc)


class Extraction(Base):
    __tablename__ = "extractions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    image_id = Column(
        String(36),
        ForeignKey("images.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
        index=True,
    )

    # Core logistics fields (null when absent, never invented)
    tracking_number = Column(String(100), nullable=True)
    weight_value = Column(Float, nullable=True)
    weight_unit = Column(String(20), nullable=True)
    length = Column(Float, nullable=True)
    width = Column(Float, nullable=True)
    height = Column(Float, nullable=True)
    dimension_unit = Column(String(20), nullable=True)
    parcel_count = Column(Integer, nullable=True)

    # Status from exact taxonomy
    status = Column(String(50), nullable=False, default="PROCESSING_FAILED")

    # Explicit parcel and shipping-label detection outcome
    parcel_label_outcome = Column(String(50), nullable=True)

    # Structured JSON evidence (stored as JSON string for audit/debug)
    raw_ocr_json = Column(Text, nullable=True)
    barcode_json = Column(Text, nullable=True)
    detection_json = Column(Text, nullable=True)
    field_confidence_json = Column(Text, nullable=True)
    evidence_json = Column(Text, nullable=True)

    # Operator review state
    reviewed = Column(Boolean, default=False, nullable=False)
    reviewed_at = Column(DateTime(timezone=True), nullable=True)

    created_at = Column(DateTime(timezone=True), default=utc_now, nullable=False)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False)

    # Relationships
    image = relationship("Image", back_populates="extraction")

    __table_args__ = (
        UniqueConstraint("image_id", name="uq_extractions_image_id"),
    )
