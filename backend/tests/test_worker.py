"""Integration tests for the per-image pipeline orchestrator and worker queue."""

import os
import tempfile
import pytest
from unittest.mock import MagicMock
from PIL import Image

from app.database import Base, SessionLocal, engine
from app.models.batch import Batch
from app.models.image import Image as ImageModel
from app.models.extraction import Extraction
from app.pipeline.interfaces import (
    DetectionBox,
    DetectionResult,
    OcrTextBox,
    OcrResult,
    BarcodeResult,
)
from app.pipeline.detector import MockObjectDetector
from app.pipeline.ocr import MockOcrEngine
from app.pipeline.barcode import MockBarcodeDecoder
from app.pipeline.parser import DeterministicFieldParser
from app.pipeline.preprocessor import OpenCVPreprocessor
from app.worker.orchestrator import PipelineOrchestrator


@pytest.fixture
def test_db():
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture
def mock_orchestrator():
    """Pipeline orchestrator configured with fast deterministic mocks."""
    det = MockObjectDetector(
        predefined_detections=[
            DetectionBox(class_name="parcel", score=0.95, bbox_xyxy=[10, 10, 200, 200]),
            DetectionBox(class_name="shipping_label", score=0.91, bbox_xyxy=[30, 30, 150, 150]),
        ]
    )
    ocr = MockOcrEngine(
        predefined_lines=[
            OcrTextBox(text="AWB: 1Z1234567890123456", score=0.98, box=[[0, 0], [1, 0], [1, 1], [0, 1]]),
            OcrTextBox(text="WT: 3.2 KG", score=0.95, box=[[0, 0], [1, 0], [1, 1], [0, 1]]),
            OcrTextBox(text="DIMS: 25x15x10 cm", score=0.92, box=[[0, 0], [1, 0], [1, 1], [0, 1]]),
        ]
    )
    barcode = MockBarcodeDecoder(
        predefined_results=[
            BarcodeResult(raw_value="1Z1234567890123456", symbology="Code128"),
        ]
    )
    return PipelineOrchestrator(
        preprocessor=OpenCVPreprocessor(),
        detector=det,
        ocr_engine=ocr,
        barcode_decoder=barcode,
        parser=DeterministicFieldParser(),
    )


def test_orchestrator_successful_image(test_db, mock_orchestrator, monkeypatch):
    """End-to-end processing of a valid image yielding SUCCESS extraction."""
    # Create temp image file
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.close(fd)
    img = Image.new("RGB", (300, 300), color="blue")
    img.save(path, "JPEG")

    # Mock S3 download to copy local path
    import shutil
    from app.services.s3_service import s3_service
    monkeypatch.setattr(s3_service, "download_to_file", lambda key, dest: shutil.copy(path, dest))

    batch = Batch(name="Worker Test Batch", total_images=1)
    test_db.add(batch)
    test_db.commit()

    image = ImageModel(
        batch_id=batch.id,
        filename="test_scan.jpg",
        content_type="image/jpeg",
        file_size=os.path.getsize(path),
        s3_bucket="test-b",
        s3_key="batches/b/originals/i/test_scan.jpg",
    )
    test_db.add(image)
    test_db.commit()

    try:
        ext = mock_orchestrator.process_image(test_db, image.id)

        assert ext is not None
        assert ext.tracking_number == "1Z1234567890123456"
        assert ext.weight_value == 3.2
        assert ext.weight_unit == "kg"
        assert (ext.length, ext.width, ext.height) == (25.0, 15.0, 10.0)
        assert ext.status == "SUCCESS"
        assert image.processing_status == "COMPLETED"

        # Verify debug bundle and provenance in evidence_json
        import json
        ev = json.loads(ext.evidence_json)
        assert "debug_bundle" in ev
        assert "tracking_provenance" in ev
        assert "scene_flags" in ev

        # Verify exactly one extraction row exists
        count = test_db.query(Extraction).filter(Extraction.image_id == image.id).count()
        assert count == 1
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_orchestrator_handles_corrupt_image_with_processing_failed(test_db, mock_orchestrator, monkeypatch):
    """Corrupt image must not crash batch; must create extraction row with PROCESSING_FAILED."""
    fd, path = tempfile.mkstemp(suffix=".jpg")
    os.write(fd, b"CORRUPT_BYTES")
    os.close(fd)

    import shutil
    from app.services.s3_service import s3_service
    monkeypatch.setattr(s3_service, "download_to_file", lambda key, dest: shutil.copy(path, dest))

    batch = Batch(name="Corrupt Image Batch", total_images=1)
    test_db.add(batch)
    test_db.commit()

    image = ImageModel(
        batch_id=batch.id,
        filename="corrupt.jpg",
        content_type="image/jpeg",
        file_size=100,
        s3_bucket="b",
        s3_key="k",
    )
    test_db.add(image)
    test_db.commit()

    try:
        ext = mock_orchestrator.process_image(test_db, image.id)

        assert ext.status == "PROCESSING_FAILED"
        assert ext.tracking_number is None
        assert ext.weight_value is None
        assert image.processing_status == "FAILED"

        # Invariant check: exactly one row persisted
        count = test_db.query(Extraction).filter(Extraction.image_id == image.id).count()
        assert count == 1
    finally:
        if os.path.exists(path):
            os.remove(path)


def test_batch_reconciliation(test_db, mock_orchestrator):
    """Reconciliation step must ensure every image has an extraction record."""
    batch = Batch(name="Reconciliation Batch", total_images=2)
    test_db.add(batch)
    test_db.commit()

    # Image with no extraction
    orphan_image = ImageModel(
        batch_id=batch.id,
        filename="orphan.jpg",
        content_type="image/jpeg",
        file_size=500,
        s3_bucket="b",
        s3_key="k",
        processing_status="PENDING",
    )
    test_db.add(orphan_image)
    test_db.commit()

    assert orphan_image.extraction is None

    # Run reconciliation
    mock_orchestrator.reconcile_batch(test_db, batch.id)

    test_db.refresh(orphan_image)
    assert orphan_image.extraction is not None
    assert orphan_image.extraction.status == "PROCESSING_FAILED"
