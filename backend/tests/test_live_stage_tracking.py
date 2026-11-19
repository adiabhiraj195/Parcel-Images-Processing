"""Tests for real-time pipeline stage tracking, progress endpoint, and SSE broadcasting."""

import json
import pytest
from fastapi.testclient import TestClient

from app.database import Base, SessionLocal, engine
from app.models.batch import Batch
from app.models.image import Image as ImageModel
from app.models.extraction import Extraction
from app.worker.broadcaster import pipeline_broadcaster, PipelineStage, STAGE_METADATA


@pytest.fixture
def test_db():
    Base.metadata.create_all(bind=engine)
    session = SessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_batch_progress_and_active_stage_api(client: TestClient, test_db):
    """Test progress endpoint returns currently processing image and stages."""
    batch = Batch(name="Live Tracking Batch", total_images=2, uploaded_images=2)
    test_db.add(batch)
    test_db.commit()

    img1 = ImageModel(
        batch_id=batch.id,
        filename="scanner_parcel_01.jpg",
        content_type="image/jpeg",
        file_size=120000,
        s3_bucket="test-bucket",
        s3_key="batches/b/originals/i1/scanner_parcel_01.jpg",
        upload_status="UPLOADED",
        processing_status="PROCESSING",
        current_stage="SCENE_DETECTION",
        stage_message="Branch C: Running zero-shot parcel detection (OWLv2)...",
        stage_progress=50,
    )
    img2 = ImageModel(
        batch_id=batch.id,
        filename="scanner_parcel_02.jpg",
        content_type="image/jpeg",
        file_size=150000,
        s3_bucket="test-bucket",
        s3_key="batches/b/originals/i2/scanner_parcel_02.jpg",
        upload_status="UPLOADED",
        processing_status="QUEUED",
        current_stage="QUEUED",
        stage_message="Waiting in processing queue...",
        stage_progress=0,
    )
    test_db.add(img1)
    test_db.add(img2)
    test_db.commit()

    # 1. Fetch batch progress
    prog_res = client.get(f"/api/batches/{batch.id}/progress")
    assert prog_res.status_code == 200
    prog_data = prog_res.json()

    assert prog_data["is_processing"] is True
    assert prog_data["active_image"] is not None
    assert prog_data["active_image"]["image_id"] == img1.id
    assert prog_data["active_image"]["filename"] == "scanner_parcel_01.jpg"
    assert prog_data["active_image"]["stage"] == "SCENE_DETECTION"
    assert prog_data["active_image"]["stage_progress"] == 50
    assert len(prog_data["images"]) == 2

    # 2. Fetch batch results
    results_res = client.get(f"/api/batches/{batch.id}/results")
    assert results_res.status_code == 200
    results_data = results_res.json()

    assert results_data["active_processing"] is not None
    assert results_data["active_processing"]["image_id"] == img1.id
    assert results_data["active_processing"]["stage"] == "SCENE_DETECTION"
    assert len(results_data["results"]) == 2

    # Check stage details on pending results
    res_img1 = next(r for r in results_data["results"] if r["image_id"] == img1.id)
    assert res_img1["processing_status"] == "PROCESSING"
    assert res_img1["current_stage"] == "SCENE_DETECTION"


def test_sse_events_stream_initial_snapshot(client: TestClient, test_db):
    """Test SSE stream connects and sends initial snapshot."""
    batch = Batch(name="SSE Batch", total_images=1, uploaded_images=1)
    test_db.add(batch)
    test_db.commit()

    img = ImageModel(
        batch_id=batch.id,
        filename="sse_test.jpg",
        content_type="image/jpeg",
        file_size=80000,
        s3_bucket="b",
        s3_key="k",
        upload_status="UPLOADED",
        processing_status="PROCESSING",
        current_stage="METADATA_EXTRACTION",
        stage_message="Extracting machine overlay...",
        stage_progress=70,
    )
    test_db.add(img)
    test_db.commit()

    # Stream SSE and read the first event line
    with client.stream("GET", f"/api/batches/{batch.id}/events") as response:
        assert response.status_code == 200
        assert "text/event-stream" in response.headers["content-type"]
        for line in response.iter_lines():
            if line and line.startswith("data: "):
                data_json = json.loads(line[6:])
                assert data_json["type"] == "initial_state"
                assert data_json["batch_id"] == batch.id
                assert data_json["is_processing"] is True
                assert data_json["active_image"]["stage"] == "METADATA_EXTRACTION"
                break
