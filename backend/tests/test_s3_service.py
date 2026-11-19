"""Unit tests for S3Service using moto mocked S3."""

import boto3
import pytest
from moto import mock_aws

from app.config import settings
from app.services.s3_service import S3Service, sanitize_filename


@pytest.fixture
def mocked_s3():
    """Mock AWS S3 context and create the test bucket."""
    with mock_aws():
        client = boto3.client("s3", region_name="us-east-1")
        client.create_bucket(Bucket="test-parcel-bucket")
        yield client


def test_sanitize_filename():
    """Filename sanitization must prevent path traversal and strip dangerous characters."""
    assert sanitize_filename("../../../etc/passwd") == "passwd"
    assert sanitize_filename("scan #1 (box).png") == "scan__1__box_.png"
    assert sanitize_filename("normal_file.jpg") == "normal_file.jpg"
    assert sanitize_filename("") == "unnamed_image.jpg"


def test_s3_key_generation():
    """Verify standard S3 object key layout."""
    service = S3Service()
    batch_id = "batch-123"
    image_id = "img-456"

    orig_key = service.generate_original_key(batch_id, image_id, "label.jpg")
    assert orig_key == "batches/batch-123/originals/img-456/label.jpg"

    derived_key = service.generate_derived_key(batch_id, image_id, "label-crop.jpg")
    assert derived_key == "batches/batch-123/derived/img-456/label-crop.jpg"

    export_key = service.generate_export_key(batch_id, "exp-789")
    assert export_key == "batches/batch-123/exports/exp-789.csv"


def test_presigned_urls_and_verification(mocked_s3, monkeypatch):
    """Test generating presigned PUT and GET URLs and verifying uploaded objects."""
    monkeypatch.setattr(settings, "AWS_S3_BUCKET", "test-parcel-bucket")
    service = S3Service()
    service.bucket = "test-parcel-bucket"
    service.client = mocked_s3

    s3_key = "batches/b1/originals/i1/test.jpg"
    put_url, headers = service.create_presigned_put_url(s3_key, "image/jpeg")

    assert "test-parcel-bucket" in put_url
    assert headers["Content-Type"] == "image/jpeg"

    # Before upload, verify_object_exists returns False
    exists, size, ctype = service.verify_object_exists(s3_key)
    assert exists is False

    # Simulate browser upload via S3 put_object
    mocked_s3.put_object(
        Bucket="test-parcel-bucket",
        Key=s3_key,
        Body=b"fake-image-bytes",
        ContentType="image/jpeg",
    )

    # After upload, verify_object_exists returns True with size and content-type
    exists, size, ctype = service.verify_object_exists(s3_key)
    assert exists is True
    assert size == len(b"fake-image-bytes")
    assert ctype == "image/jpeg"

    # Test presigned GET preview URL
    get_url = service.create_presigned_get_url(s3_key)
    assert "test-parcel-bucket" in get_url
