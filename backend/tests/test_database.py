"""Database schema and constraint unit tests."""

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import IntegrityError

from app.database import Base
from app.models.batch import Batch
from app.models.image import Image
from app.models.extraction import Extraction


@pytest.fixture
def test_db():
    """In-memory SQLite session with foreign keys enabled."""
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    
    # Enable foreign keys for in-memory SQLite
    with engine.connect() as conn:
        conn.exec_driver_sql("PRAGMA foreign_keys=ON")

    Base.metadata.create_all(bind=engine)
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.close()


def test_unique_constraint_on_extractions_image_id(test_db):
    """Enforce the non-negotiable invariant: at most one extraction per image."""
    batch = Batch(name="Test Batch", total_images=1)
    test_db.add(batch)
    test_db.commit()

    image = Image(
        batch_id=batch.id,
        filename="parcel_1.jpg",
        content_type="image/jpeg",
        file_size=1024,
        s3_bucket="test-bucket",
        s3_key="batches/test/originals/img1/parcel_1.jpg",
    )
    test_db.add(image)
    test_db.commit()

    # First extraction record
    ext1 = Extraction(
        image_id=image.id,
        tracking_number="TRACK123",
        status="SUCCESS",
    )
    test_db.add(ext1)
    test_db.commit()

    # Attempting to insert a second extraction for the same image_id MUST fail
    ext2 = Extraction(
        image_id=image.id,
        tracking_number="TRACK456",
        status="SUCCESS",
    )
    test_db.add(ext2)
    with pytest.raises(IntegrityError):
        test_db.commit()


def test_batch_cascade_delete(test_db):
    """Deleting a batch should cascade delete images and extractions."""
    batch = Batch(name="Delete Me Batch", total_images=1)
    test_db.add(batch)
    test_db.commit()

    image = Image(
        batch_id=batch.id,
        filename="p.jpg",
        content_type="image/jpeg",
        file_size=500,
        s3_bucket="b",
        s3_key="k",
    )
    test_db.add(image)
    test_db.commit()

    ext = Extraction(image_id=image.id, status="SUCCESS")
    test_db.add(ext)
    test_db.commit()

    test_db.delete(batch)
    test_db.commit()

    assert test_db.query(Image).filter(Image.id == image.id).first() is None
    assert test_db.query(Extraction).filter(Extraction.id == ext.id).first() is None
