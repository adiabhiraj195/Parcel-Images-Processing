"""Batch management, creation, and result endpoints."""

import uuid
import json
import logging
from datetime import datetime, timezone
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from sqlalchemy import desc

from app.database import get_db
from app.models.batch import Batch
from app.models.image import Image
from app.schemas.batch import (
    BatchCreateRequest,
    BatchCreateResponse,
    BatchResponse,
    BatchResultsResponse,
    ActiveProcessingItem,
    BatchProgressItem,
    BatchProgressResponse,
)
from app.schemas.image import PresignedUploadItem
from app.schemas.extraction import ExtractionResponse
from app.services.s3_service import s3_service
from app.worker.queue import worker_queue
from app.worker.broadcaster import (
    pipeline_broadcaster,
    PipelineStage,
    STAGE_METADATA,
    TOTAL_PIPELINE_STEPS,
)
from app.api.images import extract_scene_flags, extract_weight_details

router = APIRouter(prefix="/api/batches", tags=["Batches"])
logger = logging.getLogger("api")


def get_active_processing_item(batch_id: str, db: Session) -> Optional[ActiveProcessingItem]:
    """Retrieve details of the actively processing image for a batch."""
    cached = pipeline_broadcaster.get_active_stage(batch_id)
    if cached and cached.get("image_id"):
        return ActiveProcessingItem(
            image_id=cached["image_id"],
            filename=cached.get("filename", ""),
            stage=cached.get("stage", "PROCESSING"),
            stage_name=cached.get("stage_name", "Processing"),
            stage_message=cached.get("stage_message", "Processing image..."),
            stage_progress=cached.get("stage_progress", 0),
            stage_index=cached.get("stage_index", 0),
            total_stages=cached.get("total_stages", TOTAL_PIPELINE_STEPS),
            stage_started_at=cached.get("stage_started_at"),
            preview_url=cached.get("preview_url"),
        )

    # Fallback to query database for any image in PROCESSING status
    active_img = (
        db.query(Image)
        .filter(Image.batch_id == batch_id, Image.processing_status == "PROCESSING")
        .first()
    )
    if active_img:
        meta = STAGE_METADATA.get(active_img.current_stage or "", {})
        preview_url = None
        try:
            preview_url = s3_service.create_presigned_get_url(active_img.s3_key)
        except Exception:
            pass
        return ActiveProcessingItem(
            image_id=active_img.id,
            filename=active_img.filename,
            stage=active_img.current_stage or "PROCESSING",
            stage_name=meta.get("name", active_img.current_stage or "Processing"),
            stage_message=active_img.stage_message or meta.get("default_message", "Processing image..."),
            stage_progress=active_img.stage_progress if active_img.stage_progress else meta.get("progress", 50),
            stage_index=meta.get("step_index", 1),
            total_stages=TOTAL_PIPELINE_STEPS,
            stage_started_at=active_img.stage_started_at,
            preview_url=preview_url,
        )
    return None


@router.post("", response_model=BatchCreateResponse, status_code=status.HTTP_201_CREATED)
def create_batch(payload: BatchCreateRequest, db: Session = Depends(get_db)) -> BatchCreateResponse:
    """
    Create a new batch and register image records.
    Generates short-lived presigned PUT URLs for direct browser-to-S3 uploads.
    """
    batch_id = str(uuid.uuid4())
    batch = Batch(
        id=batch_id,
        name=payload.name,
        total_images=len(payload.files),
        uploaded_images=0,
        processed_images=0,
        failed_images=0,
    )
    db.add(batch)

    upload_items: List[PresignedUploadItem] = []
    for file_meta in payload.files:
        image_id = str(uuid.uuid4())
        s3_key = s3_service.generate_original_key(batch_id, image_id, file_meta.filename)

        upload_url, headers = s3_service.create_presigned_put_url(
            s3_key=s3_key,
            content_type=file_meta.content_type,
        )

        image = Image(
            id=image_id,
            batch_id=batch_id,
            filename=file_meta.filename,
            content_type=file_meta.content_type,
            file_size=file_meta.file_size,
            s3_bucket=s3_service.bucket,
            s3_key=s3_key,
            upload_status="PENDING_UPLOAD",
            processing_status="PENDING",
        )
        db.add(image)

        upload_items.append(
            PresignedUploadItem(
                image_id=image_id,
                filename=file_meta.filename,
                content_type=file_meta.content_type,
                file_size=file_meta.file_size,
                upload_url=upload_url,
                required_headers=headers,
                s3_key=s3_key,
            )
        )

    db.commit()

    return BatchCreateResponse(
        batch_id=batch_id,
        name=payload.name,
        total_images=len(payload.files),
        uploads=upload_items,
    )


@router.get("", response_model=List[BatchResponse])
def list_batches(db: Session = Depends(get_db)) -> List[BatchResponse]:
    """List all batches in reverse chronological order."""
    batches = db.query(Batch).order_by(desc(Batch.created_at)).all()
    return [BatchResponse.model_validate(b) for b in batches]


@router.get("/{batch_id}", response_model=BatchResponse)
def get_batch(batch_id: str, db: Session = Depends(get_db)) -> BatchResponse:
    """Get single batch progress and metadata."""
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch with ID '{batch_id}' not found.",
        )
    return BatchResponse.model_validate(batch)


@router.get("/{batch_id}/results", response_model=BatchResultsResponse)
def get_batch_results(batch_id: str, db: Session = Depends(get_db)) -> BatchResultsResponse:
    """Retrieve all extraction results for the batch."""
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch with ID '{batch_id}' not found.",
        )

    images = db.query(Image).filter(Image.batch_id == batch_id).all()
    results: List[ExtractionResponse] = []

    for img in images:
        if img.extraction:
            ext = img.extraction
            scene_flags = extract_scene_flags(ext.evidence_json)
            act_wt, vol_wt = extract_weight_details(ext.evidence_json)
            preview_url = s3_service.create_presigned_get_url(img.s3_key)
            resp = ExtractionResponse(
                id=ext.id,
                image_id=img.id,
                filename=img.filename,
                tracking_number=ext.tracking_number,
                weight_value=ext.weight_value,
                weight_unit=ext.weight_unit,
                actual_weight=act_wt if act_wt is not None else ext.weight_value,
                volumetric_weight=vol_wt,
                length=ext.length,
                width=ext.width,
                height=ext.height,
                dimension_unit=ext.dimension_unit,
                parcel_count=ext.parcel_count,
                status=ext.status,
                parcel_label_outcome=ext.parcel_label_outcome,
                scene_flags=scene_flags,
                raw_ocr_json=ext.raw_ocr_json,
                barcode_json=ext.barcode_json,
                detection_json=ext.detection_json,
                field_confidence_json=ext.field_confidence_json,
                evidence_json=ext.evidence_json,
                reviewed=ext.reviewed,
                reviewed_at=ext.reviewed_at,
                processing_status=img.processing_status,
                current_stage=img.current_stage or "COMPLETED",
                stage_message=img.stage_message,
                stage_progress=img.stage_progress,
                stage_started_at=img.stage_started_at,
                created_at=ext.created_at,
                updated_at=ext.updated_at,
                preview_url=preview_url,
            )
            results.append(resp)
        else:
            preview_url = s3_service.create_presigned_get_url(img.s3_key)
            status_str = (
                "PROCESSING_FAILED"
                if img.processing_status == "FAILED"
                else "LOW_CONFIDENCE"
            )
            results.append(
                ExtractionResponse(
                    id=f"pending-{img.id}",
                    image_id=img.id,
                    filename=img.filename,
                    status=status_str,
                    scene_flags=[],
                    reviewed=False,
                    processing_status=img.processing_status,
                    current_stage=img.current_stage or ("QUEUED" if img.processing_status == "QUEUED" else "PROCESSING"),
                    stage_message=img.stage_message or ("Waiting in queue..." if img.processing_status == "QUEUED" else "Processing..."),
                    stage_progress=img.stage_progress or 0,
                    stage_started_at=img.stage_started_at,
                    created_at=img.created_at,
                    updated_at=img.updated_at,
                    preview_url=preview_url,
                )
            )

    active_processing = get_active_processing_item(batch_id, db)

    return BatchResultsResponse(
        batch=BatchResponse.model_validate(batch),
        results=results,
        total_results=len(results),
        active_processing=active_processing,
    )


@router.get("/{batch_id}/progress", response_model=BatchProgressResponse)
def get_batch_progress(batch_id: str, db: Session = Depends(get_db)) -> BatchProgressResponse:
    """Lightweight real-time processing progress snapshot."""
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch with ID '{batch_id}' not found.",
        )

    images = db.query(Image).filter(Image.batch_id == batch_id).all()
    active_processing = get_active_processing_item(batch_id, db)
    is_processing = any(img.processing_status in ("QUEUED", "PROCESSING") for img in images)

    items: List[BatchProgressItem] = []
    for img in images:
        preview_url = None
        try:
            preview_url = s3_service.create_presigned_get_url(img.s3_key)
        except Exception:
            pass
        items.append(
            BatchProgressItem(
                image_id=img.id,
                filename=img.filename,
                processing_status=img.processing_status,
                current_stage=img.current_stage,
                stage_message=img.stage_message,
                stage_progress=img.stage_progress,
                stage_started_at=img.stage_started_at,
                preview_url=preview_url,
            )
        )

    return BatchProgressResponse(
        batch=BatchResponse.model_validate(batch),
        is_processing=is_processing,
        active_image=active_processing,
        images=items,
    )


@router.get("/{batch_id}/events")
async def batch_events_stream(batch_id: str, db: Session = Depends(get_db)):
    """
    Server-Sent Events (SSE) live stream for real-time stage updates.
    Enables zero-latency frontend visual updates as images process through the pipeline.
    """
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch with ID '{batch_id}' not found.",
        )

    async def sse_event_generator():
        # 1. Send initial state snapshot on connection
        try:
            active_item = get_active_processing_item(batch_id, db)
            images = db.query(Image).filter(Image.batch_id == batch_id).all()
            is_proc = any(img.processing_status in ("QUEUED", "PROCESSING") for img in images)

            initial_payload = {
                "type": "initial_state",
                "batch_id": batch_id,
                "is_processing": is_proc,
                "batch": {
                    "id": batch.id,
                    "name": batch.name,
                    "total_images": batch.total_images,
                    "uploaded_images": batch.uploaded_images,
                    "processed_images": batch.processed_images,
                    "failed_images": batch.failed_images,
                },
                "active_image": active_item.model_dump() if active_item else None,
                "images": [
                    {
                        "image_id": img.id,
                        "filename": img.filename,
                        "processing_status": img.processing_status,
                        "current_stage": img.current_stage,
                        "stage_message": img.stage_message,
                        "stage_progress": img.stage_progress,
                    }
                    for img in images
                ],
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            yield f"data: {json.dumps(initial_payload, default=str)}\n\n"
        except Exception as e:
            logger.error(f"Error preparing initial SSE snapshot for batch {batch_id}: {e}")

        # 2. Subscribe and stream live events
        async for event in pipeline_broadcaster.subscribe(batch_id):
            if event.get("type") == "heartbeat":
                yield ": keep-alive\n\n"
            else:
                yield f"data: {json.dumps(event, default=str)}\n\n"

    return StreamingResponse(
        sse_event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/{batch_id}/uploads/complete", response_model=BatchResponse)
async def complete_batch_uploads(
    batch_id: str,
    image_ids: Optional[List[str]] = None,
    db: Session = Depends(get_db),
) -> BatchResponse:
    """
    Verify uploaded objects in S3, transition status to UPLOADED / QUEUED,
    and schedule local processing.
    """
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch with ID '{batch_id}' not found.",
        )

    query = db.query(Image).filter(Image.batch_id == batch_id)
    if image_ids:
        query = query.filter(Image.id.in_(image_ids))

    images = query.all()
    verified_images = []

    for img in images:
        exists, size, _ = s3_service.verify_object_exists(img.s3_key)
        if exists:
            img.upload_status = "UPLOADED"
            img.processing_status = "QUEUED"
            img.current_stage = "QUEUED"
            img.stage_message = "Waiting in processing queue..."
            img.stage_progress = 0
            if size:
                img.file_size = size
            verified_images.append(img)
        else:
            img.upload_status = "FAILED"
            img.processing_status = "FAILED"
            img.current_stage = "FAILED"
            img.stage_message = "Upload verification failed in S3"
            img.stage_progress = 0

    total_uploaded = (
        db.query(Image)
        .filter(Image.batch_id == batch_id, Image.upload_status == "UPLOADED")
        .count()
    )
    batch.uploaded_images = total_uploaded
    db.commit()
    db.refresh(batch)

    logger.info(
        f"📤 [BATCH UPLOAD] Verified {len(verified_images)}/{len(images)} image(s) for batch '{batch.name}' ({batch_id}) -> Enqueuing for worker processing"
    )

    # Enqueue verified images for local worker processing
    for img in verified_images:
        await worker_queue.enqueue_image(img.id, batch_id)

    return BatchResponse.model_validate(batch)
