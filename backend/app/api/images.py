"""Image detail, manual review corrections, and retry endpoints."""

import json
from datetime import datetime, timezone
from typing import Optional, List, Tuple
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.image import Image
from app.models.extraction import Extraction
from app.schemas.extraction import ExtractionResponse, ExtractionReviewUpdateRequest, VALID_STATUSES
from app.services.s3_service import s3_service

router = APIRouter(prefix="/api/images", tags=["Images"])


def extract_scene_flags(evidence_json_str: Optional[str]) -> List[str]:
    """Helper to safely extract scene flags list from evidence_json."""
    if not evidence_json_str:
        return []
    try:
        ev = json.loads(evidence_json_str) if isinstance(evidence_json_str, str) else evidence_json_str
        if isinstance(ev, dict) and "scene_flags" in ev and isinstance(ev["scene_flags"], list):
            return ev["scene_flags"]
    except Exception:
        pass
    return []


def extract_weight_details(evidence_json_str: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """Helper to extract actual and volumetric weight from evidence_json."""
    if not evidence_json_str:
        return None, None
    try:
        ev = json.loads(evidence_json_str) if isinstance(evidence_json_str, str) else evidence_json_str
        if isinstance(ev, dict):
            act = ev.get("actual_weight")
            if act is None and "weight_provenance" in ev and isinstance(ev["weight_provenance"], dict):
                act = ev["weight_provenance"].get("value")
            vol = ev.get("volumetric_weight")
            if vol is None and "volumetric_weight_provenance" in ev and isinstance(ev["volumetric_weight_provenance"], dict):
                vol = ev["volumetric_weight_provenance"].get("value")
            return act, vol
    except Exception:
        pass
    return None, None


@router.get("/{image_id}", response_model=ExtractionResponse)
def get_image_details(image_id: str, db: Session = Depends(get_db)) -> ExtractionResponse:
    """Retrieve full extraction record and short-lived preview URL for an image."""
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image with ID '{image_id}' not found.",
        )

    preview_url = s3_service.create_presigned_get_url(image.s3_key)

    if not image.extraction:
        return ExtractionResponse(
            id=f"pending-{image.id}",
            image_id=image.id,
            filename=image.filename,
            status="PROCESSING_FAILED" if image.processing_status == "FAILED" else "LOW_CONFIDENCE",
            scene_flags=[],
            reviewed=False,
            processing_status=image.processing_status,
            current_stage=image.current_stage,
            stage_message=image.stage_message,
            stage_progress=image.stage_progress,
            stage_started_at=image.stage_started_at,
            created_at=image.created_at,
            updated_at=image.updated_at,
            preview_url=preview_url,
        )

    ext = image.extraction
    scene_flags = extract_scene_flags(ext.evidence_json)
    act_wt, vol_wt = extract_weight_details(ext.evidence_json)

    return ExtractionResponse(
        id=ext.id,
        image_id=image.id,
        filename=image.filename,
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
        processing_status=image.processing_status,
        current_stage=image.current_stage or "COMPLETED",
        stage_message=image.stage_message,
        stage_progress=image.stage_progress,
        stage_started_at=image.stage_started_at,
        created_at=ext.created_at,
        updated_at=ext.updated_at,
        preview_url=preview_url,
    )


@router.patch("/{image_id}/review", response_model=ExtractionResponse)
def review_image_extraction(
    image_id: str,
    payload: ExtractionReviewUpdateRequest,
    db: Session = Depends(get_db),
) -> ExtractionResponse:
    """
    Allow warehouse operator to manually correct extracted fields.
    Updates the existing extraction record and marks it reviewed.
    """
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image with ID '{image_id}' not found.",
        )

    ext = image.extraction
    if not ext:
        # Create an extraction record if one didn't exist yet
        ext = Extraction(
            image_id=image.id,
            status=payload.status or "SUCCESS",
            reviewed=True,
            reviewed_at=datetime.now(timezone.utc),
        )
        db.add(ext)

    if payload.tracking_number is not None:
        ext.tracking_number = payload.tracking_number
    if payload.weight_value is not None:
        ext.weight_value = payload.weight_value
    if payload.weight_unit is not None:
        ext.weight_unit = payload.weight_unit
    if payload.length is not None:
        ext.length = payload.length
    if payload.width is not None:
        ext.width = payload.width
    if payload.height is not None:
        ext.height = payload.height
    if payload.dimension_unit is not None:
        ext.dimension_unit = payload.dimension_unit
    if payload.parcel_count is not None:
        ext.parcel_count = payload.parcel_count

    if payload.parcel_label_outcome is not None:
        ext.parcel_label_outcome = payload.parcel_label_outcome

    if payload.status:
        if payload.status not in VALID_STATUSES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid status '{payload.status}'. Must be one of {sorted(VALID_STATUSES)}",
            )
        ext.status = payload.status

    if payload.scene_flags is not None:
        try:
            ev = json.loads(ext.evidence_json) if ext.evidence_json else {}
            ev["scene_flags"] = payload.scene_flags
            ext.evidence_json = json.dumps(ev)
        except Exception:
            pass

    ext.reviewed = payload.reviewed
    ext.reviewed_at = datetime.now(timezone.utc)

    db.commit()
    db.refresh(ext)

    scene_flags = extract_scene_flags(ext.evidence_json)
    act_wt, vol_wt = extract_weight_details(ext.evidence_json)
    preview_url = s3_service.create_presigned_get_url(image.s3_key)
    return ExtractionResponse(
        id=ext.id,
        image_id=image.id,
        filename=image.filename,
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
        processing_status=image.processing_status,
        current_stage=image.current_stage or "COMPLETED",
        stage_message=image.stage_message,
        stage_progress=image.stage_progress,
        stage_started_at=image.stage_started_at,
        created_at=ext.created_at,
        updated_at=ext.updated_at,
        preview_url=preview_url,
    )


@router.post("/{image_id}/retry", response_model=ExtractionResponse)
async def retry_image_processing(image_id: str, db: Session = Depends(get_db)) -> ExtractionResponse:
    """Reset image status to QUEUED and re-enqueue to local worker."""
    image = db.query(Image).filter(Image.id == image_id).first()
    if not image:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Image with ID '{image_id}' not found.",
        )

    image.processing_status = "QUEUED"
    image.current_stage = "QUEUED"
    image.stage_message = "Queued for reprocessing..."
    image.stage_progress = 0
    image.stage_started_at = datetime.now(timezone.utc)
    db.commit()

    from app.worker.broadcaster import pipeline_broadcaster, PipelineStage, STAGE_METADATA, TOTAL_PIPELINE_STEPS
    preview_url = s3_service.create_presigned_get_url(image.s3_key)
    pipeline_broadcaster.broadcast(
        image.batch_id,
        {
            "type": "stage_update",
            "batch_id": image.batch_id,
            "image_id": image.id,
            "filename": image.filename,
            "stage": PipelineStage.QUEUED,
            "stage_name": STAGE_METADATA[PipelineStage.QUEUED]["name"],
            "stage_message": "Queued for reprocessing...",
            "stage_progress": 0,
            "stage_index": 0,
            "total_stages": TOTAL_PIPELINE_STEPS,
            "preview_url": preview_url,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
    )

    from app.worker.queue import worker_queue
    await worker_queue.enqueue_image(image.id, image.batch_id)

    return get_image_details(image_id=image_id, db=db)
