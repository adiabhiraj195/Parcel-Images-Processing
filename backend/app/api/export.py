"""Deterministic CSV export endpoint."""

import csv
import io
import json
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.database import get_db
from app.models.batch import Batch
from app.models.image import Image

router = APIRouter(prefix="/api/batches", tags=["Export"])

CSV_COLUMNS = [
    "image_id",
    "image_name",
    "tracking_number",
    "weight_value",
    "weight_unit",
    "length",
    "width",
    "height",
    "dimension_unit",
    "volume",
    "location",
    "timestamp",
    "parcel_count",
    "status",
    "parcel_label_outcome",
    "reviewed",
]


@router.get("/{batch_id}/export.csv")
def export_batch_csv(batch_id: str, db: Session = Depends(get_db)):
    """
    Export batch results as a CSV with exactly one row per accepted image.
    Never invents missing values; empty values remain empty strings.
    """
    batch = db.query(Batch).filter(Batch.id == batch_id).first()
    if not batch:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Batch with ID '{batch_id}' not found.",
        )

    # Order images deterministically by created_at then id
    images = db.query(Image).filter(Image.batch_id == batch_id).order_by(Image.created_at, Image.id).all()

    output = io.StringIO()
    writer = csv.writer(output, dialect="excel")
    writer.writerow(CSV_COLUMNS)

    for img in images:
        ext = img.extraction
        if ext:
            evidence = {}
            if ext.evidence_json:
                try:
                    evidence = json.loads(ext.evidence_json)
                except Exception:
                    pass
            loc = evidence.get("location", "")
            vol = str(evidence.get("volume", "")) if evidence.get("volume") is not None else ""
            tstamp = evidence.get("timestamp", "")
            row = [
                img.id,
                img.filename,
                ext.tracking_number if ext.tracking_number is not None else "",
                str(ext.weight_value) if ext.weight_value is not None else "",
                ext.weight_unit if ext.weight_unit is not None else "",
                str(ext.length) if ext.length is not None else "",
                str(ext.width) if ext.width is not None else "",
                str(ext.height) if ext.height is not None else "",
                ext.dimension_unit if ext.dimension_unit is not None else "",
                vol,
                loc,
                tstamp,
                str(ext.parcel_count) if ext.parcel_count is not None else "",
                ext.status,
                ext.parcel_label_outcome or "",
                "true" if ext.reviewed else "false",
            ]
        else:
            # Fallback row if processing has not completed yet
            status_val = "PROCESSING_FAILED" if img.processing_status == "FAILED" else "QUEUED"
            row = [
                img.id,
                img.filename,
                "",  # tracking_number
                "",  # weight_value
                "",  # weight_unit
                "",  # length
                "",  # width
                "",  # height
                "",  # dimension_unit
                "",  # volume
                "",  # location
                "",  # timestamp
                "",  # parcel_count
                status_val,
                "",  # parcel_label_outcome
                "false",  # reviewed
            ]
        writer.writerow(row)

    csv_content = output.getvalue()
    filename = f"batch_{batch_id}_results.csv"

    return Response(
        content=csv_content.encode("utf-8"),
        media_type="text/csv; charset=utf-8",
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
        },
    )
