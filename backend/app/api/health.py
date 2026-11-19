"""Health check endpoints for operational monitoring."""

import platform
import sys
from typing import Optional
from fastapi import APIRouter
from pydantic import BaseModel

from app.config import settings
from app.pipeline.ocr import get_ocr_engine
from app.pipeline.detector import OwlV2ZeroShotDetector, MockObjectDetector
from app.pipeline.paddleocr_vl_adapter import PaddleOcrVlAdapter, MockPaddleOcrVlAdapter

router = APIRouter(tags=["Health"])


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str
    environment: str
    platform: str
    python_version: str


class ModelHealthDetail(BaseModel):
    engine: str
    model_version: str
    runtime: str
    readiness: str
    error: Optional[str] = None


class ModelsHealthResponse(BaseModel):
    status: str
    ocr: ModelHealthDetail
    detector: ModelHealthDetail
    paddleocr_vl: Optional[ModelHealthDetail] = None


@router.get("/health", response_model=HealthResponse)
@router.get("/api/health", response_model=HealthResponse)
def get_health() -> HealthResponse:
    """Return system operational status and runtime environment metadata."""
    return HealthResponse(
        status="ok",
        service="parcel-intelligence-backend",
        version="0.1.0",
        environment="development",
        platform=f"{platform.system()} {platform.machine()}",
        python_version=sys.version.split(" ")[0],
    )


@router.get("/health/models", response_model=ModelsHealthResponse)
@router.get("/api/health/models", response_model=ModelsHealthResponse)
def get_models_health() -> ModelsHealthResponse:
    """Return status, engine name, model version, runtime, and readiness for ML models."""
    ocr_engine = get_ocr_engine()
    ocr_health = (
        ocr_engine.get_health()
        if hasattr(ocr_engine, "get_health")
        else {
            "engine": "unknown",
            "model_version": getattr(ocr_engine, "MODEL_VERSION", "unknown"),
            "runtime": "cpu",
            "readiness": "ready",
            "error": None,
        }
    )

    if settings.MOCK_INFERENCE:
        detector = MockObjectDetector()
    else:
        detector = OwlV2ZeroShotDetector()

    det_health = (
        detector.get_health()
        if hasattr(detector, "get_health")
        else {
            "engine": "owlv2",
            "model_version": "google/owlv2-base-patch16-ensemble",
            "runtime": "Apple Silicon CPU",
            "readiness": "ready",
            "error": None,
        }
    )

    vl_adapter = MockPaddleOcrVlAdapter() if settings.MOCK_INFERENCE else PaddleOcrVlAdapter()
    vl_health = vl_adapter.get_health()

    overall_status = "ready"
    if ocr_health.get("readiness") == "degraded" or det_health.get("readiness") == "degraded":
        overall_status = "degraded"

    return ModelsHealthResponse(
        status=overall_status,
        ocr=ModelHealthDetail(**ocr_health),
        detector=ModelHealthDetail(**det_health),
        paddleocr_vl=ModelHealthDetail(**vl_health),
    )
