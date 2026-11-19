"""Tests for system health endpoints."""

from fastapi.testclient import TestClient


def test_health_endpoint(client: TestClient):
    """Verify that /api/health returns HTTP 200 and expected schema."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["service"] == "parcel-intelligence-backend"
    assert "platform" in data
    assert "python_version" in data


def test_root_health_endpoint(client: TestClient):
    """Verify that /health also returns HTTP 200."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_models_health_endpoint(client: TestClient):
    """Verify that /api/health/models returns model engine, version, runtime and readiness."""
    response = client.get("/api/health/models")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ("ready", "degraded")
    assert "ocr" in data
    assert "detector" in data
    assert data["ocr"]["engine"] in ("paddleocr", "mock-ocr", "local_cpu", "tesseract")
    assert "runtime" in data["ocr"]
    assert data["ocr"]["readiness"] in ("ready", "degraded")
    assert "runtime" in data["detector"]
    assert data["detector"]["readiness"] in ("ready", "degraded")
