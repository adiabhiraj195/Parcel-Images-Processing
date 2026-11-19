"""Pytest configuration and shared fixtures."""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    """Session-scoped test client for FastAPI application."""
    with TestClient(app) as test_client:
        yield test_client
