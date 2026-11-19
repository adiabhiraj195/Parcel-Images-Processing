"""API tests for image detail inspection, review modifications, and retries."""

from fastapi.testclient import TestClient


def test_image_review_and_manual_edit(client: TestClient):
    """Verify an operator can inspect an image and patch corrected values."""
    # Create batch with 1 image
    batch_res = client.post(
        "/api/batches",
        json={
            "name": "Single Image Test",
            "files": [{"filename": "box.jpg", "content_type": "image/jpeg", "file_size": 5000}],
        },
    )
    image_id = batch_res.json()["uploads"][0]["image_id"]

    # Retrieve initial image details
    img_res = client.get(f"/api/images/{image_id}")
    assert img_res.status_code == 200
    assert img_res.json()["image_id"] == image_id
    assert img_res.json()["reviewed"] is False

    # Review and edit fields
    patch_res = client.patch(
        f"/api/images/{image_id}/review",
        json={
            "tracking_number": "1Z9999999999999999",
            "weight_value": 2.45,
            "weight_unit": "kg",
            "length": 30.0,
            "width": 20.0,
            "height": 15.0,
            "dimension_unit": "cm",
            "parcel_count": 1,
            "status": "SUCCESS",
            "reviewed": True,
        },
    )
    assert patch_res.status_code == 200
    updated = patch_res.json()
    assert updated["tracking_number"] == "1Z9999999999999999"
    assert updated["weight_value"] == 2.45
    assert updated["weight_unit"] == "kg"
    assert updated["length"] == 30.0
    assert updated["status"] == "SUCCESS"
    assert updated["reviewed"] is True


def test_reject_invalid_status_in_review(client: TestClient):
    """Ensure invalid status string is rejected with 400 Bad Request."""
    batch_res = client.post(
        "/api/batches",
        json={
            "name": "Status Validation Batch",
            "files": [{"filename": "box2.jpg", "content_type": "image/jpeg", "file_size": 5000}],
        },
    )
    image_id = batch_res.json()["uploads"][0]["image_id"]

    patch_res = client.patch(
        f"/api/images/{image_id}/review",
        json={"status": "INVENTED_STATUS"},
    )
    assert patch_res.status_code == 400
    assert "Invalid status" in patch_res.json()["detail"]
