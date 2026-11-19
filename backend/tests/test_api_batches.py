"""API test suite for batches, uploads, and CSV export."""

from fastapi.testclient import TestClient


def test_create_batch_and_get_results(client: TestClient):
    """Test full cycle: batch creation -> retrieve -> upload completion -> results -> CSV."""
    payload = {
        "name": "Morning Intake #1",
        "files": [
            {
                "filename": "scan_001.jpg",
                "content_type": "image/jpeg",
                "file_size": 204850,
            },
            {
                "filename": "scan_002.png",
                "content_type": "image/png",
                "file_size": 150200,
            },
        ],
    }

    # 1. Create batch
    response = client.post("/api/batches", json=payload)
    assert response.status_code == 201
    batch_data = response.json()
    batch_id = batch_data["batch_id"]
    assert batch_data["name"] == "Morning Intake #1"
    assert len(batch_data["uploads"]) == 2
    assert "upload_url" in batch_data["uploads"][0]
    assert batch_data["uploads"][0]["s3_key"].startswith(f"batches/{batch_id}/originals/")

    # 2. Get batch details
    get_res = client.get(f"/api/batches/{batch_id}")
    assert get_res.status_code == 200
    assert get_res.json()["total_images"] == 2
    assert get_res.json()["uploaded_images"] == 0

    # 3. Simulate upload completion
    complete_res = client.post(f"/api/batches/{batch_id}/uploads/complete")
    assert complete_res.status_code == 200

    # 4. Get results
    results_res = client.get(f"/api/batches/{batch_id}/results")
    assert results_res.status_code == 200
    results_data = results_res.json()
    assert results_data["total_results"] == 2
    assert len(results_data["results"]) == 2

    # 5. Export CSV
    csv_res = client.get(f"/api/batches/{batch_id}/export.csv")
    assert csv_res.status_code == 200
    assert csv_res.headers["content-type"].startswith("text/csv")
    lines = csv_res.text.strip().split("\r\n") if "\r\n" in csv_res.text else csv_res.text.strip().split("\n")
    # 1 header line + 2 data rows = 3 lines total
    assert len(lines) == 3
    assert lines[0].startswith("image_id,image_name,tracking_number")


def test_reject_unsupported_file_type(client: TestClient):
    """Ensure invalid file extensions are rejected with 422 Unprocessable Entity."""
    payload = {
        "name": "Bad Batch",
        "files": [
            {
                "filename": "malicious.pdf",
                "content_type": "application/pdf",
                "file_size": 1024,
            }
        ],
    }
    response = client.post("/api/batches", json=payload)
    assert response.status_code == 422
