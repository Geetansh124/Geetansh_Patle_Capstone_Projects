import io
from PIL import Image
from fastapi.testclient import TestClient
from src.api import app, DISCLAIMER


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["education_only"] is True
    assert DISCLAIMER in data["notice"]


def test_validate_study_supported_modality():
    payload = {"study_id": "STUDY-1234", "modality": "DX", "view_position": "PA"}
    response = client.post("/studies/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["study_id"] == "STUDY-1234"


def test_validate_study_unsupported_modality():
    payload = {"study_id": "STUDY-9999", "modality": "MRI"}
    response = client.post("/studies/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is False
    assert "not supported" in data["reason"].lower()


def test_inference_classification_file_upload():
    # Create synthetic test image
    img = Image.new("RGB", (224, 224), color=(128, 128, 128))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    
    response = client.post(
        "/inference/classification",
        files={"file": ("test_xray.png", buf.getvalue(), "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert "top_findings" in data
    assert len(data["top_findings"]) > 0
    assert "gradcam_data_url" in data
    assert data["notice"] == DISCLAIMER


def test_inference_segmentation_file_upload():
    img = Image.new("RGB", (224, 224), color=(100, 100, 100))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    
    response = client.post(
        "/inference/segmentation",
        files={"file": ("test_ct.png", buf.getvalue(), "image/png")}
    )
    assert response.status_code == 200
    data = response.json()
    assert data["task"] == "MSD Task06 Lung Segmentation"
    assert "overlay_data_url" in data
    assert data["estimated_dice_score"] > 0.9


def test_review_events_endpoint(tmp_path, monkeypatch):
    import src.api as api_mod
    monkeypatch.setattr(api_mod, "AUDIT_PATH", tmp_path / "test_audit.jsonl")
    payload = {
        "study_id": "STUDY-EXP-01",
        "decision": "approve",
        "reviewer": "evaluator_1",
        "comments": "Normal bilateral lung expansion"
    }
    response = client.post("/review-events", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["decision"] == "approve"
    assert data["reviewer"] == "evaluator_1"
    
    audit_resp = client.get("/audit")
    assert audit_resp.status_code == 200
    assert audit_resp.json()["total_events"] == 1
