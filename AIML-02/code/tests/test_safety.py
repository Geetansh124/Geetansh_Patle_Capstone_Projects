from src import api
from fastapi import HTTPException


def test_education_only_notice_is_present():
    assert "not for clinical use" in api.app.description.lower()
    assert "not for clinical use" in api.audit()["notice"].lower()


def test_inference_is_explicitly_unavailable_without_orthanc(monkeypatch):
    monkeypatch.delenv("ORTHANC_URL", raising=False)
    try:
        api.infer("sample")
        assert False, "Expected unavailable Orthanc to fail closed"
    except HTTPException as exc:
        assert exc.status_code == 503
        assert "not configured" in exc.detail


def test_review_audit_persists_across_requests(tmp_path, monkeypatch):
    audit_file = tmp_path / "audit.jsonl"
    monkeypatch.setattr(api, "AUDIT_PATH", audit_file)
    response = api.review(api.Review(study_id="demo", decision="approve", reviewer="test"))
    assert response["decision"] == "approve"
    assert api.audit()["events"][0]["study_id"] == "demo"
    assert len(audit_file.read_text().splitlines()) == 1
