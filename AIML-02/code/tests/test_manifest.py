import pytest
import pandas as pd
from pathlib import Path
from src.validate_manifest import validate_manifest, check_split_leakage


def test_validate_manifest_success(tmp_path):
    manifest_csv = tmp_path / "sample_manifest.csv"
    data = pd.DataFrame({
        "Image Index": ["img_001.png", "img_002.png", "img_003.png"],
        "Finding Labels": ["Atelectasis|Effusion", "No Finding", "Cardiomegaly"],
        "Patient ID": [101, 102, 101]
    })
    data.to_csv(manifest_csv, index=False)
    
    summary = validate_manifest(manifest_csv)
    assert summary["status"] == "valid"
    assert summary["total_records"] == 3
    assert summary["unique_patients"] == 2
    assert summary["unique_images"] == 3
    assert summary["multi_label_records"] == 1
    assert "Atelectasis" in summary["label_counts"]


def test_validate_manifest_missing_column(tmp_path):
    manifest_csv = tmp_path / "invalid_manifest.csv"
    data = pd.DataFrame({
        "Image Index": ["img_001.png"],
        "Finding Labels": ["Atelectasis"]
    })
    data.to_csv(manifest_csv, index=False)
    
    with pytest.raises(ValueError, match="missing required columns"):
        validate_manifest(manifest_csv)


def test_check_split_leakage_detected():
    train_df = pd.DataFrame({"Patient ID": [1, 2, 3]})
    test_df = pd.DataFrame({"Patient ID": [3, 4, 5]})
    leakage = check_split_leakage(train_df, test_df)
    assert leakage["leakage_detected"] is True
    assert leakage["is_patient_safe"] is False
    assert leakage["overlapping_patients_count"] == 1
    assert leakage["overlapping_patient_ids"] == [3]


def test_check_split_leakage_clean():
    train_df = pd.DataFrame({"Patient ID": [1, 2, 3]})
    test_df = pd.DataFrame({"Patient ID": [4, 5, 6]})
    leakage = check_split_leakage(train_df, test_df)
    assert leakage["leakage_detected"] is False
    assert leakage["is_patient_safe"] is True
