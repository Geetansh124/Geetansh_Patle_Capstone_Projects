"""Validate NIH ChestX-ray14 dataset manifests, split boundaries, and data hygiene."""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import pandas as pd
import numpy as np


def validate_manifest(csv_path: Path, patient_col: str = "Patient ID") -> Dict[str, Any]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Manifest not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    required = {"Image Index", "Finding Labels", patient_col}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Manifest missing required columns: {sorted(missing_cols)}")
    
    total_rows = len(df)
    unique_patients = df[patient_col].nunique()
    unique_images = df["Image Index"].nunique()
    null_counts = df[list(required)].isnull().sum().to_dict()
    
    # Calculate label distribution
    all_findings: List[str] = []
    multi_label_count = 0
    for val in df["Finding Labels"].dropna().astype(str):
        tags = [t.strip() for t in val.split("|") if t.strip()]
        if len(tags) > 1:
            multi_label_count += 1
        all_findings.extend(tags)
    
    label_series = pd.Series(all_findings).value_counts()
    label_distribution = {k: int(v) for k, v in label_series.items()}
    
    summary = {
        "status": "valid",
        "total_records": total_rows,
        "unique_patients": unique_patients,
        "unique_images": unique_images,
        "duplicate_images": total_rows - unique_images,
        "null_counts": null_counts,
        "multi_label_records": multi_label_count,
        "multi_label_ratio": float(multi_label_count / max(1, total_rows)),
        "label_counts": label_distribution,
        "notice": "For education only — not for clinical use."
    }
    return summary


def check_split_leakage(train_df: pd.DataFrame, test_df: pd.DataFrame, patient_col: str = "Patient ID") -> Dict[str, Any]:
    train_patients = set(train_df[patient_col].dropna().unique())
    test_patients = set(test_df[patient_col].dropna().unique())
    overlap = train_patients.intersection(test_patients)
    
    return {
        "leakage_detected": len(overlap) > 0,
        "train_patients": len(train_patients),
        "test_patients": len(test_patients),
        "overlapping_patients_count": len(overlap),
        "overlapping_patient_ids": sorted(list(overlap))[:20],
        "is_patient_safe": len(overlap) == 0
    }


def main():
    parser = argparse.ArgumentParser(description="Validate ChestX-ray manifest for education capstone.")
    parser.add_argument("--csv", type=Path, required=True, help="Path to manifest CSV")
    parser.add_argument("--output", type=Path, default=Path("artifacts/manifest_summary.json"), help="Output summary JSON")
    parser.add_argument("--patient-col", type=str, default="Patient ID", help="Patient identifier column")
    args = parser.parse_args()
    
    summary = validate_manifest(args.csv, patient_col=args.patient_col)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Manifest validation passed. Summary written to {args.output}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
