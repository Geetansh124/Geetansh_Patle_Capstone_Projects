"""Generate synthetic Chest X-ray fixtures and dummy checkpoint for zero-setup local evaluation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from PIL import Image

DISCLAIMER = "For education only — not for clinical use."


def make_dummy_model(num_classes: int):
    try:
        from src.train_classifier import make_model
        return make_model(num_classes)
    except Exception:
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(224 * 224 * 3, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )


def generate_synthetic_data(out_dir: Path, num_patients: int = 10, num_images: int = 25):
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    
    findings_list = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "No Finding"]
    records = []
    
    print(f"Generating {num_images} synthetic chest radiographs in {img_dir}...")
    for i in range(num_images):
        fname = f"synth_{i:04d}.png"
        pid = f"PATIENT_{(i % num_patients) + 1:03d}"
        
        # Create synthetic X-ray-like image (dark with dual lung light lobes)
        img_arr = np.random.normal(50, 10, (224, 224)).clip(0, 255).astype(np.uint8)
        y, x = np.ogrid[:224, :224]
        lobe_left = np.exp(-((x - 80)**2 + (y - 110)**2) / (2 * 35**2))
        lobe_right = np.exp(-((x - 144)**2 + (y - 110)**2) / (2 * 35**2))
        img_arr = np.clip(img_arr + (lobe_left + lobe_right) * 120, 0, 255).astype(np.uint8)
        
        Image.fromarray(img_arr).save(img_dir / fname)
        
        sample_findings = np.random.choice(findings_list, size=np.random.randint(1, 3), replace=False)
        finding_str = "|".join(sample_findings)
        records.append({
            "Image Index": fname,
            "Finding Labels": finding_str,
            "Patient ID": pid,
            "Patient Age": np.random.randint(20, 80),
            "Patient Gender": np.random.choice(["M", "F"]),
            "View Position": "PA"
        })
        
    df = pd.DataFrame(records)
    csv_path = out_dir / "labels.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved manifest to {csv_path}")
    return csv_path, img_dir


def generate_mini_checkpoint(checkpoint_path: Path):
    labels = ["Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass", "Nodule", "Pneumonia", "Pneumothorax"]
    model = make_dummy_model(len(labels))
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "labels": labels,
        "disclaimer": DISCLAIMER,
        "patient_group_split": True,
        "synthetic": True
    }, checkpoint_path)
    print(f"Generated synthetic checkpoint at {checkpoint_path}")


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent.parent
    data_path = base / "data" / "synthetic_chestxray"
    csv_path, img_dir = generate_synthetic_data(data_path)
    generate_mini_checkpoint(base / "artifacts" / "densenet121_synthetic.pt")
