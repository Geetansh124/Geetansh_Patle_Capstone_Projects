"""Transfer-learning baseline for NIH ChestX-ray14 CSV and image directory."""
import argparse
from pathlib import Path

import pandas as pd
import torch
from sklearn.model_selection import GroupShuffleSplit
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms
from PIL import Image

DISCLAIMER = "For education only — not for clinical use."


class ChestXrayDataset(Dataset):
    def __init__(self, frame, root, labels, train=False):
        self.frame, self.root, self.labels = frame.reset_index(drop=True), Path(root), labels
        self.transform = transforms.Compose([transforms.Resize((224, 224)), transforms.Grayscale(3), transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x), transforms.ToTensor(), transforms.Normalize([.485, .456, .406], [.229, .224, .225])])

    def __len__(self): return len(self.frame)

    def __getitem__(self, index):
        row = self.frame.iloc[index]
        image = Image.open(self.root / row.image).convert("RGB")
        target = torch.tensor([int(label in str(row["Finding Labels"]).split("|")) for label in self.labels], dtype=torch.float32)
        return self.transform(image), target


def make_model(classes):
    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    model.classifier = nn.Linear(model.classifier.in_features, classes)
    return model


def main():
    p = argparse.ArgumentParser(description=DISCLAIMER)
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--images", type=Path, required=True)
    p.add_argument("--output", type=Path, default=Path("artifacts/densenet121.pt"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="Optional development subset; disclose it in results")
    args = p.parse_args()
    frame = pd.read_csv(args.csv)
    required = {"Image Index", "Finding Labels", "Patient ID"}
    if not required.issubset(frame.columns): raise ValueError(f"CSV must contain {sorted(required)}")
    frame = frame.rename(columns={"Image Index": "image"}).dropna(subset=["image", "Patient ID"])
    if args.limit: frame = frame.head(args.limit)
    labels = sorted({label for cell in frame["Finding Labels"].astype(str) for label in cell.split("|") if label != "No Finding"})
    splitter = GroupShuffleSplit(n_splits=1, test_size=.2, random_state=42)
    tr, va = next(splitter.split(frame, groups=frame["Patient ID"]))
    train = DataLoader(ChestXrayDataset(frame.iloc[tr], args.images, labels, True), batch_size=args.batch_size, shuffle=True, num_workers=2)
    valid = DataLoader(ChestXrayDataset(frame.iloc[va], args.images, labels), batch_size=args.batch_size, num_workers=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = make_model(len(labels)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    for epoch in range(args.epochs):
        model.train(); total = 0
        for x, y in train:
            x, y = x.to(device), y.to(device); opt.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); opt.step(); total += loss.item()
        model.eval(); val_loss = 0
        with torch.no_grad():
            for x, y in valid: val_loss += loss_fn(model(x.to(device)), y.to(device)).item()
        print(f"epoch={epoch+1} train_loss={total/max(1,len(train)):.5f} validation_loss={val_loss/max(1,len(valid)):.5f}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.cpu().state_dict(), "labels": labels, "disclaimer": DISCLAIMER, "patient_group_split": True}, args.output)
    print(f"Saved weights: {args.output}; device={device}; labels={len(labels)}")


if __name__ == "__main__": main()
