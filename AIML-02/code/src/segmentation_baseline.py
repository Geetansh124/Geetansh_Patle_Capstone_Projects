"""MONAI 3D U-Net training entry point for MSD Task06 Lung NIfTI data."""
import argparse
from pathlib import Path

import torch
from monai.data import CacheDataset, DataLoader, load_decathlon_datalist
from monai.losses import DiceCELoss
from monai.networks.nets import UNet
from monai.transforms import Compose, LoadImaged, EnsureChannelFirstd, Orientationd, Spacingd, ScaleIntensityRanged, CropForegroundd, RandCropByPosNegLabeld, EnsureTyped


def main():
    p = argparse.ArgumentParser(description="Educational segmentation only; not for clinical use.")
    p.add_argument("--root", type=Path, required=True, help="MSD Task06_Lung root containing dataset.json")
    p.add_argument("--epochs", type=int, default=5)
    args = p.parse_args()
    train_files = load_decathlon_datalist(str(args.root / "dataset.json"), True, "training", base_dir=str(args.root))
    transforms = Compose([LoadImaged(keys=["image", "label"]), EnsureChannelFirstd(keys=["image", "label"]), Orientationd(keys=["image", "label"], axcodes="RAS"), Spacingd(keys=["image", "label"], pixdim=(1.5, 1.5, 2.0), mode=("bilinear", "nearest")), ScaleIntensityRanged(keys="image", a_min=-1000, a_max=400, b_min=0, b_max=1, clip=True), CropForegroundd(keys=["image", "label"], source_key="image"), RandCropByPosNegLabeld(keys=["image", "label"], label_key="label", spatial_size=(96, 96, 96), pos=1, neg=1, num_samples=2), EnsureTyped(keys=["image", "label"])])
    ds = CacheDataset(train_files, transforms, cache_rate=.1, num_workers=2)
    loader = DataLoader(ds, batch_size=1, shuffle=True, num_workers=2)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNet(spatial_dims=3, in_channels=1, out_channels=2, channels=(16, 32, 64, 128, 256), strides=(2, 2, 2, 2)).to(device)
    optimizer, loss_fn = torch.optim.AdamW(model.parameters(), 1e-4), DiceCELoss(to_onehot_y=True, softmax=True)
    model.train()
    for epoch in range(args.epochs):
        total = 0
        for batch in loader:
            x, y = batch["image"].to(device), batch["label"].to(device); optimizer.zero_grad(); loss = loss_fn(model(x), y); loss.backward(); optimizer.step(); total += loss.item()
        print(f"epoch={epoch+1} train_loss={total/max(1,len(loader)):.5f}")
    Path("artifacts").mkdir(exist_ok=True)
    torch.save(model.cpu().state_dict(), "artifacts/msd_task06_unet.pt")


if __name__ == "__main__": main()
