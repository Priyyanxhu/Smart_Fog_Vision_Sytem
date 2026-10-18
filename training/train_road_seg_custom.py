r"""
Smart Fog Vision — Road Segmentation Training on D:\RoadSegmentation
Dataset: D:\RoadSegmentation (images/ and masks/)
Labels in masks:
  - road_surface (#3d3df5 -> B > 180, R < 100, G < 100)
  - marking (#ddff33 -> R > 180, G > 180, B < 100)
Architecture: TinyUNet (16-32-64-128) -> binary road logits, 256x256
Export: models/road_seg.pt and models/road_seg.onnx for ONNX Runtime Web
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import os
import time
import json
import random
import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="Train TinyUNet on D:\\RoadSegmentation")
    p.add_argument("--data-root", type=str, default=r"D:\RoadSegmentation",
                   help="Path to RoadSegmentation root with images/ and masks/")
    p.add_argument("--out-dir", type=str, default=r"D:\Smart_Fog_Vision\models",
                   help="Output directory for weights and ONNX model")
    p.add_argument("--epochs", type=int, default=15,
                   help="Training epochs")
    p.add_argument("--batch-size", type=int, default=4,
                   help="Batch size")
    p.add_argument("--imgsz", type=int, default=256,
                   help="Model input resolution (256x256)")
    p.add_argument("--lr", type=float, default=1e-3,
                   help="Learning rate")
    p.add_argument("--weight-decay", type=float, default=1e-4,
                   help="Weight decay")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--resume", type=str, default=r"D:\Smart_Fog_Vision\models\best_road_seg.pt",
                   help="Warm-start checkpoint (.pt)")
    return p.parse_args()


# ---------------- Model Architecture ----------------
class DoubleConv(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    """4-level U-Net: 16/32/64/128. Input 3ch -> 1ch logits (256x256)."""
    def __init__(self, in_ch=3, base=16):
        super().__init__()
        self.e1 = DoubleConv(in_ch, base)
        self.e2 = DoubleConv(base, base * 2)
        self.e3 = DoubleConv(base * 2, base * 4)
        self.e4 = DoubleConv(base * 4, base * 8)
        self.pool = nn.MaxPool2d(2)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.d3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.d2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.d1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, 1, 1)

    def forward(self, x):
        x1 = self.e1(x)
        x2 = self.e2(self.pool(x1))
        x3 = self.e3(self.pool(x2))
        x4 = self.e4(self.pool(x3))
        u3 = self.up3(x4)
        m3 = torch.cat([u3, x3], dim=1)
        y3 = self.d3(m3)
        u2 = self.up2(y3)
        m2 = torch.cat([u2, x2], dim=1)
        y2 = self.d2(m2)
        u1 = self.up1(y2)
        m1 = torch.cat([u1, x1], dim=1)
        y1 = self.d1(m1)
        return self.out(y1)


# ---------------- Dataset & Augmentation ----------------
def extract_road_mask(mask_arr):
    """
    Extracts road_surface and marking from RGBA/RGB mask into binary road map.
    road_surface: #3d3df5 -> B > 180, R < 100, G < 100
    marking:      #ddff33 -> R > 180, G > 180, B < 100
    """
    r = mask_arr[:, :, 0]
    g = mask_arr[:, :, 1]
    b = mask_arr[:, :, 2]
    is_road_surf = (b > 180) & (r < 100) & (g < 100)
    is_marking = (r > 180) & (g > 180) & (b < 100)
    return (is_road_surf | is_marking).astype(np.float32)


def add_synthetic_fog(img_pil, fog_density=0.35):
    """Adds synthetic fog to image for fog-vision domain adaptation."""
    arr = np.array(img_pil, dtype=np.float32) / 255.0
    h, w, _ = arr.shape
    # Atmospheric light (bright whitish-gray)
    A = 0.82 + 0.12 * np.random.rand()
    # Depth proxy (linear ramp top-to-bottom)
    depth = np.linspace(0.85, 0.25, h)[:, None, None]
    beta = fog_density * (0.8 + 0.4 * np.random.rand())
    transmission = np.exp(-beta * depth)
    fogged = arr * transmission + A * (1.0 - transmission)
    fogged = np.clip(fogged * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(fogged)


class CustomRoadDataset(Dataset):
    def __init__(self, pairs, imgsz=256, augment=False):
        self.pairs = pairs
        self.imgsz = imgsz
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_p, msk_p = self.pairs[idx]
        img = Image.open(img_p).convert("RGB")
        msk_raw = np.array(Image.open(msk_p))
        road_binary = extract_road_mask(msk_raw)

        # Augmentation
        if self.augment:
            # 1. Random horizontal flip
            if random.random() < 0.5:
                img = img.transpose(Image.FLIP_LEFT_RIGHT)
                road_binary = np.fliplr(road_binary).copy()

            # 2. Random synthetic fog (35% probability)
            if random.random() < 0.35:
                img = add_synthetic_fog(img, fog_density=random.uniform(0.2, 0.5))

            # 3. Random brightness/contrast jitter
            if random.random() < 0.5:
                enh_b = ImageEnhance.Brightness(img)
                img = enh_b.enhance(random.uniform(0.7, 1.3))
                enh_c = ImageEnhance.Contrast(img)
                img = enh_c.enhance(random.uniform(0.75, 1.25))

        # Resize to model resolution
        img = img.resize((self.imgsz, self.imgsz), Image.Resampling.BILINEAR)
        road_img = Image.fromarray((road_binary * 255).astype(np.uint8)).resize(
            (self.imgsz, self.imgsz), Image.Resampling.NEAREST
        )
        road = (np.array(road_img, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW

        return (
            torch.from_numpy(arr),
            torch.from_numpy(road).unsqueeze(0),
            str(img_p)
        )


def bce_dice_loss(logits, targets, bce_w=0.5, dice_w=0.5, smooth=1.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets)
    probs = torch.sigmoid(logits)
    inter = (probs * targets).sum(dim=(1, 2, 3))
    union = probs.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
    dice = 1.0 - (2.0 * inter + smooth) / (union + smooth)
    return bce_w * bce + dice_w * dice.mean(), bce, dice.mean()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    tot_i, tot_u, tot_dice_inter, tot_dice_union = 0.0, 0.0, 0.0, 0.0
    tot_correct, tot_pix, tot_loss = 0.0, 0, 0.0
    tp = fp = fn = 0.0

    for imgs, masks, _ in loader:
        imgs, masks = imgs.to(device), masks.to(device)
        logits = model(imgs)
        loss, _, _ = bce_dice_loss(logits, masks)
        tot_loss += loss.item() * imgs.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs > 0.5).float()
        tot_correct += (preds == masks).sum().item()
        tot_pix += masks.numel()

        inter = (preds * masks).sum().item()
        union = ((preds + masks) > 0.5).sum().item()
        tot_i += inter
        tot_u += union
        tot_dice_inter += inter
        tot_dice_union += (preds.sum().item() + masks.sum().item())
        tp += inter
        fp += (preds.sum().item() - inter)
        fn += (masks.sum().item() - inter)

    n = len(loader.dataset)
    iou = tot_i / max(tot_u, 1e-6)
    dice = 2 * tot_dice_inter / max(tot_dice_union, 1e-6)
    acc = tot_correct / max(tot_pix, 1)
    prec = tp / max(tp + fp, 1e-6)
    rec = tp / max(tp + fn, 1e-6)
    return {
        "loss": tot_loss / max(n, 1),
        "iou": iou,
        "dice": dice,
        "acc": acc,
        "precision": prec,
        "recall": rec
    }


def train_road_segmentation():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 64)
    print("Smart Fog Vision — Road Segmentation Training (D:\\RoadSegmentation)")
    print(f"Device: {device} | Input size: {args.imgsz}x{args.imgsz} | Epochs: {args.epochs}")
    print("=" * 64)

    data_root = Path(args.data_root)
    img_dir = data_root / "images"
    msk_dir = data_root / "masks"

    if not img_dir.exists() or not msk_dir.exists():
        raise FileNotFoundError(f"Missing images or masks directory in {data_root}")

    pairs = []
    for img_p in sorted(img_dir.glob("*.png")):
        stem = img_p.stem
        msk_p = msk_dir / f"{stem}.png"
        if msk_p.exists():
            pairs.append((img_p, msk_p))

    print(f"Loaded {len(pairs)} matched image-mask pairs from {data_root}")
    if len(pairs) == 0:
        raise ValueError("No matching image/mask pairs found!")

    # Train / Val split (80/20 split)
    rng = random.Random(args.seed)
    rng.shuffle(pairs)
    val_count = max(1, int(len(pairs) * 0.2))
    train_pairs = pairs[val_count:]
    val_pairs = pairs[:val_count]

    print(f"Training split:   {len(train_pairs)} pairs (with augmentation & synthetic fog)")
    print(f"Validation split: {len(val_pairs)} pairs")

    train_ds = CustomRoadDataset(train_pairs, imgsz=args.imgsz, augment=True)
    val_ds = CustomRoadDataset(val_pairs, imgsz=args.imgsz, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = TinyUNet().to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"TinyUNet Parameters: {total_p:,}")

    # Resume from checkpoint if provided
    if args.resume and Path(args.resume).exists():
        try:
            sd = torch.load(args.resume, map_location=device)
            model.load_state_dict(sd, strict=True)
            print(f"Successfully warm-started from pretrained weights: {args.resume}")
        except Exception as e:
            print(f"Could not load resume weights ({e}); initializing fresh weights.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    # Initial validation score
    init_metrics = evaluate(model, val_loader, device)
    print(f"Initial Validation -> IoU: {init_metrics['iou']:.4f} | Dice: {init_metrics['dice']:.4f} | Acc: {init_metrics['acc']:.4f}")

    best_iou = init_metrics["iou"]
    best_state = {k: v.cpu() for k, v in model.state_dict().items()}
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": []}

    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        ep_t0 = time.time()

        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            logits = model(imgs)
            loss, bce, dice = bce_dice_loss(logits, masks)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = imgs.size(0)
            running_loss += loss.item() * bs
            seen += bs

        scheduler.step()
        train_loss = running_loss / max(seen, 1)
        val_metrics = evaluate(model, val_loader, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_dice"].append(val_metrics["dice"])

        ep_duration = time.time() - ep_t0
        print(f"Epoch [{epoch:02d}/{args.epochs:02d}] ({ep_duration:.1f}s) "
              f"Train Loss: {train_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
              f"IoU: {val_metrics['iou']:.4f} | Dice: {val_metrics['dice']:.4f} | "
              f"Acc: {val_metrics['acc']:.4f} | P: {val_metrics['precision']:.3f} | R: {val_metrics['recall']:.3f}")

        # Save latest checkpoint
        torch.save(model.state_dict(), out_dir / "last_road_seg.pt")

        if val_metrics["iou"] >= best_iou:
            best_iou = val_metrics["iou"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            torch.save(model.state_dict(), out_dir / "best_road_seg.pt")
            print(f"  --> *** New best model saved (IoU: {best_iou:.4f})")

    total_time = (time.time() - t0) / 60.0
    print(f"\nTraining completed in {total_time:.2f} minutes. Best Val IoU: {best_iou:.4f}")

    # Load best weights for final export
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    final_metrics = evaluate(model, val_loader, device)
    print(f"\nFINAL EVALUATION: IoU: {final_metrics['iou']:.4f} | Dice: {final_metrics['dice']:.4f} | Acc: {final_metrics['acc']:.4f}")

    # Save active deployment weights
    road_seg_pt = out_dir / "road_seg.pt"
    torch.save(model.state_dict(), road_seg_pt)
    print(f"Saved PyTorch model -> {road_seg_pt}")

    # Export ONNX for Browser (ONNX Runtime Web)
    model.eval()
    dummy = torch.randn(1, 3, args.imgsz, args.imgsz, device="cpu")
    onnx_path = out_dir / "road_seg.onnx"
    print(f"Exporting ONNX model to {onnx_path}...")
    torch.onnx.export(
        model.cpu(), dummy,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        do_constant_folding=True,
        dynamo=False
    )
    # Remove any stray .data file if created
    data_f = Path(str(onnx_path) + ".data")
    if data_f.exists():
        data_f.unlink()
    print(f"ONNX Model Exported: {onnx_path} ({onnx_path.stat().st_size / (1024 * 1024):.2f} MB)")

    # Validate ONNX
    try:
        import onnx
        m = onnx.load(str(onnx_path))
        onnx.checker.check_model(m)
        print("ONNX Checker: Model integrity verified successfully.")
    except Exception as e:
        print(f"ONNX Checker warning: {e}")

    # Save model metadata card
    card = {
        "model_name": "TinyUNet Road Segmenter",
        "dataset": "D:\\RoadSegmentation (Ground Truth Road & Marking Annotations)",
        "input_size": args.imgsz,
        "parameters": total_p,
        "val_iou": float(final_metrics["iou"]),
        "val_dice": float(final_metrics["dice"]),
        "val_acc": float(final_metrics["acc"]),
        "val_precision": float(final_metrics["precision"]),
        "val_recall": float(final_metrics["recall"]),
        "train_samples": len(train_pairs),
        "val_samples": len(val_pairs),
        "epochs": args.epochs,
        "train_time_min": round(total_time, 2),
        "onnx_size_mb": round(onnx_path.stat().st_size / (1024 * 1024), 2),
        "preprocess": "RGB resize 256x256, float32 / 255.0, Sigmoid(logits) > 0.5 = road."
    }
    with open(out_dir / "road_seg_card.json", "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    print(f"Model card updated: {out_dir / 'road_seg_card.json'}")

    with open(out_dir / "history_road_seg.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)
    print("Road segmentation training complete!")


if __name__ == "__main__":
    train_road_segmentation()
