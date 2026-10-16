r"""
Smart Fog Vision — Supervised Fog Model Training via Zero-Download Streaming
Dataset: https://www.kaggle.com/datasets/mohammedawais505/supervised-fog-removal-dataset
Mirrored at: Hugging Face "Aeye-coder/Supervised-Fog-Removal-Dataset"

Zero-Download Streaming Architecture:
  Instead of downloading the 94 GB dataset, samples are streamed on-the-fly
  over HTTP directly into memory using Hugging Face datasets streaming (streaming=True).

Supported Training Tasks:
  1. 'dehaze' (Default):
     Trains AOD-Net (All-in-One Dehazing Network) mapping foggy -> clean image.
     Physics-based reformulated scattering model.
     Exports: models/fog_dehazer.pt & models/fog_dehazer.onnx (Browser-ready WASM/WebGL).
  
  2. 'classify':
     Trains MobileNetV2 fog vs non-fog classifier (clean=0, foggy=1).
     Exports: models/fog_classifier.pt & models/fog_classifier.onnx.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import time
import json
import argparse
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class AODNet(nn.Module):
    """
    All-in-One Dehazing Network (Boyi Li et al., ICCV 2017)
    Reformulates atmospheric scattering: J(x) = K(x) * I(x) - K(x) + b
    Ultra-lightweight (~2,000 parameters), perfect for 60+ FPS on-device web inference.
    """
    def __init__(self):
        super(AODNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, kernel_size=1)
        self.conv2 = nn.Conv2d(3, 3, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(6, 3, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(6, 3, kernel_size=7, padding=3)
        self.conv5 = nn.Conv2d(12, 3, kernel_size=3, padding=1)
        self.b = 1.0

    def forward(self, x):
        # x is foggy image [B, 3, H, W] in range [0, 1]
        x1 = F.relu(self.conv1(x))
        x2 = F.relu(self.conv2(x1))
        cat1 = torch.cat((x1, x2), dim=1)
        x3 = F.relu(self.conv3(cat1))
        cat2 = torch.cat((x2, x3), dim=1)
        x4 = F.relu(self.conv4(cat2))
        cat3 = torch.cat((x1, x2, x3, x4), dim=1)
        k = F.relu(self.conv5(cat3))

        # J(x) = K(x) * I(x) - K(x) + b
        output = k * x - k + self.b
        output = F.relu(output)
        output = torch.clamp(output, 0.0, 1.0)
        return output


class FogClassifier(nn.Module):
    """
    MobileNetV2 Transfer Learning for Fog vs Non-Fog detection
    Input: [B, 3, 224, 224] -> Output: [B, 2] (0: non-fog, 1: fog)
    """
    def __init__(self, pretrained=True):
        super(FogClassifier, self).__init__()
        weights = models.MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = models.mobilenet_v2(weights=weights)
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.backbone(x)


# ---------------------------------------------------------------------------
# Streaming Dataset
# ---------------------------------------------------------------------------

class StreamingFogDataset(IterableDataset):
    """
    Streams samples on-the-fly from Hugging Face dataset without saving 94 GB to disk.
    """
    def __init__(self, hf_dataset_name, split="train", max_samples=500, task="dehaze", img_size=256):
        super().__init__()
        self.hf_dataset_name = hf_dataset_name
        self.split = split
        self.max_samples = max_samples
        self.task = task
        self.img_size = img_size

        if self.task == "dehaze":
            self.transform = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),  # Scales to [0, 1]
            ])
        else:
            self.transform = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __iter__(self):
        from datasets import load_dataset
        ds = load_dataset(self.hf_dataset_name, split=self.split, streaming=True)
        count = 0

        for sample in ds:
            if count >= self.max_samples:
                break

            try:
                clean_img = sample["clean"]
                foggy_img = sample["foggy"]

                if clean_img.mode != "RGB":
                    clean_img = clean_img.convert("RGB")
                if foggy_img.mode != "RGB":
                    foggy_img = foggy_img.convert("RGB")

                if self.task == "dehaze":
                    # Pair: (foggy_input, clean_target)
                    inp = self.transform(foggy_img)
                    tgt = self.transform(clean_img)
                    yield inp, tgt
                    count += 1
                else:
                    # Classification yields both clean (0) and foggy (1)
                    clean_t = self.transform(clean_img)
                    yield clean_t, torch.tensor(0, dtype=torch.long)
                    count += 1

                    if count < self.max_samples:
                        foggy_t = self.transform(foggy_img)
                        yield foggy_t, torch.tensor(1, dtype=torch.long)
                        count += 1

            except Exception as e:
                # Skip corrupt/network hiccup items
                continue


# ---------------------------------------------------------------------------
# Training Logic
# ---------------------------------------------------------------------------

def train_dehaze(args, device):
    print("\n[Mode: Dehazing] Training AOD-Net Supervised Fog Removal Model")
    model = AODNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    criterion = nn.SmoothL1Loss()

    train_ds = StreamingFogDataset(
        args.dataset, split="train", max_samples=args.samples, task="dehaze", img_size=args.img_size
    )
    val_ds = StreamingFogDataset(
        args.dataset, split="validation", max_samples=args.val_samples, task="dehaze", img_size=args.img_size
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        train_loss = 0.0
        train_batches = 0

        print(f"\n--- Epoch {epoch}/{args.epochs} ---")
        for i, (foggy, clean) in enumerate(train_loader):
            foggy = foggy.to(device)
            clean = clean.to(device)

            optimizer.zero_grad()
            pred = model(foggy)
            loss = criterion(pred, clean)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_batches += 1
            if (i + 1) % max(1, (args.samples // (args.batch_size * 5))) == 0:
                print(f"  Batch {i+1} | Loss: {loss.item():.4f}")

        avg_train_loss = train_loss / max(1, train_batches)

        # Validation
        model.eval()
        val_loss = 0.0
        val_psnr_list = []
        val_batches = 0

        with torch.no_grad():
            for foggy, clean in val_loader:
                foggy = foggy.to(device)
                clean = clean.to(device)
                pred = model(foggy)
                v_loss = criterion(pred, clean)
                val_loss += v_loss.item()

                # PSNR calculation
                mse = F.mse_loss(pred, clean).item()
                if mse > 1e-8:
                    psnr = 10 * np.log10(1.0 / mse)
                else:
                    psnr = 50.0
                val_psnr_list.append(psnr)
                val_batches += 1

        avg_val_loss = val_loss / max(1, val_batches)
        avg_psnr = np.mean(val_psnr_list) if val_psnr_list else 0.0
        elapsed = time.time() - t0

        print(f"Epoch {epoch} Finished in {elapsed:.1f}s | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Val PSNR: {avg_psnr:.2f} dB")

        history.append({
            "epoch": epoch,
            "train_loss": avg_train_loss,
            "val_loss": avg_val_loss,
            "val_psnr": avg_psnr,
            "time_sec": elapsed
        })

        out_path = Path(args.out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), out_path / "best_fog_dehazer.pt")
            print("  [*] Saved best model checkpoint")

        torch.save(model.state_dict(), out_path / "last_fog_dehazer.pt")

    # Export ONNX
    if args.export_onnx:
        export_dehazer_onnx(model, out_path / "fog_dehazer.onnx", device, args.img_size)

    with open(out_path / "history_dehazer.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\n[Done] Training complete! Models saved in {out_path}")


def export_dehazer_onnx(model, onnx_path, device, img_size):
    print(f"\n[Export] Exporting AOD-Net Dehazer to ONNX -> {onnx_path}")
    model.eval()
    dummy_input = torch.randn(1, 3, img_size, img_size, device=device)
    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size", 2: "height", 3: "width"},
            "output": {0: "batch_size", 2: "height", 3: "width"},
        }
    )
    print(f"[Export] ONNX successfully generated! Size: {onnx_path.stat().st_size / 1024:.1f} KB")


def train_classify(args, device):
    print("\n[Mode: Classifier] Training MobileNetV2 Fog vs Non-Fog Classifier")
    model = FogClassifier(pretrained=True).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    train_ds = StreamingFogDataset(
        args.dataset, split="train", max_samples=args.samples, task="classify", img_size=args.img_size
    )
    val_ds = StreamingFogDataset(
        args.dataset, split="validation", max_samples=args.val_samples, task="classify", img_size=args.img_size
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    out_path = Path(args.out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        model.train()
        total_loss, correct, total = 0.0, 0, 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += x.size(0)

        train_acc = correct / max(1, total)
        train_loss = total_loss / max(1, total)

        # Validation
        model.eval()
        val_loss, v_correct, v_total = 0.0, 0, 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                loss = criterion(out, y)
                val_loss += loss.item() * x.size(0)
                preds = out.argmax(dim=1)
                v_correct += (preds == y).sum().item()
                v_total += x.size(0)

        val_acc = v_correct / max(1, v_total)
        elapsed = time.time() - t0

        print(f"Epoch {epoch}/{args.epochs} [{elapsed:.1f}s] | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc*100:.1f}% | "
              f"Val Acc: {val_acc*100:.1f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), out_path / "fog_classifier.pt")

    if args.export_onnx:
        dummy = torch.randn(1, 3, args.img_size, args.img_size, device=device)
        onnx_file = out_path / "fog_classifier.onnx"
        torch.onnx.export(
            model,
            dummy,
            str(onnx_file),
            opset_version=18,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}}
        )
        print(f"[Export] Saved ONNX classifier to {onnx_file}")


def main():
    parser = argparse.ArgumentParser(description="Zero-Download Supervised Fog Model Training")
    parser.add_argument("--task", type=str, default="dehaze", choices=["dehaze", "classify"],
                        help="'dehaze' for AOD-Net fog removal, 'classify' for MobileNetV2 fog detector")
    parser.add_argument("--dataset", type=str, default="Aeye-coder/Supervised-Fog-Removal-Dataset",
                        help="Hugging Face mirror identifier for Kaggle supervised fog removal dataset")
    parser.add_argument("--samples", type=int, default=200,
                        help="Number of streaming samples to train on (keeps training fast and in-RAM)")
    parser.add_argument("--val-samples", type=int, default=40,
                        help="Number of streaming validation samples")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size")
    parser.add_argument("--img-size", type=int, default=256,
                        help="Image input size (e.g. 256 for dehaze, 224 for classify)")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Learning rate")
    parser.add_argument("--out-dir", type=str, default=r"D:\Smart_Fog_Vision\models",
                        help="Output directory for saved models and ONNX")
    parser.add_argument("--export-onnx", action="store_true", default=True,
                        help="Auto-export trained model to browser-compatible ONNX")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(" Smart Fog Vision — Supervised Fog Training (Zero-Download Streaming)")
    print(f" Task: {args.task} | Device: {device} | Target Samples: {args.samples}")
    print("=" * 60)

    if args.task == "dehaze":
        train_dehaze(args, device)
    else:
        train_classify(args, device)


if __name__ == "__main__":
    main()
