r"""
Smart Fog Vision — Road Segmentation Training (BDD100K drivable/road)
Dataset: D:\RoadDetector\bdd100k_seg (7000 train / 1000 val) + D:\BDD100K mirror
Masks: *_train_id.png, train_id == 0 -> road (BDD100K semantic id 0 = road)
Model: TinyUNet (16-32-64-128) -> binary road logits, 256x256, ~0.5M params
Export: road_seg.pt + road_seg.onnx (browser ONNX Runtime Web, input/output 1x3x256x256 -> 1x1x256x256)
Frontend: road_seg.js overlays translucent green fill + bright lime boundary.
"""
import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import os
import json
import time
import random
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def parse_args():
    p = argparse.ArgumentParser(description="Train TinyUNet road segmenter on BDD100K")
    default_seg = r"D:\BDD100K" if Path(r"D:\BDD100K").exists() else r"D:\RoadDetector\bdd100k_seg\bdd100k\seg"
    p.add_argument("--seg-root", type=str, default=default_seg,
                   help="Root with images/train, labels/train, images/val, labels/val")
    p.add_argument("--alt-root", type=str, default=r"D:\BDD100K",
                   help="Alternate mirror root (images/train + labels/train) if seg-root missing")
    p.add_argument("--attr-json", type=str,
                   default=r"D:\RoadDetector\bdd100k_labels_release\bdd100k\labels\bdd100k_labels_images_train.json",
                   help="Path to BDD100K JSON attributes for adverse weather filtering")
    p.add_argument("--out-dir", type=str, default=r"D:\Smart_Fog_Vision\models")
    p.add_argument("--train-samples", type=int, default=700)
    p.add_argument("--val-samples", type=int, default=150)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--imgsz", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--resume", type=str, default="",
                   help="Optional checkpoint (.pt) to resume from, e.g. models/last_road_seg.pt")
    p.add_argument("--export-only", action="store_true",
                   help="Skip training; load --resume (or road_seg.pt) and export ONNX + preview only")
    return p.parse_args()


# ---------------- Model ----------------
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
    """4-level U-Net: 16/32/64/128. Input 3ch -> 1ch logits."""
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


# ---------------- Dataset ----------------
def collect_pairs(img_dir, lbl_dir):
    img_dir, lbl_dir = Path(img_dir), Path(lbl_dir)
    if not img_dir.exists() or not lbl_dir.exists():
        return []
    pairs = []
    for img_p in sorted(img_dir.glob("*.jpg")):
        stem = img_p.stem
        # BDD100K seg masks are named <stem>_train_id.png even for val split
        cands = [
            lbl_dir / f"{stem}_train_id.png",
            lbl_dir / f"{stem}_val_id.png",
            lbl_dir / f"{stem}.png",
        ]
        # also handle .jpeg/.png source images
        found = next((c for c in cands if c.exists()), None)
        if found is not None:
            pairs.append((str(img_p), str(found)))
    # also accept .png images
    for img_p in sorted(img_dir.glob("*.png")):
        if img_p.name.endswith("_id.png") or img_p.name.endswith("_color.png"):
            continue
        stem = img_p.stem
        cands = [lbl_dir / f"{stem}_train_id.png", lbl_dir / f"{stem}.png"]
        found = next((c for c in cands if c.exists()), None)
        if found is not None:
            pairs.append((str(img_p), str(found)))
    return pairs


def load_weather_attributes(attr_json_path):
    p = Path(attr_json_path)
    if not p.exists():
        return {}
    cache_file = Path(r"D:\Smart_Fog_Vision\models\bdd100k_seg_attr_cache.json")
    if cache_file.exists():
        try:
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    print(f"Indexing weather attributes from {p.name}...")
    matched = {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            items = json.load(f)
            for it in items:
                name = it.get("name")
                attrs = it.get("attributes", {})
                if name and attrs:
                    matched[name] = attrs
        with open(cache_file, "w", encoding="utf-8") as f:
            json.dump(matched, f)
    except Exception as e:
        print(f"Warning: could not index weather attributes: {e}")
    return matched


def score_seg_pair(img_p, attr_map):
    fname = Path(img_p).name
    attrs = attr_map.get(fname, {})
    w = attrs.get("weather", "").lower()
    t = attrs.get("timeofday", "").lower()
    score = 0
    if w == "foggy":
        score += 100
    elif w == "rainy":
        score += 80
    elif w == "snowy":
        score += 70
    elif w == "overcast":
        score += 50
    elif w == "partly cloudy":
        score += 20
    if t in ["night", "dawn/dusk"]:
        score += 35
    return score


class RoadSegDataset(Dataset):
    def __init__(self, pairs, imgsz=256, augment=False):
        self.pairs = pairs
        self.imgsz = imgsz
        self.augment = augment

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_p, msk_p = self.pairs[idx]
        img = Image.open(img_p).convert("RGB")
        msk = Image.open(msk_p)  # mode L, values 0..18 + 255 ignore
        m = np.array(msk, dtype=np.uint8)
        road = (m == 0).astype(np.float32)  # 0 = road -> 1

        img = img.resize((self.imgsz, self.imgsz), Image.BILINEAR)
        road_img = Image.fromarray((road * 255).astype(np.uint8)).resize(
            (self.imgsz, self.imgsz), Image.NEAREST)
        road = (np.array(road_img, dtype=np.float32) / 255.0 > 0.5).astype(np.float32)

        if self.augment and random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            road = road[:, ::-1].copy()

        arr = np.array(img, dtype=np.float32) / 255.0
        arr = np.transpose(arr, (2, 0, 1))  # CHW
        return (torch.from_numpy(arr),
                torch.from_numpy(road).unsqueeze(0),
                img_p)


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
    return {"loss": tot_loss / max(n, 1), "iou": iou, "dice": dice,
            "acc": acc, "precision": prec, "recall": rec}


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 64)
    print("Smart Fog Vision — Road Segmentation Training (BDD100K, TinyUNet)")
    print(f"Device: {device} | threads={torch.get_num_threads()} | imgsz={args.imgsz}")
    print("=" * 64)

    seg_root = Path(args.seg_root)
    train_pairs = collect_pairs(seg_root / "images" / "train", seg_root / "labels" / "train")
    val_pairs = collect_pairs(seg_root / "images" / "val", seg_root / "labels" / "val")
    print(f"Seg root: {seg_root} (exists={seg_root.exists()})")
    print(f"Found pairs: train={len(train_pairs)} val={len(val_pairs)}")

    if len(train_pairs) == 0:
        # fallback: D:\BDD100K mirror (images/train + labels/train with _train_id.png?)
        alt = Path(args.alt_root)
        print(f"Trying alt root {alt}")
        train_pairs = collect_pairs(alt / "images" / "train", alt / "labels" / "train")
        val_pairs = collect_pairs(alt / "images" / "val", alt / "labels" / "val")
        print(f"Alt pairs: train={len(train_pairs)} val={len(val_pairs)}")

    if len(train_pairs) == 0 or len(val_pairs) == 0:
        raise FileNotFoundError("No training pairs found. Check --seg-root layout.")

    attr_map = load_weather_attributes(args.attr_json)
    rng = random.Random(args.seed)
    rng.shuffle(train_pairs)
    rng.shuffle(val_pairs)

    if attr_map:
        train_pairs.sort(key=lambda p: score_seg_pair(p[0], attr_map), reverse=True)

    n_train = args.train_samples if 0 < args.train_samples < len(train_pairs) else len(train_pairs)
    n_val = args.val_samples if 0 < args.val_samples < len(val_pairs) else len(val_pairs)
    train_pairs = train_pairs[:n_train]
    val_pairs = val_pairs[:n_val]

    if attr_map:
        sample_w = Counter(attr_map.get(Path(p[0]).name, {}).get("weather", "unindexed") for p in train_pairs)
        sample_t = Counter(attr_map.get(Path(p[0]).name, {}).get("timeofday", "unindexed") for p in train_pairs)
        print(f"\n[Adverse Weather Sampling Stats for Road Segmentation]")
        print(f"  Train Weather:     {dict(sample_w)}")
        print(f"  Train Time-of-Day: {dict(sample_t)}")

    print(f"Using train={len(train_pairs)} val={len(val_pairs)}")

    train_ds = RoadSegDataset(train_pairs, imgsz=args.imgsz, augment=True)
    val_ds = RoadSegDataset(val_pairs, imgsz=args.imgsz, augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=False)

    model = TinyUNet().to(device)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"TinyUNet params: {total_p:,}")

    # Optional resume (continue training from interrupted checkpoint)
    if args.resume:
        rp = Path(args.resume)
        if rp.exists():
            try:
                sd = torch.load(str(rp), map_location=device)
                model.load_state_dict(sd, strict=True)
                print(f"Resumed weights from {rp}")
            except Exception as e:
                print(f"Resume load warning ({rp}): {e}")
        else:
            print(f"Resume path not found: {rp} — training from scratch")

    opt = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-6)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # If resuming, seed best_iou from current checkpoint's val score so we don't
    # overwrite a good best_road_seg.pt with a worse early epoch.
    best_iou = -1.0
    if args.resume and (out_dir / "best_road_seg.pt").exists():
        try:
            tmp = TinyUNet().to(device)
            tmp.load_state_dict(torch.load(str(out_dir / "best_road_seg.pt"), map_location=device), strict=True)
            m0 = evaluate(tmp, val_loader, device)
            best_iou = float(m0["iou"])
            print(f"Existing best_road_seg.pt val IoU={best_iou:.4f} — will only overwrite on improvement")
            del tmp
        except Exception as e:
            print(f"Could not score existing best checkpoint: {e}")
    best_state = None
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": [], "lr": []}
    # Export-only mode: skip loop, just package the resume/best weights.
    if args.export_only:
        print("Export-only mode: skipping training loop.")
        exp_src = Path(args.resume) if args.resume and Path(args.resume).exists() else (out_dir / "best_road_seg.pt")
        try:
            sd = torch.load(str(exp_src), map_location=device)
            model.load_state_dict(sd, strict=True)
            print(f"Loaded {exp_src} for export")
        except Exception as e:
            print(f"Export load failed ({exp_src}): {e}")
        best_state = {k: v.cpu() for k, v in model.state_dict().items()}
        best_iou = best_iou if best_iou > 0 else 0.0
        history = {"train_loss": [], "val_loss": [], "val_iou": [], "val_dice": [], "lr": []}
        total_min = 0.0
        final = evaluate(model, val_loader, device)
        print(f"EXPORT CANDIDATE VAL: loss {final['loss']:.4f} IoU {final['iou']:.4f} "
              f"Dice {final['dice']:.4f} acc {final['acc']:.4f}")
    t0 = time.time()
    if not args.export_only:
        for epoch in range(1, args.epochs + 1):
            model.train()
            run_loss, seen = 0.0, 0
            ep_t = time.time()
            for i, (imgs, masks, _) in enumerate(train_loader):
                imgs, masks = imgs.to(device), masks.to(device)
                opt.zero_grad()
                logits = model(imgs)
                loss, _, _ = bce_dice_loss(logits, masks)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                bs = imgs.size(0)
                run_loss += loss.item() * bs
                seen += bs
                if (i + 1) % 15 == 0 or (i + 1) == len(train_loader):
                    print(f"  Epoch {epoch:02d} | batch {i+1:03d}/{len(train_loader):03d} | loss {loss.item():.4f}")
            train_loss = run_loss / max(seen, 1)
            metrics = evaluate(model, val_loader, device)
            sched.step()
            lr = opt.param_groups[0]["lr"]
            history["train_loss"].append(train_loss)
            history["val_loss"].append(metrics["loss"])
            history["val_iou"].append(metrics["iou"])
            history["val_dice"].append(metrics["dice"])
            history["lr"].append(lr)
            dt = time.time() - ep_t
            print(f"--> Epoch {epoch:02d}/{args.epochs} ({dt:.1f}s) "
                  f"train {train_loss:.4f} | val loss {metrics['loss']:.4f} "
                  f"IoU {metrics['iou']:.4f} Dice {metrics['dice']:.4f} "
                  f"acc {metrics['acc']:.4f} P {metrics['precision']:.3f} R {metrics['recall']:.3f}")
            torch.save(model.state_dict(), out_dir / "last_road_seg.pt")
            if metrics["iou"] > best_iou:
                best_iou = metrics["iou"]
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}
                no_improve = 0
                torch.save(model.state_dict(), out_dir / "best_road_seg.pt")
                print(f"    *** new best IoU {best_iou:.4f} -> best_road_seg.pt")
            else:
                no_improve += 1
                if no_improve >= args.patience:
                    print(f"    early stop ({args.patience} epochs no improve)")
                    break

    total_min = (time.time() - t0) / 60
    print(f"\nTraining done in {total_min:.2f} min. Best val IoU={best_iou:.4f}")
    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # final eval
    final = evaluate(model, val_loader, device)
    print(f"FINAL VAL: loss {final['loss']:.4f} IoU {final['iou']:.4f} "
          f"Dice {final['dice']:.4f} acc {final['acc']:.4f}")

    torch.save(model.state_dict(), out_dir / "road_seg.pt")
    print(f"Saved {out_dir / 'road_seg.pt'}")

    with open(out_dir / "metrics_road_seg_val.json", "w", encoding="utf-8") as f:
        json.dump({"val": final, "history": history,
                   "train_samples": len(train_ds), "val_samples": len(val_ds),
                   "imgsz": args.imgsz, "epochs_trained": len(history["train_loss"]),
                   "train_time_min": total_min}, f, indent=2)
    with open(out_dir / "history_road_seg.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    # ---- ONNX export ----
    model.eval()
    dummy = torch.randn(1, 3, args.imgsz, args.imgsz)
    onnx_path = out_dir / "road_seg.onnx"
    torch.onnx.export(
        model.cpu(), dummy,
        str(onnx_path),
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17, do_constant_folding=True, dynamo=False)
    print(f"ONNX exported: {onnx_path} ({onnx_path.stat().st_size/1024/1024:.2f} MB)")
    try:
        import onnx
        m = onnx.load(str(onnx_path))
        onnx.checker.check_model(m)
        print("ONNX checker PASSED")
        print(" inputs:", [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input])
        print(" outputs:", [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output])
    except Exception as e:
        print(f"ONNX check warning: {e}")

    card = {
        "model_name": "TinyUNet Road Segmenter (BDD100K)",
        "dataset": "BDD100K semantic seg (D:\\RoadDetector\\bdd100k_seg) — road id 0 -> binary",
        "input_size": args.imgsz,
        "params": total_p,
        "val_iou": final["iou"], "val_dice": final["dice"], "val_acc": final["acc"],
        "val_precision": final["precision"], "val_recall": final["recall"],
        "train_samples": len(train_ds), "val_samples": len(val_ds),
        "epochs": len(history["train_loss"]), "train_time_min": round(total_min, 2),
        "onnx_size_mb": round(onnx_path.stat().st_size / 1024 / 1024, 2),
        "preprocess": "RGB resize 256, /255, no mean/std. Sigmoid(logits) > 0.5 = road.",
    }
    with open(out_dir / "road_seg_card.json", "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    print(f"Card -> {out_dir / 'road_seg_card.json'}")
    print("\nDone.")


if __name__ == "__main__":
    main()
