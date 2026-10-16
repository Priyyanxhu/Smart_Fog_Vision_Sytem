r"""
Smart Fog Vision — Fog vs Non-Fog Classifier Training & ONNX Export
Supports training on:
  - C:\Users\Priya\Downloads\train\train (GT: clear, hazy: fog)
  - Or custom --data-root / --fog-dir / --nonfog-dir
Model: MobileNetV2 (ImageNet transfer learning) -> 2 classes ("non-fog", "fog")
Exports: PyTorch .pt + standalone browser-compatible ONNX (for onnxruntime-web)
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import os
import argparse
import random
import json
import time
from collections import Counter
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models
from PIL import Image
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Train MobileNetV2 Fog Classifier for Smart Fog Vision")
    default_root = r"D:\DehazingDatasets" if Path(r"D:\DehazingDatasets").exists() else (
        r"D:\Downloads\train\train" if Path(r"D:\Downloads\train\train").exists() else r"C:\Users\Priya\Downloads\train\train"
    )
    parser.add_argument("--data-root", type=str, default=default_root,
                        help="Path containing GT/hazy folders or Dehazing dataset CSVs")
    parser.add_argument("--fog-dir", type=str, default=None,
                        help="Explicit path to fog/hazy images directory")
    parser.add_argument("--nonfog-dir", type=str, default=None,
                        help="Explicit path to non-fog/clear/GT images directory")
    parser.add_argument("--out-dir", type=str, default=r"D:\Smart_Fog_Vision\models",
                        help="Output directory for saved models and metrics")
    parser.add_argument("--samples-per-class", type=int, default=500,
                        help="Max images to use per class (-1 for all)")
    parser.add_argument("--epochs", type=int, default=6,
                        help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay")
    parser.add_argument("--patience", type=int, default=4,
                        help="Early stopping patience")
    parser.add_argument("--threads", type=int, default=8,
                        help="Number of PyTorch CPU threads")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()

def main():
    args = parse_args()

    # --- Seed & Environment ---
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"==================================================")
    print(f"Smart Fog Vision — Fog vs Non-Fog Model Training")
    print(f"Device: {device} | CPU threads: {torch.get_num_threads()}")
    print(f"==================================================")

    # --- Dataset Directories & Loading ---
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Data Root:        {data_root} (exists={data_root.exists()})")
    print(f"Output Directory: {out_dir}")

    class_names = ["non-fog", "fog"]
    num_classes = 2

    # Check for DehazingDatasets CSVs (e.g. dehaing_dataset_train.csv)
    train_csv = None
    for name in ["dehaing_dataset_train.csv", "dehazing_dataset_train.csv"]:
        if (data_root / name).exists():
            train_csv = data_root / name
            break

    if train_csv:
        print(f"Detected Dehazing benchmark CSV dataset at: {data_root}")

        def load_pairs_from_csv(csv_path):
            pairs = []
            if not csv_path.exists():
                return pairs
            import csv
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.reader(f)
                next(reader, None)
                for row in reader:
                    if len(row) < 2:
                        continue
                    gt_rel = row[0].replace("/kaggle/input/dehazing-dataset-thesis/", "").replace("/", "\\")
                    hazy_rel = row[1].replace("/kaggle/input/dehazing-dataset-thesis/", "").replace("/", "\\")
                    gt_p = data_root / gt_rel
                    hazy_p = data_root / hazy_rel
                    if gt_p.exists() and hazy_p.exists():
                        pairs.append((str(gt_p), str(hazy_p)))
            return pairs

        raw_train_pairs = load_pairs_from_csv(train_csv)
        test_csv = data_root / "dehazing_dataset_test.csv"
        raw_test_pairs = load_pairs_from_csv(test_csv) if test_csv.exists() else []
        val_csv = data_root / "dehazing_dataset_val.csv"
        raw_val_pairs = load_pairs_from_csv(val_csv) if val_csv.exists() else []

        print(f"Loaded CSV pairs: Train={len(raw_train_pairs)}, Val={len(raw_val_pairs)}, Test={len(raw_test_pairs)}")

        rng = random.Random(args.seed)
        rng.shuffle(raw_train_pairs)
        rng.shuffle(raw_test_pairs)
        rng.shuffle(raw_val_pairs)

        n_train = args.samples_per_class if (args.samples_per_class > 0 and args.samples_per_class < len(raw_train_pairs)) else len(raw_train_pairs)
        train_pairs = raw_train_pairs[:n_train]

        n_test = min(max(int(n_train * 0.15), 40), len(raw_test_pairs)) if raw_test_pairs else int(n_train * 0.1)
        test_pairs = raw_test_pairs[:n_test] if raw_test_pairs else []

        n_val = min(max(int(n_train * 0.1), 20), len(raw_val_pairs)) if raw_val_pairs else int(n_train * 0.1)
        val_pairs = raw_val_pairs[:n_val] if raw_val_pairs else []

        if not test_pairs:
            test_pairs = train_pairs[-n_test:]
            train_pairs = train_pairs[:-n_test]
        if not val_pairs:
            val_pairs = train_pairs[-n_val:]
            train_pairs = train_pairs[:-n_val]

        def pairs_to_items(pairs):
            items = []
            for gt_p, hazy_p in pairs:
                items.append((gt_p, 0))  # 0 = non-fog (clear)
                items.append((hazy_p, 1))  # 1 = fog (hazy)
            return items

        splits = {
            "train": pairs_to_items(train_pairs),
            "val": pairs_to_items(val_pairs),
            "test": pairs_to_items(test_pairs)
        }
        for k in splits:
            rng.shuffle(splits[k])

        fog_count = len(train_pairs) + len(val_pairs) + len(test_pairs)
        non_count = fog_count
        print(f"Selected dataset split from DehazingDatasets benchmark: {fog_count*2} images total")
    else:
        # Directory mode
        if args.fog_dir:
            fog_dir = Path(args.fog_dir)
        elif (data_root / "hazy").exists():
            fog_dir = data_root / "hazy"
        elif (data_root / "fog images").exists():
            fog_dir = data_root / "fog images"
        elif (data_root / "fog").exists():
            fog_dir = data_root / "fog"
        else:
            raise FileNotFoundError(f"Could not locate fog directory in {data_root}")

        if args.nonfog_dir:
            nonfog_dir = Path(args.nonfog_dir)
        elif (data_root / "GT").exists():
            nonfog_dir = data_root / "GT"
        elif (data_root / "non-fog images").exists():
            nonfog_dir = data_root / "non-fog images"
        elif (data_root / "non-fog").exists():
            nonfog_dir = data_root / "non-fog"
        elif (data_root / "clear").exists():
            nonfog_dir = data_root / "clear"
        else:
            raise FileNotFoundError(f"Could not locate clear/GT directory in {data_root}")

        print(f"Fog Directory:    {fog_dir} (exists={fog_dir.exists()})")
        print(f"Non-Fog Directory:{nonfog_dir} (exists={nonfog_dir.exists()})")

        def get_images(d):
            exts = ["*.jpg", "*.jpeg", "*.png", "*.bmp"]
            files = []
            for ext in exts:
                files.extend(list(d.glob(ext)))
            return sorted(files)

        fog_files = get_images(fog_dir)
        non_files = get_images(nonfog_dir)
        print(f"Found on disk: {len(fog_files)} fog images, {len(non_files)} non-fog images.")

        if args.samples_per_class > 0:
            if len(fog_files) > args.samples_per_class:
                fog_files = fog_files[:args.samples_per_class]
            if len(non_files) > args.samples_per_class:
                non_files = non_files[:args.samples_per_class]
            print(f"Selected sample subset: {len(fog_files)} fog, {len(non_files)} non-fog (Total: {len(fog_files)+len(non_files)})")

        all_items = [(str(p), 1) for p in fog_files] + [(str(p), 0) for p in non_files]
        random.shuffle(all_items)

        def stratified_split(items, ratios=(0.8, 0.1, 0.1), seed=args.seed):
            rng = random.Random(seed)
            per_class = {0: [], 1: []}
            for p, l in items:
                per_class[l].append((p, l))
            sp = {"train": [], "val": [], "test": []}
            for label, lst in per_class.items():
                rng.shuffle(lst)
                n = len(lst)
                n_train = int(n * ratios[0])
                n_val = int(n * ratios[1])
                sp["train"].extend(lst[:n_train])
                sp["val"].extend(lst[n_train:n_train + n_val])
                sp["test"].extend(lst[n_train + n_val:])
            for k in sp:
                rng.shuffle(sp[k])
            return sp

        splits = stratified_split(all_items)
        fog_count = len(fog_files)
        non_count = len(non_files)

    for k, v in splits.items():
        cnt = Counter([l for _, l in v])
        print(f"  Split {k.upper():5s}: {len(v):4d} images (non-fog={cnt[0]:4d}, fog={cnt[1]:4d})")

    # --- Transforms ---
    img_size = 224
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std  = [0.229, 0.224, 0.225]

    train_tf = T.Compose([
        T.Resize(256),
        T.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomRotation(10),
        T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
        T.ToTensor(),
        T.Normalize(imagenet_mean, imagenet_std),
    ])

    eval_tf = T.Compose([
        T.Resize(256),
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(imagenet_mean, imagenet_std),
    ])

    class FogDataset(Dataset):
        def __init__(self, items, transform=None):
            self.items = items
            self.transform = transform
        def __len__(self):
            return len(self.items)
        def __getitem__(self, idx):
            path, label = self.items[idx]
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, label, path

    train_ds = FogDataset(splits["train"], train_tf)
    val_ds   = FogDataset(splits["val"], eval_tf)
    test_ds  = FogDataset(splits["test"], eval_tf)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    # --- Model ---
    print("\nLoading MobileNetV2 pretrained backbone...")
    try:
        mnet = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    except:
        mnet = models.mobilenet_v2(pretrained=True)

    # Replace classifier head cleanly: (Dropout, Linear(1280 -> 2))
    in_features = mnet.classifier[1].in_features
    mnet.classifier = nn.Sequential(
        nn.Dropout(p=0.2),
        nn.Linear(in_features, num_classes)
    )

    # Freeze lower feature blocks (features[:14]) for faster training on CPU
    # Fine-tune features[14:] and classifier head
    for param in mnet.features[:14].parameters():
        param.requires_grad = False

    mnet = mnet.to(device)
    trainable_params = sum(p.numel() for p in mnet.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in mnet.parameters())
    print(f"Total params: {total_params:,} | Trainable params: {trainable_params:,} (lower 14 blocks frozen)")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, mnet.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    def evaluate(loader, model):
        model.eval()
        total_loss = 0
        total_acc = 0
        total = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for imgs, labels, _ in loader:
                imgs = imgs.to(device)
                labels = labels.to(device)
                logits = model(imgs)
                loss = criterion(logits, labels)
                total_loss += loss.item() * imgs.size(0)
                total_acc += (logits.argmax(1) == labels).sum().item()
                total += imgs.size(0)
                all_preds.extend(logits.argmax(1).cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        avg_loss = total_loss / total if total > 0 else 0
        acc = total_acc / total if total > 0 else 0
        return avg_loss, acc, all_preds, all_labels

    def compute_metrics(preds, labels):
        cm = [[0, 0], [0, 0]]  # rows: true [0: non-fog, 1: fog], cols: pred
        for p, t in zip(preds, labels):
            cm[t][p] += 1
        metrics = {}
        for cls in range(num_classes):
            tp = cm[cls][cls]
            fp = sum(cm[r][cls] for r in range(num_classes)) - tp
            fn = sum(cm[cls][c] for c in range(num_classes)) - tp
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
            metrics[class_names[cls]] = {"precision": prec, "recall": rec, "f1": f1, "support": sum(cm[cls])}
        acc = sum(cm[i][i] for i in range(num_classes)) / max(1, sum(sum(r) for r in cm))
        macro_f1 = float(np.mean([metrics[c]["f1"] for c in class_names]))
        return cm, metrics, acc, macro_f1

    # --- Training Loop ---
    print(f"\nStarting training ({args.epochs} epochs, batch size {args.batch_size})...")
    best_val_f1 = 0
    best_val_acc = 0
    best_state = None
    epochs_no_improve = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": [], "lr": []}

    t_start = time.time()
    for epoch in range(1, args.epochs + 1):
        mnet.train()
        running_loss = 0
        running_acc = 0
        seen = 0
        ep_t0 = time.time()

        for i, (imgs, labels, _) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = mnet(imgs)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(mnet.parameters(), 1.0)
            optimizer.step()

            batch_size = imgs.size(0)
            running_loss += loss.item() * batch_size
            running_acc += (logits.argmax(1) == labels).sum().item()
            seen += batch_size

            if (i + 1) % 10 == 0 or (i + 1) == len(train_loader):
                batch_acc = (logits.argmax(1) == labels).float().mean().item()
                print(f"  Epoch {epoch:02d} | Batch {i+1:02d}/{len(train_loader):02d} | loss: {loss.item():.4f} | acc: {batch_acc:.3f}")

        train_loss = running_loss / seen
        train_acc = running_acc / seen

        val_loss, val_acc, val_preds, val_labels = evaluate(val_loader, mnet)
        _, val_metrics, _, val_f1 = compute_metrics(val_preds, val_labels)

        scheduler.step()
        cur_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_f1"].append(val_f1)
        history["lr"].append(cur_lr)

        ep_duration = time.time() - ep_t0
        print(f"--> Epoch {epoch:02d}/{args.epochs:02d} ({ep_duration:.1f}s) | train loss {train_loss:.4f} acc {train_acc:.3f} | val loss {val_loss:.4f} acc {val_acc:.3f} f1 {val_f1:.3f}")
        print(f"    Val per-class: " + " | ".join([f"{c}: P={val_metrics[c]['precision']:.2f} R={val_metrics[c]['recall']:.2f} F1={val_metrics[c]['f1']:.2f}" for c in class_names]))

        if val_f1 > best_val_f1 or (val_f1 == best_val_f1 and val_acc >= best_val_acc):
            best_val_f1 = val_f1
            best_val_acc = val_acc
            best_state = {k: v.cpu() for k, v in mnet.state_dict().items()}
            epochs_no_improve = 0
            torch.save(mnet.state_dict(), out_dir / "best_mobilenetv2_fog.pt")
            print(f"    *** New best val F1 {val_f1:.4f} (acc {val_acc:.3f}) saved to best_mobilenetv2_fog.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"    Early stopping triggered after {args.patience} epochs with no improvement.")
                break

        torch.save(mnet.state_dict(), out_dir / "last_mobilenetv2_fog.pt")

    t_total = time.time() - t_start
    print(f"\nTraining completed in {t_total/60:.2f} minutes.")
    print(f"Best Validation Accuracy: {best_val_acc*100:.2f}% | Best Macro F1: {best_val_f1*100:.2f}%")

    # Load best weights for final evaluation
    if best_state is not None:
        mnet.load_state_dict({k: v.to(device) for k, v in best_state.items()})

    # Save clean state dict
    torch.save(mnet.state_dict(), out_dir / "fog_classifier.pt")
    print(f"Saved deployment PyTorch model to: {out_dir / 'fog_classifier.pt'}")

    # --- Final Evaluation on Val and Test ---
    for split_name, loader in [("val", val_loader), ("test", test_loader)]:
        loss, acc, preds, labels = evaluate(loader, mnet)
        cm, metrics, _, f1 = compute_metrics(preds, labels)
        print(f"\n=== FINAL {split_name.upper()} RESULTS ===")
        print(f"Loss: {loss:.4f} | Accuracy: {acc*100:.2f}% | Macro F1: {f1*100:.2f}%")
        print(f"Confusion Matrix [rows=True, cols=Pred]:")
        print(f"  Non-Fog: TN={cm[0][0]}, FP={cm[0][1]}")
        print(f"  Fog:     FN={cm[1][0]}, TP={cm[1][1]}")
        for c in class_names:
            print(f"  {c:8s}: Precision={metrics[c]['precision']:.3f} Recall={metrics[c]['recall']:.3f} F1={metrics[c]['f1']:.3f} Support={metrics[c]['support']}")

        res = {
            "split": split_name,
            "loss": loss,
            "accuracy": acc,
            "macro_f1": f1,
            "confusion_matrix": cm,
            "per_class": metrics,
            "class_names": class_names
        }
        with open(out_dir / f"metrics_{split_name}.json", "w") as f:
            json.dump(res, f, indent=2)

    # Save history and labels
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)
    with open(out_dir / "labels.json", "w") as f:
        json.dump({"labels": class_names, "indices": {c: i for i, c in enumerate(class_names)}}, f, indent=2)

    # Save model card
    card = {
        "model": "mobilenet_v2",
        "num_classes": num_classes,
        "labels": class_names,
        "image_size": img_size,
        "mean": imagenet_mean,
        "std": imagenet_std,
        "best_val_acc": best_val_acc,
        "best_val_f1": best_val_f1,
        "train_time_min": t_total / 60,
        "epochs_trained": len(history["train_loss"]),
        "dataset": {
            "fog": fog_count,
            "non_fog": non_count,
            "total": fog_count + non_count
        },
        "splits": {k: len(v) for k, v in splits.items()}
    }
    with open(out_dir / "model_card.json", "w") as f:
        json.dump(card, f, indent=2)
    print(f"Saved model card to: {out_dir / 'model_card.json'}")

    # --- Export Standalone ONNX ---
    print("\nExporting ONNX model for browser inference...")
    mnet.eval()
    dummy = torch.randn(1, 3, img_size, img_size, device=device)
    onnx_path = out_dir / "fog_classifier.onnx"

    try:
        torch.onnx.export(
            mnet,
            dummy,
            str(onnx_path),
            input_names=["input"],
            output_names=["logits"],
            dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
            opset_version=18,
            dynamo=False,
            do_constant_folding=True
        )
        print(f"ONNX model exported: {onnx_path} ({onnx_path.stat().st_size / (1024*1024):.2f} MB)")

        try:
            import onnx
            onnx_model = onnx.load(str(onnx_path))
            onnx.checker.check_model(onnx_model)
            print("ONNX model verification PASSED!")
        except Exception as e:
            print(f"ONNX check warning: {e}")
    except Exception as e:
        print(f"ONNX export error: {e}")

    # --- Sanity Inference on Sample Test Images ---
    print("\n--- Sanity Inference Check on 4 Test Samples ---")
    mnet.eval()
    with torch.no_grad():
        for i in range(min(4, len(test_ds))):
            img, label, path = test_ds[i]
            logits = mnet(img.unsqueeze(0).to(device))
            probs = torch.softmax(logits, 1).cpu().numpy()[0]
            pred = int(probs.argmax())
            fname = Path(path).name
            print(f"  Sample {i+1} [{fname}]: GroundTruth={class_names[label]} | Prediction={class_names[pred]} (NonFog: {probs[0]*100:.1f}%, Fog: {probs[1]*100:.1f}%)")

    print("\nTraining and deployment pipeline completed successfully!")

if __name__ == "__main__":
    main()
