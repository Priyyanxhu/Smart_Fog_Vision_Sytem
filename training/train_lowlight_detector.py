r"""
Smart Fog Vision — Road & Low-Light Driving Object Detector Training Pipeline
Dataset: D:\LowLightening (Low-lighting Dash Cam Video Dataset) + Adverse Weather Road Dataset
Target: YOLOv8n fine-tuned on night, low-light, and foggy road traffic
Output: models/road_detector.pt & models/road_detector.onnx (Browser-compatible ONNX for ONNX Runtime Web)
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import os
import json
import time
import shutil
import random
import argparse
from pathlib import Path
from collections import Counter

from PIL import Image
import torch
from ultralytics import YOLO

# Standard 10 Road Classes for Smart Fog Vision
CLASSES = [
    "person",        # 0
    "rider",         # 1
    "car",           # 2
    "bus",           # 3
    "truck",         # 4
    "bike",          # 5
    "motor",         # 6
    "traffic light", # 7
    "traffic sign",  # 8
    "train"          # 9
]
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}

# Mapping from LowLightening labels to Smart Fog Vision classes
LOWLIGHT_LABEL_MAP = {
    "car": "car",
    "van": "car",
    "truck": "truck",
    "bus": "bus",
    "person": "person",
    "electricbicycle": "rider",
    "bicycle": "bike",
    "motorcycle": "motor",
    "motor": "motor",
}


def parse_args():
    p = argparse.ArgumentParser(description="Train Low-Light and Road Object Detector")
    p.add_argument("--lowlight-dir", type=str,
                   default=r"D:\LowLightening\Low lighting Dash Cam Video Dataset_Samples",
                   help="Path to Low-lighting Dash Cam dataset")
    p.add_argument("--road-data-root", type=str, default=r"D:\RoadDetector",
                   help="Path to BDD100K / RoadDetector root")
    p.add_argument("--yolo-dir", type=str, default=r"D:\Smart_Fog_Vision\training\yolo_lowlight_dataset",
                   help="Directory to construct YOLO dataset")
    p.add_argument("--out-dir", type=str, default=r"D:\Smart_Fog_Vision\models",
                   help="Directory for trained models and ONNX")
    p.add_argument("--weights", type=str, default=r"D:\Smart_Fog_Vision\models\road_detector.pt",
                   help="Pretrained weights to fine-tune from")
    p.add_argument("--epochs", type=int, default=10,
                   help="Training epochs")
    p.add_argument("--batch-size", type=int, default=2,
                   help="Batch size")
    p.add_argument("--imgsz", type=int, default=640,
                   help="Image input size")
    p.add_argument("--lr0", type=float, default=0.005,
                   help="Initial learning rate")
    p.add_argument("--seed", type=int, default=42,
                   help="Random seed")
    p.add_argument("--device", type=str, default="cpu",
                   help="Device ('cpu' or '0')")
    return p.parse_args()


def load_lowlight_samples(lowlight_dir):
    """
    Parses D:\LowLightening samples and converts coordinates to YOLO format.
    """
    lowlight_p = Path(lowlight_dir)
    if not lowlight_p.exists():
        print(f"Warning: LowLightening path {lowlight_p} not found.")
        return []

    samples = []
    json_files = sorted(lowlight_p.glob("*.json"))
    print(f"Found {len(json_files)} annotation files in {lowlight_dir}")

    for jf in json_files:
        stem = jf.stem
        # Corresponding base jpg
        img_p = lowlight_p / f"{stem}.jpg"
        if not img_p.exists():
            continue

        try:
            im = Image.open(img_p)
            w_img, h_img = im.size
        except Exception as e:
            print(f"Error opening image {img_p}: {e}")
            continue

        with open(jf, "r", encoding="utf-8") as f:
            data = json.load(f)

        boxes = []
        for obj in data.get("objects", []):
            raw_label = str(obj.get("label", "")).strip().lower()
            mapped_cls = LOWLIGHT_LABEL_MAP.get(raw_label)
            if not mapped_cls or mapped_cls not in CLASS_TO_ID:
                continue

            cls_id = CLASS_TO_ID[mapped_cls]
            coord = obj.get("coordinate", [])
            if len(coord) != 2:
                continue

            (x1, y1), (x2, y2) = coord
            x1, x2 = sorted([float(x1), float(x2)])
            y1, y2 = sorted([float(y1), float(y2)])

            bw = max(0.0, x2 - x1)
            bh = max(0.0, y2 - y1)
            if bw < 4.0 or bh < 4.0:
                continue

            # Normalized YOLO coordinates [x_center, y_center, width, height]
            xc = min(max((x1 + x2) / 2.0 / w_img, 0.0), 1.0)
            yc = min(max((y1 + y2) / 2.0 / h_img, 0.0), 1.0)
            wn = min(max(bw / w_img, 0.0), 1.0)
            hn = min(max(bh / h_img, 0.0), 1.0)

            boxes.append((cls_id, xc, yc, wn, hn))

        if boxes:
            samples.append((img_p, boxes, "lowlight_night"))

    print(f"Successfully processed {len(samples)} low-light images with {sum(len(b) for _, b, _ in samples)} bounding boxes.")
    return samples


def load_adverse_road_samples(road_data_root, max_samples=120, seed=42):
    """
    Loads adverse weather / foggy / night driving samples from RoadDetector if available.
    """
    root = Path(road_data_root)
    img_dir = root / "bdd100k" / "bdd100k" / "images" / "100k" / "val"
    json_p = root / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_val.json"

    if not img_dir.exists() or not json_p.exists():
        print(f"RoadDetector BDD100K not found at {root}; proceeding with LowLightening only.")
        return []

    print(f"Loading adverse weather road samples from {json_p.name}...")
    with open(json_p, "r", encoding="utf-8") as f:
        items = json.load(f)

    # Filter for foggy, rainy, night, or overcast
    def is_adverse(it):
        attrs = it.get("attributes", {})
        w = attrs.get("weather", "").lower()
        t = attrs.get("timeofday", "").lower()
        return (w in ["foggy", "rainy", "snowy", "overcast"] or t in ["night", "dawn/dusk"])

    adverse_items = [it for it in items if is_adverse(it)]
    rng = random.Random(seed)
    rng.shuffle(adverse_items)

    samples = []
    for it in adverse_items:
        name = it.get("name")
        img_p = img_dir / name
        if not img_p.exists():
            continue

        boxes = []
        for lbl in it.get("labels", []):
            cat = lbl.get("category")
            if cat in CLASS_TO_ID and "box2d" in lbl:
                b = lbl["box2d"]
                x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
                bw = max(0.0, x2 - x1)
                bh = max(0.0, y2 - y1)
                if bw >= 4.0 and bh >= 4.0:
                    xc = min(max((x1 + x2) / 2.0 / 1280.0, 0.0), 1.0)
                    yc = min(max((y1 + y2) / 2.0 / 720.0, 0.0), 1.0)
                    wn = min(max(bw / 1280.0, 0.0), 1.0)
                    hn = min(max(bh / 720.0, 0.0), 1.0)
                    boxes.append((CLASS_TO_ID[cat], xc, yc, wn, hn))

        if boxes:
            weather_tag = it.get("attributes", {}).get("weather", "adverse")
            samples.append((img_p, boxes, weather_tag))
            if len(samples) >= max_samples:
                break

    print(f"Selected {len(samples)} adverse weather road samples.")
    return samples


def prepare_dataset(lowlight_samples, road_samples, yolo_dir, seed=42):
    yolo_p = Path(yolo_dir)
    if yolo_p.exists():
        shutil.rmtree(yolo_p, ignore_errors=True)

    for split in ["train", "val"]:
        (yolo_p / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_p / "labels" / split).mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)

    # Lowlight samples: split 8 train, 2 val
    rng.shuffle(lowlight_samples)
    val_low_count = max(1, int(len(lowlight_samples) * 0.2))
    val_low = lowlight_samples[:val_low_count]
    train_low = lowlight_samples[val_low_count:]

    # Road samples: split 80% train, 20% val
    rng.shuffle(road_samples)
    val_road_count = int(len(road_samples) * 0.2)
    val_road = road_samples[:val_road_count]
    train_road = road_samples[val_road_count:]

    train_all = train_low + train_road
    val_all = val_low + val_road

    rng.shuffle(train_all)
    rng.shuffle(val_all)

    print(f"\nConstructing YOLO Dataset in {yolo_p}:")
    print(f"  Train: {len(train_all)} images ({len(train_low)} low-light, {len(train_road)} adverse road)")
    print(f"  Val:   {len(val_all)} images ({len(val_low)} low-light, {len(val_road)} adverse road)")

    def write_split(samples, split_name):
        cat_counts = Counter()
        tot_boxes = 0
        img_out = yolo_p / "images" / split_name
        lbl_out = yolo_p / "labels" / split_name

        for idx, (img_src, boxes, tag) in enumerate(samples):
            fname = f"{split_name}_{idx:04d}_{img_src.stem}.jpg"
            img_dst = img_out / fname
            lbl_dst = lbl_out / f"{img_dst.stem}.txt"

            # Copy or convert image
            try:
                shutil.copy2(img_src, img_dst)
            except Exception:
                im = Image.open(img_src).convert("RGB")
                im.save(img_dst)

            with open(lbl_dst, "w", encoding="utf-8") as lf:
                for cls_id, xc, yc, wn, hn in boxes:
                    lf.write(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
                    cat_counts[CLASSES[cls_id]] += 1
                    tot_boxes += 1

        print(f"  -> {split_name.upper()} wrote {len(samples)} images, {tot_boxes} boxes.")
        return cat_counts, tot_boxes

    train_dist, train_b = write_split(train_all, "train")
    val_dist, val_b = write_split(val_all, "val")

    # Generate data.yaml
    yaml_p = yolo_p / "dataset.yaml"
    yaml_text = f"""# Smart Fog Vision — Low-Light & Road Detector Dataset
path: {yolo_p.as_posix()}
train: images/train
val: images/val

names:
"""
    for i, c in enumerate(CLASSES):
        yaml_text += f"  {i}: {c}\n"

    with open(yaml_p, "w", encoding="utf-8") as yf:
        yf.write(yaml_text)

    return yaml_p, {
        "train_images": len(train_all),
        "val_images": len(val_all),
        "train_boxes": train_b,
        "val_boxes": val_b,
        "train_dist": dict(train_dist),
        "val_dist": dict(val_dist)
    }


def train_lowlight_detector():
    args = parse_args()
    print("=" * 64)
    print("Smart Fog Vision — Low-Light & Road Object Detector Training")
    print("=" * 64)

    # 1. Ingest D:\LowLightening
    lowlight_samples = load_lowlight_samples(args.lowlight_dir)

    # 2. Ingest adverse road samples
    road_samples = load_adverse_road_samples(args.road_data_root, max_samples=30, seed=args.seed)

    if not lowlight_samples and not road_samples:
        raise RuntimeError("No training samples found in either LowLightening or RoadDetector!")

    # 3. Build YOLO dataset
    yaml_path, stats = prepare_dataset(lowlight_samples, road_samples, args.yolo_dir, seed=args.seed)

    # 4. Initialize YOLOv8 model
    weights_path = args.weights
    if not Path(weights_path).exists():
        fallback_w = Path(r"D:\Smart_Fog_Vision\yolov8n.pt")
        weights_path = str(fallback_w) if fallback_w.exists() else "yolov8n.pt"

    print(f"\n[Training] Loading model weights: {weights_path}")
    import gc
    gc.collect()
    model = YOLO(weights_path)

    # 5. Fine-tune YOLOv8n with low-memory CPU settings
    t0 = time.time()
    print(f"Starting training on device={args.device}, epochs={args.epochs}, imgsz={args.imgsz}, batch={args.batch_size}...")
    train_results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        lr0=args.lr0,
        device=args.device,
        seed=args.seed,
        workers=0,
        cache=False,
        verbose=True,
        plots=False,
        save=True,
        project=str(Path(args.out_dir) / "yolo_runs"),
        name="lowlight_road_train",
        exist_ok=True
    )
    train_min = (time.time() - t0) / 60.0
    print(f"Training finished in {train_min:.2f} minutes.")

    # 6. Evaluate model
    print("\n[Evaluation] Validating model on val split...")
    val_results = model.val(data=str(yaml_path), batch=args.batch_size, imgsz=args.imgsz, device=args.device, verbose=False)
    metrics_map = val_results.results_dict
    print("Validation Metrics:")
    for k, v in metrics_map.items():
        print(f"  {k}: {v:.4f}")

    # 7. Save trained weights to models/
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trained_pt = out_dir / "road_detector.pt"
    # Copy best weights from run directory
    best_run_pt = Path(model.trainer.best) if hasattr(model, "trainer") and hasattr(model.trainer, "best") else None
    if best_run_pt and best_run_pt.exists():
        shutil.copy2(best_run_pt, trained_pt)
        print(f"Copied best checkpoint to: {trained_pt}")
    else:
        model.save(str(trained_pt))
        print(f"Saved weights to: {trained_pt}")

    # 8. Export ONNX (input [1, 3, 640, 640] -> output [1, 14, 8400])
    onnx_path = out_dir / "road_detector.onnx"
    print(f"\n[Export] Exporting to browser ONNX Runtime Web format: {onnx_path}...")
    export_result = model.export(
        format="onnx",
        imgsz=args.imgsz,
        dynamic=False,
        opset=17,
        simplify=True
    )
    # Ensure it's in models/road_detector.onnx
    exported_onnx_p = Path(export_result) if export_result else None
    if exported_onnx_p and exported_onnx_p.exists() and exported_onnx_p != onnx_path:
        shutil.copy2(exported_onnx_p, onnx_path)
    print(f"ONNX Model saved to: {onnx_path} ({onnx_path.stat().st_size / (1024*1024):.2f} MB)")

    # 9. Write road classes JSON
    classes_json = out_dir / "road_classes.json"
    with open(classes_json, "w", encoding="utf-8") as f:
        json.dump(CLASSES, f, indent=2)
    print(f"Saved road classes -> {classes_json}")

    # 10. Write model card
    card = {
        "model_name": "YOLOv8n Low-Light & Road Detector",
        "base_model": "yolov8n",
        "datasets": [
            "D:\\LowLightening (Low lighting Dash Cam Video Dataset)",
            "D:\\RoadDetector (BDD100K Adverse Weather Driving)"
        ],
        "input_size": args.imgsz,
        "classes": CLASSES,
        "num_classes": len(CLASSES),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_time_min": round(train_min, 2),
        "onnx_size_mb": round(onnx_path.stat().st_size / (1024 * 1024), 2),
        "metrics": {
            "mAP50": float(metrics_map.get("metrics/mAP50(B)", 0.0)),
            "mAP50_95": float(metrics_map.get("metrics/mAP50-95(B)", 0.0)),
            "precision": float(metrics_map.get("metrics/precision(B)", 0.0)),
            "recall": float(metrics_map.get("metrics/recall(B)", 0.0)),
        },
        "dataset_statistics": stats
    }
    with open(out_dir / "road_detector_card.json", "w", encoding="utf-8") as f:
        json.dump(card, f, indent=2)
    print(f"Model card written -> {out_dir / 'road_detector_card.json'}")
    print("\nLow-Light & Road Detector Training Pipeline complete!")


if __name__ == "__main__":
    train_lowlight_detector()
