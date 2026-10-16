r"""
Smart Fog Vision — Road Detector Training Pipeline & ONNX Export
Dataset: BDD100K (D:\RoadDetector)
Model: YOLOv8n fine-tuned on road objects (cars, pedestrians, trucks, buses, etc.)
Target: PyTorch weights (.pt) + Browser-compatible ONNX (.onnx) for ONNX Runtime Web
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

import torch
from ultralytics import YOLO

# Standard BDD100K 10 Road Object Classes
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


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8 Road Detector on BDD100K")
    parser.add_argument("--data-root", type=str, default=r"D:\RoadDetector",
                        help="Path to RoadDetector / BDD100K root")
    parser.add_argument("--out-dir", type=str, default=r"D:\Smart_Fog_Vision\models",
                        help="Output directory for trained models and metrics")
    parser.add_argument("--yolo-data-dir", type=str, default=r"D:\RoadDetector\yolo_bdd100k",
                        help="Directory to build YOLO formatted dataset")
    parser.add_argument("--train-samples", type=int, default=600,
                        help="Number of training images to use (-1 for all)")
    parser.add_argument("--val-samples", type=int, default=120,
                        help="Number of validation images to use (-1 for all)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640,
                        help="Input image size")
    parser.add_argument("--lr0", type=float, default=0.01,
                        help="Initial learning rate")
    default_w = r"D:\Smart_Fog_Vision\models\road_detector.pt" if Path(r"D:\Smart_Fog_Vision\models\road_detector.pt").exists() else "yolov8n.pt"
    parser.add_argument("--weights", type=str, default=default_w,
                        help="Pretrained weights to fine-tune from")
    parser.add_argument("--device", type=str, default="cpu",
                        help="Device: 'cpu' or '0'")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    return parser.parse_args()


def prepare_yolo_dataset(data_root, yolo_dir, train_count=500, val_count=100, seed=42):
    """
    Parses BDD100K annotations and creates a YOLO format dataset with images and labels.
    Uses hardlinks to avoid duplicating disk space.
    """
    data_root = Path(data_root)
    yolo_dir = Path(yolo_dir)

    images_dir = data_root / "bdd100k" / "bdd100k" / "images" / "100k" / "val"
    val_json_path = data_root / "bdd100k_labels_release" / "bdd100k" / "labels" / "bdd100k_labels_images_val.json"

    if not images_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {images_dir}")
    if not val_json_path.exists():
        raise FileNotFoundError(f"Annotation file not found: {val_json_path}")

    print(f"\n[1/4] Preparing YOLO dataset in: {yolo_dir}")
    print(f"Reading BDD100K annotations from {val_json_path}...")
    with open(val_json_path, "r", encoding="utf-8") as f:
        all_items = json.load(f)

    print(f"Loaded {len(all_items)} annotated items.")

    # Filter to items that have at least 1 valid bounding box and image exists on disk
    valid_items = []
    for item in all_items:
        name = item.get("name")
        if not name:
            continue
        img_file = images_dir / name
        if not img_file.exists():
            continue

        boxes = []
        for lbl in item.get("labels", []):
            cat = lbl.get("category")
            if cat in CLASS_TO_ID and "box2d" in lbl:
                b = lbl["box2d"]
                x1, y1, x2, y2 = float(b["x1"]), float(b["y1"]), float(b["x2"]), float(b["y2"])
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                if w >= 2.0 and h >= 2.0:
                    xc = min(max((x1 + x2) / 2.0 / 1280.0, 0.0), 1.0)
                    yc = min(max((y1 + y2) / 2.0 / 720.0, 0.0), 1.0)
                    wn = min(max(w / 1280.0, 0.0), 1.0)
                    hn = min(max(h / 720.0, 0.0), 1.0)
                    boxes.append((CLASS_TO_ID[cat], xc, yc, wn, hn))

        if boxes:
            valid_items.append((item, boxes))

    print(f"Total images with valid road object annotations: {len(valid_items)}")

    # Score each image by adverse weather & low-visibility conditions
    def adverse_score(it):
        attrs = it.get("attributes", {})
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

    # Shuffle with fixed seed for reproducibility, then sort by adverse score (highest first)
    rng = random.Random(seed)
    rng.shuffle(valid_items)
    valid_items.sort(key=lambda x: adverse_score(x[0]), reverse=True)

    total_needed = train_count + val_count
    if total_needed > len(valid_items):
        train_count = int(len(valid_items) * 0.8)
        val_count = len(valid_items) - train_count

    train_subset = valid_items[:train_count]
    val_subset = valid_items[train_count:train_count + val_count]

    train_weather = Counter(item.get("attributes", {}).get("weather", "unknown") for item, _ in train_subset)
    train_time = Counter(item.get("attributes", {}).get("timeofday", "unknown") for item, _ in train_subset)
    val_weather = Counter(item.get("attributes", {}).get("weather", "unknown") for item, _ in val_subset)
    print(f"\n[Adverse Weather Sampling Stats]")
    print(f"  Train Weather:     {dict(train_weather)}")
    print(f"  Train Time-of-Day: {dict(train_time)}")
    print(f"  Val Weather:       {dict(val_weather)}")

    # Setup directories
    for split in ["train", "val"]:
        (yolo_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (yolo_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    def write_split(subset, split_name):
        cat_counts = Counter()
        total_boxes = 0
        img_dest_dir = yolo_dir / "images" / split_name
        lbl_dest_dir = yolo_dir / "labels" / split_name

        for item, boxes in subset:
            name = item["name"]
            stem = Path(name).stem
            src_img = images_dir / name
            dst_img = img_dest_dir / name

            # Link or copy image
            if not dst_img.exists():
                try:
                    os.link(src_img, dst_img)
                except Exception:
                    shutil.copy2(src_img, dst_img)

            # Write label txt
            lbl_file = lbl_dest_dir / f"{stem}.txt"
            with open(lbl_file, "w", encoding="utf-8") as lf:
                for cls_id, xc, yc, wn, hn in boxes:
                    lf.write(f"{cls_id} {xc:.6f} {yc:.6f} {wn:.6f} {hn:.6f}\n")
                    cat_counts[CLASSES[cls_id]] += 1
                    total_boxes += 1

        print(f"  {split_name.upper():5s}: {len(subset)} images | {total_boxes} boxes")
        return cat_counts, total_boxes

    print("Writing YOLO images and annotations:")
    train_cats, train_boxes = write_split(train_subset, "train")
    val_cats, val_boxes = write_split(val_subset, "val")

    # Generate data.yaml with forward slashes for cross-platform compatibility
    yaml_path = yolo_dir / "road_detector.yaml"
    yaml_content = f"""# Ultralytics YOLOv8 Road Detector Dataset (BDD100K)
path: {yolo_dir.as_posix()}
train: images/train
val: images/val

names:
"""
    for i, name in enumerate(CLASSES):
        yaml_content += f"  {i}: {name}\n"

    with open(yaml_path, "w", encoding="utf-8") as yf:
        yf.write(yaml_content)

    print(f"Dataset YAML written to: {yaml_path}")
    return yaml_path, {
        "train_images": len(train_subset),
        "val_images": len(val_subset),
        "train_boxes": train_boxes,
        "val_boxes": val_boxes,
        "train_distribution": dict(train_cats),
        "val_distribution": dict(val_cats),
        "train_weather": dict(train_weather),
        "train_time_of_day": dict(train_time),
        "val_weather": dict(val_weather)
    }


def train_model(yaml_path, args):
    """Fine-tunes YOLOv8n on the BDD100K road dataset."""
    print(f"\n[2/4] Initializing YOLOv8n fine-tuning ({args.weights})...")
    print(f"Epochs: {args.epochs} | Batch size: {args.batch_size} | Imgsz: {args.imgsz} | Device: {args.device}")

    model = YOLO(args.weights)

    start_time = time.time()
    results = model.train(
        data=str(yaml_path),
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.imgsz,
        lr0=args.lr0,
        device=args.device,
        seed=args.seed,
        workers=0,
        verbose=True,
        plots=False,
        save=True,
        project=str(Path(args.out_dir) / "yolo_runs"),
        name="bdd100k_road"
    )
    elapsed_sec = time.time() - start_time
    print(f"Training completed in {elapsed_sec/60:.2f} minutes!")

    # Locate best trained weights (handle ultralytics auto-increment: bdd100k_road, bdd100k_road-2, ...)
    runs_root = Path(args.out_dir) / "yolo_runs"
    candidates = sorted(runs_root.glob("bdd100k_road*"), key=lambda p: p.stat().st_mtime) if runs_root.exists() else []
    run_dir = candidates[-1] if candidates else (Path(args.out_dir) / "yolo_runs" / "bdd100k_road")
    best_pt = run_dir / "weights" / "best.pt"
    if not best_pt.exists():
        best_pt = run_dir / "weights" / "last.pt"

    return model, best_pt, results, elapsed_sec


def export_and_package(best_pt, args, dataset_stats, elapsed_sec):
    """Exports the trained weights to ONNX and packages metadata."""
    print(f"\n[3/4] Exporting model to browser-compatible ONNX...")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trained_model = YOLO(str(best_pt))

    # Export ONNX with imgsz=640 and opset=17
    onnx_file = trained_model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=17,
        simplify=False
    )

    dest_pt = out_dir / "road_detector.pt"
    dest_onnx = out_dir / "road_detector.onnx"
    dest_classes = out_dir / "road_classes.json"
    dest_card = out_dir / "road_detector_card.json"

    shutil.copy2(best_pt, dest_pt)
    shutil.copy2(onnx_file, dest_onnx)

    with open(dest_classes, "w", encoding="utf-8") as f:
        json.dump(CLASSES, f, indent=2)

    # Validate ONNX
    import onnx
    m = onnx.load(str(dest_onnx))
    onnx.checker.check_model(m)
    input_shapes = [(i.name, [d.dim_value for d in i.type.tensor_type.shape.dim]) for i in m.graph.input]
    output_shapes = [(o.name, [d.dim_value for d in o.type.tensor_type.shape.dim]) for o in m.graph.output]

    print(f"ONNX Model successfully verified!")
    print(f"  Inputs:  {input_shapes}")
    print(f"  Outputs: {output_shapes}")
    print(f"  File size: {dest_onnx.stat().st_size / (1024*1024):.2f} MB")

    # Generate Model Card
    model_card = {
        "model_name": "YOLOv8n Road Detector (BDD100K)",
        "base_model": "yolov8n",
        "dataset": "BDD100K Autonomous Driving (D:\\RoadDetector)",
        "input_size": args.imgsz,
        "classes": CLASSES,
        "num_classes": len(CLASSES),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "train_time_min": round(elapsed_sec / 60, 2),
        "onnx_size_mb": round(dest_onnx.stat().st_size / (1024*1024), 2),
        "dataset_statistics": dataset_stats
    }

    with open(dest_card, "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)

    print(f"\n[4/4] Model packaging complete!")
    print(f"  PyTorch weights: {dest_pt}")
    print(f"  ONNX web model:  {dest_onnx}")
    print(f"  Class labels:    {dest_classes}")
    print(f"  Model card:      {dest_card}")


def main():
    args = parse_args()
    print("=" * 60)
    print("Smart Fog Vision — YOLOv8 Road Detector Training (BDD100K)")
    print(f"Data Root: {args.data_root}")
    print(f"Out Dir:   {args.out_dir}")
    print("=" * 60)

    yaml_path, dataset_stats = prepare_yolo_dataset(
        args.data_root,
        args.yolo_data_dir,
        train_count=args.train_samples,
        val_count=args.val_samples,
        seed=args.seed
    )

    model, best_pt, results, elapsed_sec = train_model(yaml_path, args)
    export_and_package(best_pt, args, dataset_stats, elapsed_sec)
    print("\nTraining and deployment pipeline completed successfully!")


if __name__ == "__main__":
    main()
