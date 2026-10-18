r"""
Smart Fog Vision — Master Training Pipeline
Trains both core neural networks:
  1. Road Segmentation Model (TinyUNet) on D:\RoadSegmentation
  2. Low-Light & Road Object Detector (YOLOv8n) on D:\LowLightening + Adverse Weather

Usage:
  python training/train_all.py --all
  python training/train_all.py --seg-only
  python training/train_all.py --detector-only
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import argparse
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description="Master Training Runner for Smart Fog Vision")
    p.add_argument("--all", action="store_true", help="Train both Road Segmentation and Low-Light Detector")
    p.add_argument("--seg-only", action="store_true", help="Train only Road Segmentation model")
    p.add_argument("--detector-only", action="store_true", help="Train only Low-Light Road Detector")
    p.add_argument("--lowlight-dir", type=str,
                   default=r"D:\LowLightening\Low lighting Dash Cam Video Dataset_Samples")
    p.add_argument("--road-seg-dir", type=str, default=r"D:\RoadSegmentation")
    p.add_argument("--seg-epochs", type=int, default=6)
    p.add_argument("--det-epochs", type=int, default=3)
    args = p.parse_args()

    train_seg = args.seg_only or args.all or (not args.detector-only and not args.seg_only)
    train_det = args.detector_only or args.all or (not args.detector-only and not args.seg_only)

    base_dir = Path(__file__).resolve().parent

    print("=" * 70)
    print(" Smart Fog Vision — Multi-Model Training Pipeline")
    print("=" * 70)

    if train_seg:
        print("\n>>> [1/2] Training Road Segmentation Model (D:\\RoadSegmentation)...")
        seg_script = base_dir / "train_road_seg_custom.py"
        cmd = [
            sys.executable, str(seg_script),
            "--data-root", args.road_seg_dir,
            "--epochs", str(args.seg_epochs),
            "--batch-size", "4"
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"Error during Road Segmentation training (exit code {res.returncode})")
        else:
            print("Road Segmentation training completed successfully.")

    if train_det:
        print("\n>>> [2/2] Training Low-Light Road Detector (D:\\LowLightening)...")
        det_script = base_dir / "train_lowlight_detector.py"
        cmd = [
            sys.executable, str(det_script),
            "--lowlight-dir", args.lowlight_dir,
            "--epochs", str(args.det_epochs),
            "--batch-size", "2"
        ]
        res = subprocess.run(cmd)
        if res.returncode != 0:
            print(f"Error during Low-Light Detector training (exit code {res.returncode})")
        else:
            print("Low-Light Detector training completed successfully.")

    print("\n" + "=" * 70)
    print(" All training tasks finished. ONNX models ready in models/")
    print("=" * 70)


if __name__ == "__main__":
    main()
