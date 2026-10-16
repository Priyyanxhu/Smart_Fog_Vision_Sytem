"""Test inference with the trained BDD100K road detector (.pt and .onnx)."""
import sys
import json
from pathlib import Path
from PIL import Image
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO

MODEL_DIR = Path(r"D:\Smart_Fog_Vision\models")
PT_PATH = MODEL_DIR / "road_detector.pt"
ONNX_PATH = MODEL_DIR / "road_detector.onnx"
CLASSES_PATH = MODEL_DIR / "road_classes.json"

with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    CLASSES = json.load(f)

def test_pt(image_path):
    print(f"\n--- Testing PyTorch Model ({PT_PATH}) ---")
    model = YOLO(str(PT_PATH))
    results = model.predict(source=str(image_path), conf=0.25, verbose=False)
    boxes = results[0].boxes
    print(f"Detected {len(boxes)} objects:")
    for b in boxes:
        cls_id = int(b.cls[0].item())
        cls_name = CLASSES[cls_id] if cls_id < len(CLASSES) else f"cls{cls_id}"
        conf = float(b.conf[0].item())
        xywh = [round(x, 1) for x in b.xywh[0].tolist()]
        print(f"  [{cls_name:13s}] conf: {conf:.2f} | box: {xywh}")

def test_onnx(image_path):
    print(f"\n--- Testing ONNX Model ({ONNX_PATH}) ---")
    session = ort.InferenceSession(str(ONNX_PATH), providers=["CPUExecutionProvider"])
    
    # Preprocess 640x640 letterbox
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    scale = min(640 / w, 640 / h)
    nw, nh = int(w * scale), int(h * scale)
    im_resized = im.resize((nw, nh), Image.Resampling.BILINEAR)
    canvas = Image.new("RGB", (640, 640), (0, 0, 0))
    canvas.paste(im_resized, ((640 - nw) // 2, (640 - nh) // 2))
    
    arr = np.array(canvas, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))  # HWC -> CHW
    tensor = np.expand_dims(arr, axis=0) # 1, 3, 640, 640
    
    inputs = {session.get_inputs()[0].name: tensor}
    outputs = session.run(None, inputs)
    out = outputs[0] # [1, 14, 8400]
    print(f"ONNX output shape: {out.shape}")
    
    # Decode detections
    preds = out[0] # [14, 8400]
    num_boxes = preds.shape[1]
    detections = []
    for i in range(num_boxes):
        scores = preds[4:, i]
        max_idx = int(np.argmax(scores))
        max_score = float(scores[max_idx])
        if max_score >= 0.25:
            cls_name = CLASSES[max_idx] if max_idx < len(CLASSES) else f"cls{max_idx}"
            detections.append((cls_name, max_score))
            
    print(f"Detected {len(detections)} candidate box hits (conf >= 0.25):")
    for cls_name, score in detections[:8]:
        print(f"  [{cls_name:13s}] conf: {score:.2f}")

if __name__ == "__main__":
    sample_img = Path(r"D:\RoadDetector\bdd100k\bdd100k\images\100k\val\b1c66a42-6f7d68ca.jpg")
    if len(sys.argv) > 1:
        sample_img = Path(sys.argv[1])
    print(f"Testing Road Detector on: {sample_img}")
    test_pt(sample_img)
    test_onnx(sample_img)
    print("\nAll model checks completed successfully!")
