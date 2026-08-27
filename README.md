# Smart Fog Vision — AI Dehazing & Hazard Alert System

**Live website:** `http://localhost:8001` (already running) | Open `index.html` directly or via any static server.

AI-powered computer vision website that captures real-time video/images in foggy conditions and returns a **clear enhanced stream** with **YOLO object detection, distance estimation, multi-object tracking, and voice/visual alerts** for the driver.

---

## Features

| Pipeline Stage | Technique |
|---|---|
| **Fog Detection** | Dark Channel Prior (DCP) mean + contrast variance → fog density %, visibility (m), level |
| **Dehazing / Enhancement** | DCP transmission `t = 1 - ω·dark/A`, guided-filter (box-blur approx) smoothing, scene recovery `J=(I-A)/t +A`, CLAHE-like gain, unsharp mask sharpen |
| **Object Detection** | YOLO-based: COCO-SSD (`lite_mobilenet_v2` via TF.js) by default, optional **YOLOv8n ONNX** via `onnxruntime-web` (80 COCO classes: car, person, bus, truck, etc.) |
| **Distance Estimation** | Pinhole model `dist = (realHeight · focal)/bboxHeight` — calibrated per class (car 1.5m, person 1.7m, bus 3m…) |
| **Tracking** | IOU tracker (threshold 0.32, maxAge 15) → persistent IDs across frames |
| **Alerts** | Color-coded hazard levels (DANGER <12m, WARNING <25m, CAUTION <45m) + top banner + log + **Web Speech API** voice warnings (throttled) |

**Dual-feed UI:** LEFT = raw foggy feed, RIGHT = enhanced dehazed + YOLO overlay. Real-time metrics (FPS, latency, fog %, visibility, nearest hazard) + processing pipeline controls.

---

## Quick Start

### Option 1 — Python (already running on port 8001)
```powershell
python -m http.server 8001 --directory "D:\Smart_Fog_Vision"
# open http://localhost:8001
```

### Option 2 — VS Code Live Server / any static server
Open `D:\Smart_Fog_Vision\index.html` directly — but camera requires `http://` (not `file://`). Use a local server.

### Option 3 — Node
```powershell
npx serve D:\Smart_Fog_Vision -l 8001
```

No build step, no `npm install` required. All libs via CDN.

---

## How to Use

1. **START CAMERA** → allow webcam permission → left feed shows raw, right shows enhanced dehazed stream.
2. **IMAGE / VIDEO** buttons → upload a foggy traffic image or dashcam video to test without a camera.
3. **DEMO FOG** → overlays synthetic fog on any input to demo the dehazer when you don't have a foggy scene.
4. Tune pipeline in real-time:
   - `Strength (ω)` 0–1 (dehaze aggressiveness, default 0.95)
   - `CLAHE` gain 0.8–2.0 + Sharpen toggle
   - `Confidence` 0.1–0.9 for YOLO, model switch COCO-SSD ↔ YOLOv8n
5. **SNAPSHOT** saves enhanced PNG, **REC** records enhanced canvas to `webm`.
6. Voice alerts toggle with `VOICE ON/OFF`; alerts auto-trigger when object <18 m or visibility very low.

Keyboard: `Space` = start/stop, `F` = toggle demo fog.

---

## Project Structure

```
Smart_Fog_Vision/
├── index.html   — Tailwind UI, dual video grid, metrics, controls
├── style.css    — custom scrollbars & alert styles
├── app.js       — main loop: capture → fog metrics → dehaze → detect → track → alert → render
├── dehaze.js    — DCP + atmospheric light + transmission + CLAHE + sharpen + fog synthesis
├── detector.js  — COCO-SSD + YOLOv8 ONNX (ort) wrapper, NMS, letterbox preprocess
└── tracker.js   — IOU tracker + distance estimator + hazardLevel()
```

All processing is **100% on-device in the browser** (TF.js, ONNX WASM). No server or API key.

---

## Technical Notes

- **Input resolution** auto-capped to `640px` wide for real-time performance; detection throttled to ~7 FPS.
- **YOLOv8n ONNX** is ~6 MB, loaded on-demand from `cdn.jsdelivr.net/gh/ultralytics/assets`. Falls back to COCO-SSD on CORS/failure.
- **Fog metric** formula: `fog = 0.65·darkMean + 0.35·(1-contrastNorm)`; visibility `1000·(1-fog)^1.4`.
- Tested on Chrome/Edge desktop. Firefox requires `mediaDevices` permission. Mobile works via rear camera (`facingMode: environment`).

---

## Credits

Built with TailwindCSS, TensorFlow.js, COCO-SSD, ONNX Runtime Web, Font Awesome. Inspired by Dark Channel Prior (He et al. 2009) + YOLO.

© 2026 Smart Fog Vision — on-device, privacy-preserving driver assistance.
