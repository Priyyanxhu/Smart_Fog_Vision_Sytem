(function(){
// road_seg.js — Browser ONNX road segmentation (TinyUNet, 256, RGB /255, sigmoid>0.5)
// Model: models/road_seg.onnx (BDD100K, road id 0 -> binary). Runs on-device via ONNX Runtime Web.
let segSession = null;
const SEG_SIZE = 256;

async function loadRoadSeg(url="models/road_seg.onnx") {
  if (segSession) return segSession;
  try {
    // Ensure WASM files resolve to CDN (avoids 404 on localhost)
    try{
      if(ort && ort.env && ort.env.wasm){
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.simd = true;
      }
    }catch{}
    segSession = await ort.InferenceSession.create(url, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    console.log("[RoadSeg] loaded", url);
    return segSession;
  } catch (e) {
    console.warn("[RoadSeg] load failed", e);
    segSession = null;
    throw e;
  }
}

function preprocessCanvas(canvas, size=SEG_SIZE){
  const off = document.createElement("canvas");
  off.width = size; off.height = size;
  const ctx = off.getContext("2d", { willReadFrequently: true });
  // Stretch to square (matches training resize to 256)
  ctx.drawImage(canvas, 0, 0, size, size);
  const data = ctx.getImageData(0,0,size,size).data;
  const float = new Float32Array(3*size*size);
  for(let y=0; y<size; y++){
    for(let x=0; x<size; x++){
      const idx = (y*size + x)*4;
      const p = y*size + x;
      float[p] = data[idx]/255;
      float[p + size*size] = data[idx+1]/255;
      float[p + 2*size*size] = data[idx+2]/255;
    }
  }
  return new ort.Tensor("float32", float, [1,3,size,size]);
}

async function predictRoadMask(canvas){
  if(!segSession) throw new Error("Road seg model not loaded");
  const tensor = preprocessCanvas(canvas);
  const feeds = {};
  feeds[segSession.inputNames[0]] = tensor;
  const out = await segSession.run(feeds);
  const logits = out[segSession.outputNames[0]].data;
  const mask = new Uint8Array(SEG_SIZE*SEG_SIZE);
  let road = 0;
  for(let i=0; i<mask.length; i++){
    const prob = 1/(1+Math.exp(-logits[i]));
    if(prob > 0.5){ mask[i] = 1; road++; }
  }
  return { mask, size: SEG_SIZE, roadFrac: road / mask.length };
}

function isSegReady(){ return !!segSession; }
function getSegSize(){ return SEG_SIZE; }

window.SFVMods = window.SFVMods || {};
window.SFVMods.roadSeg = { loadRoadSeg, predictRoadMask, isSegReady, getSegSize };
})();
