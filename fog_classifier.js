(function(){
// fog_classifier.js — Browser ONNX fog vs non-fog (MobileNetV2, 224, ImageNet norm)
let fogSession = null;
let fogLabels = ["non-fog","fog"];
const FOG_SIZE = 224;
const FOG_MEAN = [0.485, 0.456, 0.406];
const FOG_STD  = [0.229, 0.224, 0.225];

async function loadFogClassifier(url="models/fog_classifier.onnx") {
  if (fogSession) return fogSession;
  try {
    // Ensure WASM files resolve to CDN (avoids 404 on localhost)
    try{
      if(ort && ort.env && ort.env.wasm){
        ort.env.wasm.wasmPaths = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.19.2/dist/";
        ort.env.wasm.numThreads = 1;
        ort.env.wasm.simd = true;
      }
    }catch{}
    // Load labels if available
    try {
      const r = await fetch("models/labels.json");
      if (r.ok) {
        const j = await r.json();
        if (j.labels) fogLabels = j.labels;
      }
    } catch {}
    fogSession = await ort.InferenceSession.create(url, { executionProviders: ["wasm"], graphOptimizationLevel: "all" });
    console.log("[FogClassifier] loaded", url, "labels", fogLabels);
    return fogSession;
  } catch (e) {
    console.warn("[FogClassifier] load failed", e);
    fogSession = null;
    throw e;
  }
}

function preprocessCanvas(canvas, size=FOG_SIZE){
  const off = document.createElement("canvas");
  off.width = size; off.height = size;
  const ctx = off.getContext("2d", { willReadFrequently: true });
  // Keep aspect via center crop after resize? Simple: resize to fill via drawImage with cover logic
  // Use object-fit: cover equivalent: compute scale to fill
  const scale = Math.max(size / canvas.width, size / canvas.height);
  const nw = canvas.width * scale, nh = canvas.height * scale;
  const dx = (size - nw) / 2, dy = (size - nh) / 2;
  ctx.drawImage(canvas, dx, dy, nw, nh);
  // Could also just stretch: ctx.drawImage(canvas, 0,0,size,size) but cover is better for classification
  const data = ctx.getImageData(0,0,size,size).data;
  const float = new Float32Array(3*size*size);
  for(let y=0; y<size; y++){
    for(let x=0; x<size; x++){
      const idx = (y*size + x)*4;
      const r = data[idx]/255, g=data[idx+1]/255, b=data[idx+2]/255;
      const rn = (r - FOG_MEAN[0]) / FOG_STD[0];
      const gn = (g - FOG_MEAN[1]) / FOG_STD[1];
      const bn = (b - FOG_MEAN[2]) / FOG_STD[2];
      const p = y*size + x;
      float[p] = rn;
      float[p + size*size] = gn;
      float[p + 2*size*size] = bn;
    }
  }
  return new ort.Tensor("float32", float, [1,3,size,size]);
}

async function predictFog(canvas){
  if(!fogSession) throw new Error("Fog model not loaded");
  const tensor = preprocessCanvas(canvas);
  const feeds = {};
  const inputName = fogSession.inputNames[0];
  feeds[inputName] = tensor;
  const out = await fogSession.run(feeds);
  const outName = fogSession.outputNames[0];
  const logits = out[outName].data;
  // logits [2]
  // softmax
  const maxLogit = Math.max(logits[0], logits[1]);
  const exp0 = Math.exp(logits[0]-maxLogit);
  const exp1 = Math.exp(logits[1]-maxLogit);
  const sum = exp0+exp1;
  const p0 = exp0/sum, p1 = exp1/sum; // non-fog, fog
  const predIdx = p1 > p0 ? 1 : 0;
  return {
    pFog: p1,
    pNonFog: p0,
    label: fogLabels[predIdx],
    confidence: Math.max(p0,p1),
    logits: [logits[0], logits[1]],
    predIdx
  };
}

function getFogLabels(){ return fogLabels; }
function isFogReady(){ return !!fogSession; }

window.SFVMods = window.SFVMods || {};
window.SFVMods.fogClassifier = { loadFogClassifier, predictFog, getFogLabels, isFogReady };
})();
