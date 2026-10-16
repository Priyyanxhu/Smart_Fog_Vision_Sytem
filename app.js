import { estimateFogMetrics, dehazeImageData, addFogEffect } from './dehaze.js';
import { Tracker, hazardLevel } from './tracker.js';
import { loadCoco, loadYolo, detectCoco, detectYolo } from './detector.js';
import { loadFogClassifier, predictFog, isFogReady } from './fog_classifier.js';
import { loadRoadSeg, predictRoadMask, isSegReady } from './road_seg.js';

// DOM
const video = document.getElementById('video');
const fileVideo = document.getElementById('fileVideo');
const canvasRaw = document.getElementById('canvasRaw');
const ctxRaw = canvasRaw.getContext('2d', { willReadFrequently: true });
const canvasEnhanced = document.getElementById('canvasEnhanced');
const ctxEnh = canvasEnhanced.getContext('2d', { willReadFrequently: true });
const overlayRaw = document.getElementById('overlayRaw');
const oRaw = overlayRaw.getContext('2d');
const overlayEnhanced = document.getElementById('overlayEnhanced');
const oEnh = overlayEnhanced.getContext('2d');
// Offscreen composite = what the user sees on the right (clear enhanced + YOLO boxes).
// Snapshots and video recordings export from here so you get true clear output.
const exportCanvas = document.createElement('canvas');
const exportCtx = exportCanvas.getContext('2d');
function syncExportCanvas(){
  if(exportCanvas.width !== canvasEnhanced.width || exportCanvas.height !== canvasEnhanced.height){
    exportCanvas.width = canvasEnhanced.width;
    exportCanvas.height = canvasEnhanced.height;
  }
}
function updateExportFrame(){
  syncExportCanvas();
  exportCtx.drawImage(canvasEnhanced, 0, 0, exportCanvas.width, exportCanvas.height);
  exportCtx.drawImage(overlayEnhanced, 0, 0, exportCanvas.width, exportCanvas.height);
}

const btnStart = document.getElementById('btnStart');
const btnStop = document.getElementById('btnStop');
const btnDemoFog = document.getElementById('btnDemoFog');
const placeholderRaw = document.getElementById('placeholderRaw');
const placeholderEnhanced = document.getElementById('placeholderEnhanced');
const inputImage = document.getElementById('inputImage');
const inputVideo = document.getElementById('inputVideo');

let stream = null;
let running = false;
let useFileVideo = false;
let uploadedImage = null;
let demoFog = false;
let lastFrameTime = 0;
let fps = 0;
let frameCount = 0;
let lastFpsUpdate = performance.now();
let dehazeMs = 0;
let detectMs = 0;
let rafId = null;
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let voiceEnabled = true;
let lastVoiceTime = 0;

const tracker = new Tracker(0.32, 15);
let currentTracks = [];

let opts = {
  dehaze: true,
  omega: 0.88,
  t0: 0.18,
  clahe: true,
  claheStrength: 1.20,
  sharpen: true,
  detect: true,
  conf: 0.5,
  tracking: true,
  distance: true,
  alerts: true,
  roadSeg: true,
  model: 'road_yolo'
};

let fogMetrics = { fog: 0, level: 'HIGH', visibility: 1000 };

// Colors per class
const CLASS_COLORS = {
  'person': '#22c55e', 'rider': '#10b981', 'car': '#0ea5e9', 'truck': '#f59e0b', 'bus': '#ef4444',
  'motorcycle': '#a855f7', 'motor': '#a855f7', 'bicycle': '#06b6d4', 'bike': '#06b6d4',
  'traffic light': '#eab308', 'stop sign': '#dc2626', 'traffic sign': '#f43f5e', 'train': '#8b5cf6',
  'dog': '#84cc16', 'cat': '#f97316'
};
function colorFor(cls){ return CLASS_COLORS[cls] || '#38bdf8'; }

// UI elements
const el = {
  headerFog: document.getElementById('headerFog'),
  headerVis: document.getElementById('headerVis'),
  headerFps: document.getElementById('headerFps'),
  headerFpsTop: document.getElementById('headerFpsTop'),
  headerAiFog: document.getElementById('headerAiFog'),
  mFog: document.getElementById('mFog'),
  mVis: document.getElementById('mVis'),
  mFps: document.getElementById('mFps'),
  fogBarRaw: document.getElementById('fogBarRaw'),
  fogValueRaw: document.getElementById('fogValueRaw'),
  rawObjCount: document.getElementById('rawObjCount'),
  rawDetList: document.getElementById('rawDetList'),
  rawResolution: document.getElementById('rawResolution'),
  sourceLabel: document.getElementById('sourceLabel'),
  latencyRaw: document.getElementById('latencyRaw'),
  enhancedFPS: document.getElementById('enhancedFPS'),
  enhancedCount: document.getElementById('enhancedCount'),
  nearestHazard: document.getElementById('nearestHazard'),
  nearestDist: document.getElementById('nearestDist'),
  visibilityBadge: document.getElementById('visibilityBadge'),
  trackingBadge: document.getElementById('trackingBadge'),
  aiFogBadge: document.getElementById('aiFogBadge'),
  aiFogValue: document.getElementById('aiFogValue'),
  aiFogConf: document.getElementById('aiFogConf'),
  dehazeBar: document.getElementById('dehazeBar'),
  dehazeValue: document.getElementById('dehazeValue'),
  dehazeMs: document.getElementById('dehazeMs'),
  yoloLabel: document.getElementById('yoloLabel'),
  modelStatus: document.getElementById('modelStatus'),
  metricDark: document.getElementById('metricDark'),
  metricContrast: document.getElementById('metricContrast'),
  metricVis: document.getElementById('metricVis'),
  // AI classifier UI
  aiModelStatus: document.getElementById('aiModelStatus'),
  aiLabelBadge: document.getElementById('aiLabelBadge'),
  aiProbBadge: document.getElementById('aiProbBadge'),
  aiFogBar: document.getElementById('aiFogBar'),
  aiFogPct: document.getElementById('aiFogPct'),
  aiPredLabel: document.getElementById('aiPredLabel'),
  aiPredConf: document.getElementById('aiPredConf'),
  objectList: document.getElementById('objectList'),
  objBadge: document.getElementById('objBadge'),
  totalDet: document.getElementById('totalDet'),
  hazardCount: document.getElementById('hazardCount'),
  alertLog: document.getElementById('alertLog'),
  alertBanner: document.getElementById('alertBanner'),
  alertContent: document.getElementById('alertContent'),
  headerFog2: null,
  voiceLabel: document.getElementById('voiceLabel'),
  voiceStatus: document.getElementById('voiceStatus')
};

// UI helpers — range fill, toasts, micro-interactions
function setRangeFill(inp){
  const min = parseFloat(inp.min), max = parseFloat(inp.max), val = parseFloat(inp.value);
  const pct = ((val - min) / (max - min)) * 100;
  inp.style.background = `linear-gradient(90deg, ${inp.classList.contains('accent-amber-500') ? '#f59e0b, #78350f' : '#f97316, #7c2d12'} ${pct}%, rgba(255,255,255,0.08) ${pct}%)`;
}
function showToast(msg, level='info'){
  const stack = document.getElementById('toastStack');
  if(!stack) return;
  const icons = { danger:'fa-triangle-exclamation', warn:'fa-triangle-exclamation', info:'fa-circle-info', success:'fa-check' };
  const bg = level==='danger' ? 'bg-red-500' : level==='warn' ? 'bg-amber-500' : level==='success' ? 'bg-emerald-500' : 'bg-orange-500';
  const div = document.createElement('div');
  div.className = `pointer-events-auto flex items-start gap-3 px-4 py-3 rounded-2xl text-white text-sm font-bold shadow-[0_16px_40px_rgba(0,0,0,0.45)] border border-white/10 backdrop-blur-xl fade-in ${bg}`;
  div.style.animation='fadeIn 0.35s cubic-bezier(0.16,1,0.3,1)';
  div.innerHTML = `<i class="fa-solid ${icons[level]||icons.info} mt-0.5"></i><span class="flex-1 leading-tight">${msg}</span><button class="ml-2 w-6 h-6 rounded-full bg-white/15 flex items-center justify-center hover:bg-white/25"><i class="fa-solid fa-xmark text-xs"></i></button>`;
  div.querySelector('button').onclick=()=> div.remove();
  stack.appendChild(div);
  setTimeout(()=>{ div.style.opacity='0'; div.style.transform='translateY(6px)'; div.style.transition='all 0.35s'; setTimeout(()=>div.remove(),350); }, 3800);
}

function logAlert(msg, level='info'){
  const time = new Date().toLocaleTimeString();
  const colors = { danger: 'bg-red-500/10 border-red-500/20 text-red-300', warn: 'bg-amber-500/10 border-amber-500/20 text-amber-300', info: 'bg-orange-500/10 border-orange-500/20 text-orange-300' };
  const c = level==='danger'? colors.danger : level==='warn'? colors.warn : colors.info;
  const icon = level==='danger'? 'fa-triangle-exclamation' : level==='warn'? 'fa-triangle-exclamation' : 'fa-circle-info';
  // clear placeholder on first real log
  const placeholder = el.alertLog.querySelector('.text-slate-300') || el.alertLog.querySelector('.text-slate-500');
  if(placeholder && el.alertLog.children.length===1 && placeholder.closest('#alertLog')){
    // check if it's the empty state card
    if(el.alertLog.innerHTML.includes('System ready')) el.alertLog.innerHTML='';
  }
  const div = document.createElement('div');
  div.className = `px-3 py-2.5 rounded-xl border ${c} flex gap-2.5 items-start fade-in`;
  div.innerHTML = `<i class="fa-solid ${icon} mt-0.5"></i><div class="flex-1"><div class="font-bold leading-tight">${msg}</div><div class="text-[10px] opacity-70 font-medium">${time} • ${fogMetrics.level} visibility • ${fogMetrics.fog}% fog</div></div>`;
  el.alertLog.prepend(div);
  while(el.alertLog.children.length>12) el.alertLog.removeChild(el.alertLog.lastChild);
  // toast for important levels
  if(level!=='info' || msg.toLowerCase().includes('camera') || msg.toLowerCase().includes('snapshot') || msg.toLowerCase().includes('recording')){
    showToast(msg, level==='danger'?'danger': level==='warn'?'warn' : 'info');
  }
}

function showBanner(msg, level){
  let icon='fa-circle-info';
  if(level==='danger') icon='fa-triangle-exclamation';
  else if(level==='warn') icon='fa-triangle-exclamation';
  el.alertBanner.className = 'block fade-in';
  el.alertContent.className = `rounded-2xl px-4 py-3.5 flex items-center gap-3 font-bold text-sm ${level==='danger'?'alert-danger':level==='warn'?'alert-warn':'alert-info'}`;
  el.alertContent.innerHTML = `<i class="fa-solid ${icon} text-lg"></i><span class="flex-1">${msg}</span><button onclick="this.parentElement.parentElement.parentElement.classList.add('hidden')" class="ml-auto w-7 h-7 rounded-full bg-black/20 hover:bg-black/30 flex items-center justify-center transition"><i class="fa-solid fa-xmark text-xs"></i></button>`;
  clearTimeout(showBanner._t);
  showBanner._t=setTimeout(()=>{ el.alertBanner.classList.add('hidden'); }, 5000);
}

function speak(text){
  if(!voiceEnabled) return;
  if(!('speechSynthesis' in window)) return;
  const now = performance.now();
  if(now - lastVoiceTime < 3500) return; // throttle 3.5s
  lastVoiceTime = now;
  speechSynthesis.cancel();
  const ut = new SpeechSynthesisUtterance(text);
  ut.rate = 1.05; ut.pitch = 1; ut.volume = 0.9;
  speechSynthesis.speak(ut);
  el.voiceStatus.textContent = 'Speaking...';
  setTimeout(()=> el.voiceStatus.textContent = 'Ready', 1500);
}

// Load model
async function initModel(){
  const select = document.getElementById('modelSelect');
  // First attempt: Trained BDD100K Road Detector ONNX
  try {
    el.modelStatus.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span> Loading BDD100K Road Model...`;
    await loadYolo('models/road_detector.onnx');
    opts.model = 'road_yolo';
    if(select) select.value = 'road_yolo';
    el.modelStatus.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400"></span> BDD100K YOLO Ready`;
    el.yoloLabel.textContent = 'BDD100K Road Detector (YOLOv8 ONNX • 10 Road Classes)';
    logAlert('AI model loaded: BDD100K Road Detector (YOLOv8 ONNX)', 'info');
    return;
  } catch(e) {
    console.warn('[Model] BDD100K road detector not ready or failed, falling back to COCO-SSD', e);
  }

  // Fallback to COCO-SSD
  el.modelStatus.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span> Loading COCO-SSD...`;
  try{
    await loadCoco();
    opts.model = 'coco';
    if(select) select.value = 'coco';
    el.modelStatus.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400"></span> COCO-SSD Ready`;
    el.yoloLabel.textContent = 'COCO-SSD • 80 classes • lite_mobilenet_v2';
    logAlert('AI model loaded: COCO-SSD (YOLO-compatible) ready', 'info');
  }catch(e){
    el.modelStatus.innerHTML = `<span class="w-2 h-2 rounded-full bg-red-500"></span> Model failed`;
    console.error(e);
  }
}
initModel();

// Fog classifier (MobileNetV2 trained on your dataset)
let fogAiEnabled = true;
let lastFogAiResult = null;
let fogAiThrottle = 0;
async function initFogClassifier(){
  const status = el.aiModelStatus;
  if(status) status.textContent = 'loading ONNX…';
  if(status) status.className = 'px-2 py-0.5 rounded-full bg-amber-500/15 border border-amber-400/20 text-amber-200 text-[10px] font-bold';
  try{
    await loadFogClassifier('models/fog_classifier.onnx');
    if(status){ status.textContent = 'AI READY ✓'; status.className='px-2 py-0.5 rounded-full bg-emerald-500/15 border border-emerald-400/20 text-emerald-300 text-[10px] font-black'; }
    logAlert('Fog AI classifier loaded (MobileNetV2 • FoggyRoad 100% val) — hybrid DCP+AI active', 'info');
    showToast('Fog AI ready — FoggyRoad 100% val, 100% test','success');
  }catch(e){
    if(status){ status.textContent = 'AI off (ONNX failed)'; status.className='px-2 py-0.5 rounded-full bg-red-500/15 border border-red-400/20 text-red-300 text-[10px] font-bold'; }
    console.warn('Fog classifier load failed', e);
    fogAiEnabled = false;
  }
}
initFogClassifier();

// Road segmentation (TinyUNet trained on BDD100K — orange road border overlay)
let segEnabled = true;
let lastRoadMask = null;
let segThrottle = 0;
async function initRoadSeg(){
  try{
    await loadRoadSeg('models/road_seg.onnx');
    logAlert('Road segmenter loaded (TinyUNet • BDD100K) — orange road border active', 'info');
    showToast('Road highlight ready — orange border','success');
  }catch(e){
    console.warn('Road seg load failed', e);
    segEnabled = false;
    const t = document.getElementById('toggleSeg');
    if(t) t.checked = false;
  }
}
initRoadSeg();

function updateAiFogUI(res){
  if(!res) return;
  const pctFog = (res.pFog*100).toFixed(1);
  const pctNon = (res.pNonFog*100).toFixed(1);
  const isFog = res.label === 'fog';
  const conf = (res.confidence*100).toFixed(1);
  const col = isFog ? '#f59e0b' : '#22c55e';
  const bg = isFog ? 'rgba(245,158,11,0.18)' : 'rgba(34,197,94,0.14)';
  // Header
  if(el.headerAiFog){
    el.headerAiFog.textContent = `${res.label.toUpperCase()} ${conf}%`;
    el.headerAiFog.style.color = col;
  }
  // HUD badge
  if(el.aiFogValue){
    el.aiFogValue.textContent = res.label.toUpperCase();
    el.aiFogValue.style.color = col;
  }
  if(el.aiFogConf){
    el.aiFogConf.textContent = `${conf}%`;
  }
  if(el.aiFogBadge){
    el.aiFogBadge.style.borderColor = isFog ? 'rgba(245,158,11,0.3)' : 'rgba(34,197,94,0.3)';
    el.aiFogBadge.style.background = isFog ? 'rgba(245,158,11,0.10)' : 'rgba(34,197,94,0.08)';
  }
  // Pipeline card
  if(el.aiFogBar){
    el.aiFogBar.style.width = `${pctFog}%`;
    el.aiFogBar.style.background = isFog ? 'linear-gradient(90deg,#f59e0b,#ef4444)' : 'linear-gradient(90deg,#22c55e,#10b981)';
  }
  if(el.aiFogPct) el.aiFogPct.textContent = `${pctFog}% fog`;
  if(el.aiPredLabel){
    el.aiPredLabel.textContent = res.label.toUpperCase();
    el.aiPredLabel.style.color = col;
  }
  if(el.aiPredConf) el.aiPredConf.textContent = `${conf}% conf`;
  if(el.aiLabelBadge){
    el.aiLabelBadge.textContent = res.label.toUpperCase();
    el.aiLabelBadge.style.background = bg;
    el.aiLabelBadge.style.color = col;
    el.aiLabelBadge.style.borderColor = col+'40';
    el.aiLabelBadge.className = 'px-3 py-1.5 rounded-full text-xs font-black tracking-widest border';
  }
  if(el.aiProbBadge){
    el.aiProbBadge.textContent = `${pctFog}%`;
    el.aiProbBadge.style.background = col;
  }
}

// Try YOLO on demand
async function ensureYolo(targetModel = 'road_yolo'){
  const isRoad = targetModel === 'road_yolo';
  const url = isRoad ? 'models/road_detector.onnx' : 'https://cdn.jsdelivr.net/gh/ultralytics/assets@main/yolov8n.onnx';
  el.modelStatus.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span> Loading ${isRoad ? 'BDD100K Road' : 'YOLOv8n'} ONNX...`;
  try{
    await loadYolo(url);
    el.modelStatus.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400"></span> ${isRoad ? 'BDD100K Road' : 'YOLOv8n'} Ready`;
    el.yoloLabel.textContent = isRoad ? 'BDD100K Road Detector (YOLOv8 ONNX • 10 Road Classes)' : 'YOLOv8n (ONNX Runtime • 80 classes)';
    logAlert(`${isRoad ? 'BDD100K Road Detector' : 'YOLOv8n'} ONNX loaded successfully`, 'info');
    return true;
  }catch(e){
    el.modelStatus.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400"></span> YOLO failed, using COCO-SSD`;
    console.warn(e);
    // revert select
    document.getElementById('modelSelect').value='coco';
    opts.model='coco';
    logAlert('YOLO load failed, fell back to COCO-SSD', 'warn');
    return false;
  }
}

// Controls wiring — with range fill & haptics
['strength','clahe','conf'].forEach(id=>{
  const r=document.getElementById(id);
  if(r){ setRangeFill(r); r.addEventListener('input', ()=> setRangeFill(r)); }
});
document.getElementById('strength').addEventListener('input', e=>{
  opts.omega = parseFloat(e.target.value);
  document.getElementById('strengthVal').textContent = opts.omega.toFixed(2);
  el.dehazeBar.style.width = (opts.omega*100)+'%';
  el.dehazeValue.textContent = opts.omega.toFixed(2);
});
document.getElementById('clahe').addEventListener('input', e=>{
  opts.claheStrength = parseFloat(e.target.value);
  document.getElementById('claheVal').textContent = opts.claheStrength.toFixed(2);
});
document.getElementById('toggleCLAHE').addEventListener('change', e=> opts.clahe = e.target.checked);
document.getElementById('toggleSharpen').addEventListener('change', e=> opts.sharpen = e.target.checked);
document.getElementById('toggleDehaze').addEventListener('change', e=> opts.dehaze = e.target.checked);
document.getElementById('toggleDetect').addEventListener('change', e=> opts.detect = e.target.checked);
const btnUltraClear = document.getElementById('btnUltraClear');
if(btnUltraClear){
  btnUltraClear.addEventListener('click', ()=>{
    opts.omega = 0.88;
    opts.t0 = 0.18;
    opts.claheStrength = 1.20;
    opts.clahe = true;
    opts.sharpen = true;
    opts.dehaze = true;
    const s = document.getElementById('strength');
    if(s){ s.value = 0.88; setRangeFill(s); }
    const c = document.getElementById('clahe');
    if(c){ c.value = 1.20; setRangeFill(c); }
    const sv = document.getElementById('strengthVal');
    if(sv) sv.textContent = '0.88';
    const cv = document.getElementById('claheVal');
    if(cv) cv.textContent = '1.20';
    if(el.dehazeBar) el.dehazeBar.style.width = '88%';
    if(el.dehazeValue) el.dehazeValue.textContent = '0.88';
    const tC = document.getElementById('toggleCLAHE');
    if(tC) tC.checked = true;
    const tS = document.getElementById('toggleSharpen');
    if(tS) tS.checked = true;
    const tD = document.getElementById('toggleDehaze');
    if(tD) tD.checked = true;
    logAlert('Winter Clear profile: dense-fog auto-boost, neutral WB, Gamma adaptive, 88% ω, t₀ 0.18', 'info');
    showToast('⚡ Winter Clear profile applied ✓', 'success');
  });
}
document.getElementById('toggleTrack').addEventListener('change', e=> opts.tracking = e.target.checked);
document.getElementById('toggleDistance').addEventListener('change', e=> opts.distance = e.target.checked);
document.getElementById('toggleAlerts').addEventListener('change', e=> opts.alerts = e.target.checked);
document.getElementById('toggleSeg').addEventListener('change', e=>{ opts.roadSeg = e.target.checked; if(!e.target.checked) lastRoadMask = null; });
document.getElementById('conf').addEventListener('input', e=>{
  opts.conf = parseFloat(e.target.value);
  document.getElementById('confVal').textContent = opts.conf.toFixed(2);
});
document.getElementById('modelSelect').addEventListener('change', async e=>{
  opts.model = e.target.value;
  if(opts.model==='road_yolo' || opts.model==='yolo'){
    await ensureYolo(opts.model);
  } else {
    await loadCoco();
    el.modelStatus.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400"></span> COCO-SSD Ready`;
    el.yoloLabel.textContent = 'COCO-SSD • 80 classes • lite_mobilenet_v2';
    logAlert('Switched detection model: COCO-SSD', 'info');
  }
});

document.getElementById('btnVoice').addEventListener('click', ()=>{
  voiceEnabled = !voiceEnabled;
  const btn = document.getElementById('btnVoice');
  if(voiceEnabled){ btn.innerHTML=`<i class="fa-solid fa-volume-high"></i> VOICE ON`; btn.className="flex-1 py-2.5 rounded-xl bg-gradient-to-r from-emerald-500 to-teal-500 hover:from-emerald-400 hover:to-teal-400 text-white font-black text-xs flex items-center justify-center gap-2 shadow-[0_8px_20px_rgba(16,185,129,0.35)] border border-white/10 btn-shine"; el.voiceLabel.textContent='Enabled'; el.voiceStatus.innerHTML='<span class="w-2 h-2 rounded-full bg-emerald-400 animate-pulse"></span> Ready'; speak('Voice alerts enabled'); showToast('Voice alerts enabled','success'); }
  else { btn.innerHTML=`<i class="fa-solid fa-volume-xmark"></i> VOICE OFF`; btn.className="flex-1 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 font-black text-xs flex items-center justify-center gap-2 border border-white/10"; el.voiceLabel.textContent='Muted'; el.voiceStatus.innerHTML='<span class="w-2 h-2 rounded-full bg-slate-500"></span> Muted'; speechSynthesis.cancel(); showToast('Voice muted','warn'); }
});
document.getElementById('btnMute').addEventListener('click', ()=>{
  voiceEnabled=false; const b=document.getElementById('btnVoice'); b.innerHTML=`<i class="fa-solid fa-volume-xmark"></i> VOICE OFF`; b.className="flex-1 py-2.5 rounded-xl bg-slate-800 hover:bg-slate-700 text-slate-300 font-black text-xs flex items-center justify-center gap-2 border border-white/10"; el.voiceLabel.textContent='Muted'; el.voiceStatus.innerHTML='<span class="w-2 h-2 rounded-full bg-slate-500"></span> Muted'; speechSynthesis.cancel(); showToast('Voice muted','warn');
});
document.getElementById('btnClearAlerts').addEventListener('click', ()=>{
  el.alertLog.innerHTML='<div class="text-center py-7 rounded-2xl glass border border-dashed border-white/[0.08]"><div class="w-10 h-10 mx-auto rounded-xl bg-white/[0.06] border border-white/[0.08] flex items-center justify-center mb-2"><i class="fa-regular fa-bell text-slate-400"></i></div><div class="text-slate-300 font-bold">Alerts cleared</div><div class="text-[11px] text-slate-500">Awaiting new events</div></div>';
  el.alertBanner.classList.add('hidden');
  showToast('Alerts cleared','info');
});

document.getElementById('btnHelp').addEventListener('click', ()=> document.getElementById('infoSection').scrollIntoView({behavior:'smooth'}));
document.getElementById('btnFullscreen').addEventListener('click', ()=>{
  if(!document.fullscreenElement) document.documentElement.requestFullscreen();
  else document.exitFullscreen();
});

// Camera controls
btnStart.addEventListener('click', startCamera);
btnStop.addEventListener('click', stopCamera);
btnDemoFog.addEventListener('click', ()=>{
  demoFog = !demoFog;
  if(demoFog){
    btnDemoFog.className="px-4 py-3 rounded-xl bg-amber-500 hover:bg-amber-400 border border-amber-400 text-white font-extrabold text-sm flex items-center gap-2 shadow-[0_8px_20px_rgba(245,158,11,0.35)]";
    logAlert('Demo fog overlay enabled — testing dehazing', 'info'); speak('Demo fog enabled'); showToast('Demo fog ON — overlaying synthetic fog','warn');
  } else {
    btnDemoFog.className="px-4 py-3 rounded-xl bg-amber-500/10 hover:bg-amber-500/15 border border-amber-400/20 text-amber-200 font-extrabold text-sm flex items-center gap-2";
    logAlert('Demo fog disabled', 'info'); showToast('Demo fog off','info');
  }
});

async function startCamera(){
  try{
    stream = await navigator.mediaDevices.getUserMedia({ video: { width:{ ideal:1280 }, height:{ ideal:720 }, facingMode:'environment' }, audio:false });
    video.srcObject = stream;
    await video.play();
    useFileVideo = false;
    uploadedImage = null;
    running = true;
    placeholderRaw.style.display='none';
    placeholderEnhanced.style.display='none';
    btnStart.classList.add('hidden');
    btnStop.classList.remove('hidden');
    el.sourceLabel.textContent = 'Webcam (Live)';
    logAlert('Camera started — real-time fog vision active', 'info');
    speak('Camera started, fog vision active');
    // set canvas size based on video
    const vw = video.videoWidth || 640, vh = video.videoHeight || 480;
    setupCanvases(vw, vh);
    if(!rafId) loop();
  }catch(e){
    alert('Camera access failed: '+e.message + '\nTry uploading an image instead.');
    console.error(e);
    logAlert('Camera access denied: '+e.message, 'warn');
  }
}
function stopCamera(){
  running=false;
  if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; }
  if(rafId){ cancelAnimationFrame(rafId); rafId=null; }
  video.srcObject=null;
  fileVideo.pause(); fileVideo.src='';
  useFileVideo=false;
  btnStart.classList.remove('hidden');
  btnStop.classList.add('hidden');
  placeholderRaw.style.display='flex';
  // keep enhanced placeholder? show last frame
  el.sourceLabel.textContent='Idle';
  logAlert('Camera stopped', 'info');
}

function setupCanvases(w,h){
  const maxW=640;
  let nw=w, nh=h;
  if(w>maxW){ const r=maxW/w; nw=maxW; nh=Math.round(h*r); }
  [canvasRaw, canvasEnhanced, overlayRaw, overlayEnhanced].forEach(c=>{
    c.width=nw; c.height=nh;
    c.style.width='100%'; c.style.height='100%';
  });
  exportCanvas.width=nw; exportCanvas.height=nh;
  el.rawResolution.textContent = `${nw}×${nh}`;
}

// Image upload
inputImage.addEventListener('change', e=>{
  const file=e.target.files[0]; if(!file) return;
  const img=new Image();
  img.onload=()=>{
    uploadedImage=img;
    useFileVideo=false;
    running=true;
    if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; video.srcObject=null; }
    placeholderRaw.style.display='none';
    placeholderEnhanced.style.display='none';
    btnStart.classList.remove('hidden'); btnStop.classList.add('hidden');
    setupCanvases(img.width, img.height);
    el.sourceLabel.textContent='Image: '+file.name;
    logAlert(`Image loaded: ${file.name} (${img.width}×${img.height})`, 'info');
    // Process single frame then continue loop for detection
    if(!rafId) loop();
    // After 2s, pause loop if not video
    // Keep running to allow interactive dehazing tweaks
  };
  img.src=URL.createObjectURL(file);
});

// Video upload
inputVideo.addEventListener('change', e=>{
  const file=e.target.files[0]; if(!file) return;
  const url=URL.createObjectURL(file);
  fileVideo.src=url;
  fileVideo.muted=true;
  fileVideo.loop=true;
  fileVideo.play().then(()=>{
    useFileVideo=true;
    uploadedImage=null;
    running=true;
    if(stream){ stream.getTracks().forEach(t=>t.stop()); stream=null; video.srcObject=null; }
    placeholderRaw.style.display='none'; placeholderEnhanced.style.display='none';
    setupCanvases(fileVideo.videoWidth||640, fileVideo.videoHeight||480);
    el.sourceLabel.textContent='Video: '+file.name;
    logAlert(`Video loaded: ${file.name}`, 'info');
    if(!rafId) loop();
  });
});

// Snapshot — gives the CLEAR image, not the foggy one.
// • Photo upload → full-resolution HD dehaze (clean, highest quality)
// • Camera / video → composite of what you see (clear enhanced + YOLO overlay)
document.getElementById('btnSnapshot').addEventListener('click', async ()=>{
  try{
    // HD path for still images: re-run dehazer at natural resolution
    if(uploadedImage && uploadedImage.naturalWidth){
      const nw = Math.min(uploadedImage.naturalWidth, 1920);
      const scale = nw / uploadedImage.naturalWidth;
      const nh = Math.round(uploadedImage.naturalHeight * scale);
      showToast('Processing HD clear image…','info');
      const off = document.createElement('canvas');
      off.width = nw; off.height = nh;
      const octx = off.getContext('2d', { willReadFrequently: true });
      octx.drawImage(uploadedImage, 0, 0, nw, nh);
      let imgData = octx.getImageData(0, 0, nw, nh);
      if(demoFog){ const c = new ImageData(new Uint8ClampedArray(imgData.data), nw, nh); addFogEffect(c, 0.38); imgData = c; }
      const clear = opts.dehaze
        ? dehazeImageData(imgData, { omega: opts.omega, t0: opts.t0, clahe: opts.clahe, claheStrength: opts.claheStrength, sharpen: opts.sharpen })
        : imgData;
      octx.putImageData(clear, 0, 0);
      const a=document.createElement('a');
      a.download=`fogvision-clear-HD-${Date.now()}.png`;
      a.href=off.toDataURL('image/png');
      a.click();
      logAlert(`HD clear image saved (${nw}×${nh})`, 'info');
      showToast('HD clear image saved ✓','success');
    } else {
      updateExportFrame();
      const a=document.createElement('a');
      a.download=`fogvision-clear-${Date.now()}.png`;
      a.href=exportCanvas.toDataURL('image/png');
      a.click();
      logAlert('Snapshot saved (clear enhanced view)', 'info');
      showToast('Clear snapshot saved ✓','success');
    }
  }catch(e){
    console.warn('snapshot failed, fallback to display canvas', e);
    updateExportFrame();
    const a=document.createElement('a');
    a.download=`fogvision-clear-${Date.now()}.png`;
    a.href=exportCanvas.toDataURL('image/png');
    a.click();
  }
  const b=document.getElementById('btnSnapshot'); b.style.transform='scale(0.96)'; setTimeout(()=> b.style.transform='',140);
});

// Recording — gives CLEAR video (enhanced + overlay composite), not raw foggy feed
document.getElementById('btnRecord').addEventListener('click', ()=>{
  const btn=document.getElementById('btnRecord');
  if(!isRecording){
    updateExportFrame();
    const streamRec = exportCanvas.captureStream(30);
    recordedChunks=[];
    try{
      mediaRecorder = new MediaRecorder(streamRec, { mimeType:'video/webm;codecs=vp9' });
    }catch{ mediaRecorder = new MediaRecorder(streamRec); }
    mediaRecorder.ondataavailable=e=>{ if(e.data.size>0) recordedChunks.push(e.data); };
    mediaRecorder.onstop=()=>{
      const blob=new Blob(recordedChunks,{type:'video/webm'});
      const url=URL.createObjectURL(blob);
      const a=document.createElement('a'); a.href=url; a.download=`fogvision-clear-${Date.now()}.webm`; a.click();
      setTimeout(()=>URL.revokeObjectURL(url), 5000);
      logAlert('Clear video saved (dehazed)', 'info'); showToast('Clear video saved ✓','success');
    };
    mediaRecorder.start();
    isRecording=true;
    btn.innerHTML=`<span class="w-2 h-2 rounded-full bg-white animate-pulse"></span> STOP REC`;
    btn.className="px-4 py-3 rounded-xl bg-red-500 hover:bg-red-400 text-white font-black text-sm flex items-center gap-2 border border-red-400 shadow-[0_8px_20px_rgba(239,68,68,0.35)]";
    logAlert('Recording started (clear enhanced stream)', 'info'); showToast('Recording clear video • REC','warn');
  } else {
    try{ mediaRecorder.stop(); }catch{}
    isRecording=false;
    const btn2=document.getElementById('btnRecord');
    btn2.innerHTML=`<span class="w-7 h-7 rounded-lg bg-red-500/15 border border-red-400/20 flex items-center justify-center"><i class="fa-solid fa-circle-dot text-red-400 text-xs"></i></span> REC`;
    btn2.className="px-4 py-3 rounded-xl glass hover:bg-white/[0.08] font-extrabold text-sm flex items-center gap-2 card-lift";
  }
});

// Main loop
let lastDetections = [];
let detectionThrottle = 0;

async function loop(){
  rafId = requestAnimationFrame(loop);
  if(!running) return;
  const now = performance.now();
  // Determine source
  let sourceCanvas = null;
  let w, h;
  if(useFileVideo){
    if(fileVideo.readyState < 2) return;
    w=canvasRaw.width; h=canvasRaw.height;
    ctxRaw.drawImage(fileVideo, 0,0,w,h);
  } else if(uploadedImage){
    w=canvasRaw.width; h=canvasRaw.height;
    ctxRaw.drawImage(uploadedImage, 0,0,w,h);
  } else if(video.readyState >= 2){
    w=canvasRaw.width; h=canvasRaw.height;
    ctxRaw.drawImage(video, 0,0,w,h);
  } else {
    return;
  }

  // Optionally add demo fog before processing
  let rawImageData = ctxRaw.getImageData(0,0,w,h);
  let displayRawData = rawImageData;
  if(demoFog){
    // clone then add fog
    const fogged = new ImageData(new Uint8ClampedArray(rawImageData.data), w, h);
    addFogEffect(fogged, 0.38);
    displayRawData = fogged;
    // draw fogged version to raw canvas for visual
    ctxRaw.putImageData(fogged,0,0);
    rawImageData = fogged;
  }

  // Fog metrics (DCP)
  fogMetrics = estimateFogMetrics(rawImageData);
  updateFogUI(fogMetrics);

  // AI Fog classifier (hybrid — throttled ~4 FPS)
  if(fogAiEnabled && isFogReady()){
    fogAiThrottle++;
    if(fogAiThrottle % 8 === 0){
      // run async without blocking loop
      predictFog(canvasRaw).then(res=>{
        lastFogAiResult = res;
        updateAiFogUI(res);
        // Optionally fuse DCP+AI for banner: if AI says fog >0.85 and DCP fog <40, still alert
        if(res.pFog > 0.88 && fogMetrics.fog < 35){
          // gentle reconcile — boost displayed fog slightly
          // but keep DCP as primary for visibility; AI is secondary
        }
      }).catch(e=>{ /* silent */ });
    } else if(lastFogAiResult){
      // keep UI fresh between inferences?
    }
  }

  // Dehaze
  let t0 = performance.now();
  let enhancedData;
  if(opts.dehaze){
    enhancedData = dehazeImageData(rawImageData, { omega: opts.omega, t0: opts.t0, clahe: opts.clahe, claheStrength: opts.claheStrength, sharpen: opts.sharpen });
  } else {
    enhancedData = rawImageData;
  }
  dehazeMs = performance.now() - t0;
  ctxEnh.putImageData(enhancedData,0,0);

  // Detection (throttled to ~6 FPS to keep perf)
  detectionThrottle++;
  let detections = lastDetections;
  if(opts.detect && detectionThrottle % 4 === 0){
    const t1 = performance.now();
    try{
      let preds=[];
      if((opts.model==='road_yolo' || opts.model==='yolo') && typeof ort !== 'undefined'){
        try{ preds = await detectYolo(canvasEnhanced, opts.conf); }
        catch(e){ preds = await detectCoco(canvasEnhanced, opts.conf); }
      } else {
        preds = await detectCoco(canvasEnhanced, opts.conf);
      }
      detections = preds;
      lastDetections = preds;
      detectMs = performance.now() - t1;
    }catch(e){
      console.warn('detect error',e);
    }
  }

  // Road segmentation (throttled — TinyUNet 256px on dehazed frame so border aligns with boxes)
  if(opts.roadSeg && segEnabled && isSegReady()){
    segThrottle++;
    if(segThrottle % 6 === 0){
      predictRoadMask(canvasEnhanced).then(res=>{
        lastRoadMask = (res.roadFrac > 0.005) ? res : null;
      }).catch(e=>{ /* silent */ });
    }
  } else if(!opts.roadSeg && lastRoadMask){
    lastRoadMask = null;
  }

  // Tracking
  if(opts.tracking){
    currentTracks = tracker.update(detections, h);
  } else {
    // no tracking, just wrap detections as tracks
    currentTracks = detections.map((d,i)=> ({ id:i+1, bbox:d.bbox, class:d.class, score:d.score, dist: tracker.estimateDistance(d.class, d.bbox[3], h), age:0 }));
    currentTracks.sort((a,b)=>a.dist-b.dist);
  }

  // Render overlays
  drawOverlays(currentTracks, w, h);
  // Keep export canvas in sync so snapshots/recordings are the CLEAR view
  updateExportFrame();

  // Alerts
  if(opts.alerts) processAlerts(currentTracks, fogMetrics);

  // Update stats
  updateObjectList(currentTracks);
  frameCount++;
  if(now - lastFpsUpdate > 900){
    fps = Math.round(frameCount * 1000 / (now - lastFpsUpdate));
    frameCount=0; lastFpsUpdate=now;
    el.headerFps.textContent=fps;
    if(el.headerFpsTop) el.headerFpsTop.textContent=fps+' FPS';
    el.mFps.textContent=fps+' FPS';
    el.enhancedFPS.textContent=fps+' FPS';
    el.latencyRaw.textContent=(dehazeMs+detectMs).toFixed(0)+' ms';
    el.dehazeMs.textContent=dehazeMs.toFixed(0)+' ms';
  }

  // For static image, we still loop to allow slider real-time updates (dehazing runs every frame)
  // Optionally reduce FPS for image to save CPU by throttling
}

function updateFogUI(m){
  el.fogBarRaw.style.width=m.fog+'%';
  el.fogValueRaw.textContent=m.fog+'%';
  el.headerFog.textContent=m.fog+'%';
  el.mFog.textContent=m.fog+'%';
  el.headerVis.textContent=m.level;
  el.mVis.textContent=m.level;
  el.metricDark.textContent=m.darkMean;
  el.metricContrast.textContent=m.contrastLoss+'%';
  el.metricVis.textContent=m.visibility+' m';
  const fogColor = m.fog>70? '#ef4444' : m.fog>45? '#f59e0b' : m.fog>20? '#eab308' : '#22c55e';
  el.fogValueRaw.style.color=fogColor;
  el.fogBarRaw.style.background = `linear-gradient(90deg, ${fogColor}, ${fogColor}cc)`;
  el.headerFog.style.color=fogColor;
  el.mFog.style.color=fogColor;
  const badge = el.visibilityBadge;
  if(m.level==='VERY LOW'){ badge.className='px-3.5 py-2 rounded-full bg-red-500 text-white text-xs font-black tracking-widest shadow-[0_10px_24px_rgba(239,68,68,0.35)] border border-white/10'; badge.textContent='VISIBILITY: VERY LOW'; }
  else if(m.level==='LOW'){ badge.className='px-3.5 py-2 rounded-full bg-amber-500 text-white text-xs font-black tracking-widest shadow-[0_10px_24px_rgba(245,158,11,0.35)] border border-white/10'; badge.textContent='VISIBILITY: LOW'; }
  else if(m.level==='MODERATE'){ badge.className='px-3.5 py-2 rounded-full bg-yellow-500 text-white text-xs font-black tracking-widest shadow-[0_10px_24px_rgba(234,179,8,0.35)] border border-white/10'; badge.textContent='VISIBILITY: MODERATE'; }
  else { badge.className='px-3.5 py-2 rounded-full bg-emerald-500 text-white text-xs font-black tracking-widest shadow-[0_10px_24px_rgba(16,185,129,0.35)] border border-white/10'; badge.textContent='VISIBILITY: HIGH'; }
}

// Road overlay — translucent orange fill + bright orange border from seg mask
const segFillCanvas = document.createElement('canvas');
const segEdgeCanvas = document.createElement('canvas');
function drawRoadOverlay(){
  if(!opts.roadSeg || !lastRoadMask) return;
  const { mask, size } = lastRoadMask;
  const W = overlayEnhanced.width, H = overlayEnhanced.height;
  if(!W || !H || !mask) return;
  if(segFillCanvas.width !== size){ segFillCanvas.width = size; segFillCanvas.height = size; segEdgeCanvas.width = size; segEdgeCanvas.height = size; }
  const fctx = segFillCanvas.getContext('2d');
  const ectx = segEdgeCanvas.getContext('2d');
  const fillImg = fctx.createImageData(size, size);
  const edgeImg = ectx.createImageData(size, size);
  const fd = fillImg.data, ed = edgeImg.data;
  for(let y=0; y<size; y++){
    for(let x=0; x<size; x++){
      const i = y*size+x;
      if(!mask[i]) continue;
      const o = i*4;
      // translucent orange fill
      fd[o]=249; fd[o+1]=115; fd[o+2]=22; fd[o+3]=52;
      // boundary: any 4-neighbour outside road (frame edge counts as boundary)
      const left = x>0 ? mask[i-1] : 0;
      const right = x<size-1 ? mask[i+1] : 0;
      const up = y>0 ? mask[i-size] : 0;
      const down = y<size-1 ? mask[i+size] : 0;
      if(!left || !right || !up || !down){
        ed[o]=251; ed[o+1]=146; ed[o+2]=60; ed[o+3]=255;
      }
    }
  }
  fctx.putImageData(fillImg, 0, 0);
  ectx.putImageData(edgeImg, 0, 0);
  oEnh.save();
  oEnh.imageSmoothingEnabled = true;
  oEnh.drawImage(segFillCanvas, 0, 0, W, H);
  try{ oEnh.filter = 'blur(3px)'; oEnh.globalAlpha = 0.8; }catch{}
  oEnh.drawImage(segEdgeCanvas, 0, 0, W, H);
  try{ oEnh.filter = 'none'; oEnh.globalAlpha = 1; }catch{}
  oEnh.drawImage(segEdgeCanvas, 0, 0, W, H);
  oEnh.restore();
}

function drawOverlays(tracks, W, H){
  // clear
  oRaw.clearRect(0,0,overlayRaw.width, overlayRaw.height);
  oEnh.clearRect(0,0,overlayEnhanced.width, overlayEnhanced.height);
  // road highlight under the boxes
  drawRoadOverlay();
  // scaling: overlay canvas same size as main canvas (640-ish), no scaling needed
  // But if we use devicePixelRatio, handle
  const scaleX = overlayEnhanced.width / W;
  const scaleY = overlayEnhanced.height / H;
  // Actually W = canvas width already, so 1:1

  // raw overlay = dimmed boxes (show original detection without enhancement hint)
  // enhanced overlay = full
  for(const t of tracks){
    if(t.age>0) continue; // only active
    const [x,y,w,h]=t.bbox;
    const color=colorFor(t.class);
    const haz=hazardLevel(t.dist);
    // Draw on enhanced
    oEnh.strokeStyle = haz.color;
    oEnh.lineWidth = t.dist<12? 3 : 2;
    oEnh.fillStyle = haz.bg;
    // box
    oEnh.beginPath();
    oEnh.roundRect(x,y,w,h,6);
    oEnh.fill();
    oEnh.stroke();
    // header label
    const label = `${t.class.toUpperCase()} #${t.id} • ${(t.score*100).toFixed(0)}% • ${t.dist.toFixed(1)}m`;
    oEnh.font = 'bold 11px ui-sans-serif, system-ui';
    const textW = oEnh.measureText(label).width + 14;
    const lblH = 18;
    const lx = Math.max(0, Math.min(x, overlayEnhanced.width - textW));
    const ly = y>18? y-18 : y+h;
    oEnh.fillStyle = color;
    oEnh.beginPath();
    oEnh.roundRect(lx, ly, textW, lblH, 6);
    oEnh.fill();
    oEnh.fillStyle='white';
    oEnh.fillText(label, lx+7, ly+12);
    // distance dot
    oEnh.fillStyle=haz.color;
    oEnh.beginPath();
    oEnh.arc(x+w/2, y+h/2, 3,0,Math.PI*2);
    oEnh.fill();
    // Raw overlay (lighter)
    oRaw.strokeStyle = color + 'AA';
    oRaw.lineWidth = 1.6;
    oRaw.strokeRect(x,y,w,h);
  }
  // counts
  const active = tracks.filter(t=>t.age===0);
  el.rawObjCount.textContent = `${active.length} OBJECTS`;
  el.enhancedCount.textContent = `${active.length} TRACKED`;
  el.trackingBadge.textContent = `TRACKING: ${active.length? active.length+' ACTIVE' : 'IDLE'}`;
  el.rawDetList.textContent = active.length? active.map(t=> `${t.class}×${active.filter(x=>x.class===t.class).length}`).filter((v,i,a)=>a.indexOf(v)===i).join(', ') : '—';
}

function updateObjectList(tracks){
  const active=tracks.filter(t=>t.age===0);
  el.objBadge.textContent=active.length;
  el.totalDet.textContent=active.length;
  const hazards=active.filter(t=>t.dist<30).length;
  el.hazardCount.textContent=hazards;
  if(active.length===0){
    el.objectList.innerHTML=`<div class="py-8 text-center empty-glow rounded-2xl border border-white/[0.04]"><div class="w-12 h-12 mx-auto rounded-2xl glass flex items-center justify-center mb-2"><i class="fa-solid fa-radar text-slate-500"></i></div><div class="text-sm font-bold text-slate-400">No detections yet</div><div class="text-xs text-slate-500">Point camera at road / upload traffic image</div></div>`;
    el.nearestHazard.textContent='—';
    el.nearestDist.textContent='— m';
    el.nearestDist.style.color='';
    return;
  }
  el.objectList.innerHTML='';
  for(const t of active.slice(0,8)){
    const haz=hazardLevel(t.dist);
    const div=document.createElement('div');
    div.className='obj-card flex items-center gap-3 bg-[#0b132e]/90 border border-white/[0.07] rounded-xl px-3 py-2.5 hover:border-white/[0.12] transition';
    div.style.setProperty('--accent', colorFor(t.class));
    div.innerHTML=`
      <div class="w-9 h-9 rounded-xl flex items-center justify-center text-white font-black text-xs shrink-0 shadow-[0_4px_12px_rgba(0,0,0,0.3)]" style="background:${colorFor(t.class)}">${t.id}</div>
      <div class="flex-1 min-w-0">
        <div class="font-black text-sm leading-none flex items-center gap-1.5 flex-wrap">${t.class} <span class="text-[10px] px-1.5 py-0.5 rounded-full bg-white/[0.06] border border-white/[0.08] text-slate-300">${(t.score*100).toFixed(0)}%</span> <span class="text-[10px] px-1.5 py-0.5 rounded-full font-black tracking-widest" style="background:${haz.bg};color:${haz.color};border:1px solid ${haz.color}40">${haz.level}</span></div>
        <div class="text-xs text-slate-400 font-medium">${t.bbox[2].toFixed(0)}×${t.bbox[3].toFixed(0)} px • x${t.bbox[0].toFixed(0)} y${t.bbox[1].toFixed(0)}</div>
      </div>
      <div class="text-right shrink-0">
        <div class="font-black font-mono text-sm" style="color:${haz.color}">${t.dist.toFixed(1)} m</div>
        <div class="text-[10px] tracking-widest font-black" style="color:${haz.color}; opacity:0.85">${haz.level}</div>
      </div>
    `;
    el.objectList.appendChild(div);
  }
  const nearest=active[0];
  el.nearestHazard.textContent=`${nearest.class.toUpperCase()} #${nearest.id} • ${hazardLevel(nearest.dist).level}`;
  el.nearestDist.textContent=`${nearest.dist.toFixed(1)} m • ${(nearest.score*100).toFixed(0)}% conf`;
  el.nearestDist.style.color=hazardLevel(nearest.dist).color;
}

let lastAlertForId = new Map();
function processAlerts(tracks, fog){
  const active=tracks.filter(t=>t.age===0);
  if(active.length===0) return;
  const nearest=active[0];
  const haz=hazardLevel(nearest.dist);
  // Visual banner for close hazards
  if(nearest.dist < 18 && haz.level!=='INFO'){
    const now=performance.now();
    const key=nearest.id+'-'+haz.level;
    const last=lastAlertForId.get(key)||0;
    if(now-last>6000){
      lastAlertForId.set(key, now);
      const msg = haz.level==='DANGER' ? `DANGER: ${nearest.class} ahead ${nearest.dist.toFixed(0)} meters — BRAKE!` : `WARNING: ${nearest.class} at ${nearest.dist.toFixed(0)}m — Slow down`;
      showBanner(msg, haz.level==='DANGER'?'danger':'warn');
      logAlert(msg, haz.level==='DANGER'?'danger':'warn');
      speak(msg);
    }
  }
  // Fog visibility alert
  if(fog.fog>75 && fog.level==='VERY LOW'){
    const now=performance.now();
    if(now - (lastAlertForId.get('fog')||0) > 12000){
      lastAlertForId.set('fog', now);
      const msg=`Very low visibility ${fog.visibility}m — Drive with extreme caution`;
      logAlert(msg,'warn');
      showBanner(msg,'warn');
      speak('Very low visibility, drive with extreme caution');
    }
  }
}

// Keyboard shortcuts + micro polish
window.addEventListener('keydown', e=>{
  if(e.code==='Space'){ e.preventDefault(); if(running) stopCamera(); else startCamera(); }
  if(e.code==='KeyF'){
    e.preventDefault();
    demoFog=!demoFog;
    const b=document.getElementById('btnDemoFog');
    if(demoFog){ b.className="px-4 py-3 rounded-xl bg-amber-500 hover:bg-amber-400 border border-amber-400 text-white font-extrabold text-sm flex items-center gap-2 shadow-[0_8px_20px_rgba(245,158,11,0.35)]"; logAlert('Demo fog overlay enabled — testing dehazing', 'info'); showToast('Demo fog ON','warn'); }
    else { b.className="px-4 py-3 rounded-xl bg-amber-500/10 hover:bg-amber-500/15 border border-amber-400/20 text-amber-200 font-extrabold text-sm flex items-center gap-2"; logAlert('Demo fog disabled','info'); showToast('Demo fog off','info'); }
  }
  if(e.code==='Escape' && running){ stopCamera(); showToast('Camera stopped','info'); }
});
// Fullscreen icon swap
document.addEventListener('fullscreenchange', ()=>{
  const icon=document.querySelector('#btnFullscreen i');
  if(icon) icon.className = document.fullscreenElement ? 'fa-solid fa-compress text-slate-300 text-sm' : 'fa-solid fa-expand text-slate-300 text-sm';
});

// Initial canvas setup
setupCanvases(640,480);
// Render a demo gradient so UI not empty
ctxRaw.fillStyle='#0f172a'; ctxRaw.fillRect(0,0,640,480);
ctxRaw.fillStyle='#1e293b'; ctxRaw.font='bold 18px sans-serif'; ctxRaw.fillText(' awaiting input — start camera or upload', 110, 240);
ctxEnh.fillStyle='#0b1220'; ctxEnh.fillRect(0,0,640,480);
ctxEnh.fillStyle='#38bdf8'; ctxEnh.font='bold 16px sans-serif'; ctxEnh.fillText(' Enhanced output will appear here', 160, 240);

// Expose for debug
window.SFV = { opts, tracker, startCamera, stopCamera };

