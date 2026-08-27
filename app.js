import { estimateFogMetrics, dehazeImageData, addFogEffect } from './dehaze.js';
import { Tracker, hazardLevel } from './tracker.js';
import { loadCoco, loadYolo, detectCoco, detectYolo } from './detector.js';

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
  omega: 0.95,
  t0: 0.1,
  clahe: true,
  claheStrength: 1.2,
  sharpen: true,
  detect: true,
  conf: 0.5,
  tracking: true,
  distance: true,
  alerts: true,
  model: 'coco'
};

let fogMetrics = { fog: 0, level: 'HIGH', visibility: 1000 };

// Colors per class
const CLASS_COLORS = {
  'person': '#22c55e', 'car': '#0ea5e9', 'truck': '#f59e0b', 'bus': '#ef4444',
  'motorcycle': '#a855f7', 'bicycle': '#06b6d4', 'traffic light': '#eab308',
  'stop sign': '#dc2626', 'dog': '#84cc16', 'cat': '#f97316'
};
function colorFor(cls){ return CLASS_COLORS[cls] || '#38bdf8'; }

// UI elements
const el = {
  headerFog: document.getElementById('headerFog'),
  headerVis: document.getElementById('headerVis'),
  headerFps: document.getElementById('headerFps'),
  headerFpsTop: document.getElementById('headerFpsTop'),
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
  dehazeBar: document.getElementById('dehazeBar'),
  dehazeValue: document.getElementById('dehazeValue'),
  dehazeMs: document.getElementById('dehazeMs'),
  yoloLabel: document.getElementById('yoloLabel'),
  modelStatus: document.getElementById('modelStatus'),
  metricDark: document.getElementById('metricDark'),
  metricContrast: document.getElementById('metricContrast'),
  metricVis: document.getElementById('metricVis'),
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

function logAlert(msg, level='info'){
  const time = new Date().toLocaleTimeString();
  const colors = { danger: 'bg-red-500/10 border-red-500/20 text-red-300', warn: 'bg-amber-500/10 border-amber-500/20 text-amber-300', info: 'bg-sky-500/10 border-sky-500/20 text-sky-300' };
  const c = level==='danger'? colors.danger : level==='warn'? colors.warn : colors.info;
  const icon = level==='danger'? 'fa-triangle-exclamation' : level==='warn'? 'fa-triangle-exclamation' : 'fa-circle-info';
  const div = document.createElement('div');
  div.className = `px-3 py-2 rounded-xl border ${c} flex gap-2 items-start`;
  div.innerHTML = `<i class="fa-solid ${icon} mt-0.5"></i><div class="flex-1"><div class="font-bold">${msg}</div><div class="text-[10px] opacity-70">${time} • ${fogMetrics.level} visibility • ${fogMetrics.fog}% fog</div></div>`;
  el.alertLog.prepend(div);
  // keep 12 max
  while(el.alertLog.children.length>12) el.alertLog.removeChild(el.alertLog.lastChild);
  if(el.alertLog.children.length===1 || el.alertLog.firstChild.textContent.includes('System ready')){
    // remove placeholder
    const ph = el.alertLog.querySelector('.text-slate-500');
    if(ph) ph.remove();
  }
}

function showBanner(msg, level){
  const cfg = hazardLevel(level==='danger'?5:level==='warn'?20:60);
  // Actually map
  let bg='bg-sky-500', icon='fa-circle-info';
  if(level==='danger'){ bg='bg-red-600'; icon='fa-triangle-exclamation'; }
  else if(level==='warn'){ bg='bg-amber-500'; icon='fa-triangle-exclamation'; }
  el.alertBanner.className = 'block';
  el.alertContent.className = `rounded-xl px-4 py-3 flex items-center gap-3 font-bold text-sm ${level==='danger'?'alert-danger':level==='warn'?'alert-warn':'alert-info'}`;
  el.alertContent.innerHTML = `<i class="fa-solid ${icon}"></i><span>${msg}</span><button onclick="this.parentElement.parentElement.parentElement.classList.add('hidden')" class="ml-auto w-7 h-7 rounded-full bg-black/20 flex items-center justify-center"><i class="fa-solid fa-xmark text-xs"></i></button>`;
  setTimeout(()=>{ el.alertBanner.classList.add('hidden'); }, 5000);
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
  el.modelStatus.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span> Loading COCO-SSD...`;
  try{
    await loadCoco();
    el.modelStatus.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400"></span> COCO-SSD Ready`;
    el.yoloLabel.textContent = 'COCO-SSD • 80 classes • lite_mobilenet_v2';
    logAlert('AI model loaded: COCO-SSD (YOLO-compatible) ready', 'info');
  }catch(e){
    el.modelStatus.innerHTML = `<span class="w-2 h-2 rounded-full bg-red-500"></span> Model failed`;
    console.error(e);
  }
}
initModel();

// Try YOLO on demand
async function ensureYolo(){
  if(el.modelStatus.textContent.includes('YOLO')) return true;
  el.modelStatus.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span> Loading YOLOv8n (ONNX)...`;
  try{
    // Use CDN-hosted YOLOv8n ONNX - may be large; try
    const url = 'https://cdn.jsdelivr.net/gh/ultralytics/assets@main/yolov8n.onnx';
    // Fallback smaller: try yolov5n
    await loadYolo(url);
    el.modelStatus.innerHTML = `<span class="w-2 h-2 rounded-full bg-emerald-400"></span> YOLOv8n Ready`;
    el.yoloLabel.textContent = 'YOLOv8n (ONNX Runtime • 80 classes)';
    logAlert('YOLOv8n ONNX loaded successfully', 'info');
    return true;
  }catch(e){
    el.modelStatus.innerHTML = `<span class="w-2 h-2 rounded-full bg-amber-400"></span> YOLO failed, using COCO-SSD`;
    console.warn(e);
    // revert select
    document.getElementById('modelSelect').value='coco';
    opts.model='coco';
    logAlert('YOLO load failed (CORS/large file), fell back to COCO-SSD', 'warn');
    return false;
  }
}

// Controls wiring
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
document.getElementById('toggleTrack').addEventListener('change', e=> opts.tracking = e.target.checked);
document.getElementById('toggleDistance').addEventListener('change', e=> opts.distance = e.target.checked);
document.getElementById('toggleAlerts').addEventListener('change', e=> opts.alerts = e.target.checked);
document.getElementById('conf').addEventListener('input', e=>{
  opts.conf = parseFloat(e.target.value);
  document.getElementById('confVal').textContent = opts.conf.toFixed(2);
});
document.getElementById('modelSelect').addEventListener('change', async e=>{
  opts.model = e.target.value;
  if(opts.model==='yolo') await ensureYolo();
});

document.getElementById('btnVoice').addEventListener('click', ()=>{
  voiceEnabled = !voiceEnabled;
  const btn = document.getElementById('btnVoice');
  if(voiceEnabled){ btn.innerHTML=`<i class="fa-solid fa-volume-high"></i> VOICE ON`; btn.className="flex-1 py-2 rounded-xl bg-emerald-500 hover:bg-emerald-400 text-white font-black text-xs flex items-center justify-center gap-2"; el.voiceLabel.textContent='Enabled'; speak('Voice alerts enabled'); }
  else { btn.innerHTML=`<i class="fa-solid fa-volume-xmark"></i> VOICE OFF`; btn.className="flex-1 py-2 rounded-xl bg-slate-700 hover:bg-slate-600 text-white font-black text-xs flex items-center justify-center gap-2"; el.voiceLabel.textContent='Muted'; speechSynthesis.cancel(); }
});
document.getElementById('btnMute').addEventListener('click', ()=>{
  voiceEnabled=false; document.getElementById('btnVoice').innerHTML=`<i class="fa-solid fa-volume-xmark"></i> VOICE OFF`; document.getElementById('btnVoice').className="flex-1 py-2 rounded-xl bg-slate-700 hover:bg-slate-600 text-white font-black text-xs flex items-center justify-center gap-2"; el.voiceLabel.textContent='Muted'; speechSynthesis.cancel();
});
document.getElementById('btnClearAlerts').addEventListener('click', ()=>{
  el.alertLog.innerHTML='<div class="text-slate-500 text-center py-4">Alerts cleared</div>';
  el.alertBanner.classList.add('hidden');
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
  btnDemoFog.classList.toggle('bg-amber-500', demoFog);
  btnDemoFog.classList.toggle('text-white', demoFog);
  if(demoFog){ logAlert('Demo fog overlay enabled — testing dehazing', 'info'); speak('Demo fog enabled'); }
  else logAlert('Demo fog disabled', 'info');
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

// Snapshot
document.getElementById('btnSnapshot').addEventListener('click', ()=>{
  const a=document.createElement('a');
  a.download=`fogvision-${Date.now()}.png`;
  a.href=canvasEnhanced.toDataURL('image/png');
  a.click();
  logAlert('Snapshot saved (enhanced view)', 'info');
});

// Recording (canvas capture)
document.getElementById('btnRecord').addEventListener('click', ()=>{
  const btn=document.getElementById('btnRecord');
  if(!isRecording){
    const streamRec = canvasEnhanced.captureStream(30);
    recordedChunks=[];
    mediaRecorder = new MediaRecorder(streamRec, { mimeType:'video/webm;codecs=vp9' });
    mediaRecorder.ondataavailable=e=>{ if(e.data.size>0) recordedChunks.push(e.data); };
    mediaRecorder.onstop=()=>{
      const blob=new Blob(recordedChunks,{type:'video/webm'});
      const url=URL.createObjectURL(blob);
      const a=document.createElement('a'); a.href=url; a.download=`fogvision-record-${Date.now()}.webm`; a.click();
      logAlert('Recording saved', 'info');
    };
    mediaRecorder.start();
    isRecording=true;
    btn.innerHTML=`<i class="fa-solid fa-stop text-white"></i> STOP REC`;
    btn.classList.add('bg-red-500','text-white','border-red-600');
    logAlert('Recording started (enhanced stream)', 'info');
  } else {
    mediaRecorder.stop();
    isRecording=false;
    const btn2=document.getElementById('btnRecord');
    btn2.innerHTML=`<i class="fa-solid fa-circle-dot text-red-500"></i> REC`;
    btn2.classList.remove('bg-red-500','text-white','border-red-600');
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

  // Fog metrics
  fogMetrics = estimateFogMetrics(rawImageData);
  updateFogUI(fogMetrics);

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
      if(opts.model==='yolo' && typeof ort !== 'undefined'){
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
  // color logic
  const fogColor = m.fog>70? '#ef4444' : m.fog>45? '#f59e0b' : m.fog>20? '#eab308' : '#22c55e';
  el.fogValueRaw.style.color=fogColor;
  el.fogBarRaw.style.background=fogColor;
  el.headerFog.style.color=fogColor;
  // visibility badge
  const badge = el.visibilityBadge;
  if(m.level==='VERY LOW'){ badge.className='px-3 py-1.5 rounded-full bg-red-500 text-white text-xs font-black shadow-lg'; badge.textContent='VISIBILITY: VERY LOW'; }
  else if(m.level==='LOW'){ badge.className='px-3 py-1.5 rounded-full bg-amber-500 text-white text-xs font-black shadow-lg'; badge.textContent='VISIBILITY: LOW'; }
  else if(m.level==='MODERATE'){ badge.className='px-3 py-1.5 rounded-full bg-yellow-500 text-white text-xs font-black shadow-lg'; badge.textContent='VISIBILITY: MODERATE'; }
  else { badge.className='px-3 py-1.5 rounded-full bg-emerald-500 text-white text-xs font-black shadow-lg'; badge.textContent='VISIBILITY: HIGH'; }
}

function drawOverlays(tracks, W, H){
  // clear
  oRaw.clearRect(0,0,overlayRaw.width, overlayRaw.height);
  oEnh.clearRect(0,0,overlayEnhanced.width, overlayEnhanced.height);
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
    el.objectList.innerHTML=`<div class="py-8 text-center text-slate-500 text-sm">No detections yet.<br><span class="text-xs">Point camera at road / upload traffic image</span></div>`;
    el.nearestHazard.textContent='—';
    el.nearestDist.textContent='— m';
    return;
  }
  el.objectList.innerHTML='';
  for(const t of active.slice(0,8)){
    const haz=hazardLevel(t.dist);
    const div=document.createElement('div');
    div.className='flex items-center gap-3 bg-slate-900 border border-slate-800 rounded-xl px-3 py-2.5';
    div.innerHTML=`
      <div class="w-9 h-9 rounded-xl flex items-center justify-center text-white font-black text-xs" style="background:${colorFor(t.class)}">${t.id}</div>
      <div class="flex-1 min-w-0">
        <div class="font-black text-sm leading-none flex items-center gap-2">${t.class} <span class="text-[10px] px-1.5 py-0.5 rounded-full bg-slate-800 border border-slate-700">${(t.score*100).toFixed(0)}%</span> <span class="text-[10px] px-1.5 py-0.5 rounded-full" style="background:${haz.bg};color:${haz.color};border:1px solid ${haz.color}40">${haz.level}</span></div>
        <div class="text-xs text-slate-400">${t.bbox[2].toFixed(0)}×${t.bbox[3].toFixed(0)} px • x${t.bbox[0].toFixed(0)} y${t.bbox[1].toFixed(0)}</div>
      </div>
      <div class="text-right">
        <div class="font-black" style="color:${haz.color}">${t.dist.toFixed(1)} m</div>
        <div class="text-[10px] tracking-widest font-bold text-slate-500">${haz.level}</div>
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

// Keyboard shortcuts
window.addEventListener('keydown', e=>{
  if(e.code==='Space'){ e.preventDefault(); if(running) stopCamera(); else startCamera(); }
  if(e.code==='KeyF'){ demoFog=!demoFog; }
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

