// detector.js - wraps COCO-SSD and optional YOLOv8 ONNX
let cocoModel = null;
let yoloSession = null;
let yoloInputSize = 640;

// COCO class names for mapping if we later use ONNX
const YOLO_CLASSES = ['person','bicycle','car','motorcycle','airplane','bus','train','truck','boat','traffic light','fire hydrant','stop sign','parking meter','bench','bird','cat','dog','horse','sheep','cow','elephant','bear','zebra','giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee','skis','snowboard','sports ball','kite','baseball bat','baseball glove','skateboard','surfboard','tennis racket','bottle','wine glass','cup','fork','knife','spoon','bowl','banana','apple','sandwich','orange','broccoli','carrot','hot dog','pizza','donut','cake','chair','couch','potted plant','bed','dining table','toilet','tv','laptop','mouse','remote','keyboard','cell phone','microwave','oven','toaster','sink','refrigerator','book','clock','vase','scissors','teddy bear','hair drier','toothbrush'];

export async function loadCoco() {
  if (cocoModel) return cocoModel;
  cocoModel = await cocoSsd.load({ base: 'lite_mobilenet_v2' });
  return cocoModel;
}

export async function loadYolo(modelUrl) {
  if (yoloSession) return yoloSession;
  try {
    yoloSession = await ort.InferenceSession.create(modelUrl, { executionProviders: ['wasm'] });
    return yoloSession;
  } catch (e) {
    console.warn('YOLO load failed', e);
    yoloSession = null;
    throw e;
  }
}

export async function detectCoco(canvas, confTh=0.5) {
  if (!cocoModel) await loadCoco();
  const preds = await cocoModel.detect(canvas);
  // filter by conf and map to unified format
  return preds.filter(p=> p.score >= confTh).map(p=> ({
    bbox: p.bbox, // [x,y,w,h]
    class: p.class,
    score: p.score
  }));
}

// YOLOv8 ONNX: expects 640x640 letterboxed input, output [1,84,8400] or [1,5+classes,8400]
function preprocessForYolo(canvas, size=640) {
  const off = document.createElement('canvas');
  off.width = size; off.height = size;
  const ctx = off.getContext('2d');
  ctx.fillStyle = '#000'; ctx.fillRect(0,0,size,size);
  const scale = Math.min(size/canvas.width, size/canvas.height);
  const nw = canvas.width*scale, nh=canvas.height*scale;
  const dx=(size-nw)/2, dy=(size-nh)/2;
  ctx.drawImage(canvas, 0,0,canvas.width,canvas.height, dx,dy,nw,nh);
  const imgData = ctx.getImageData(0,0,size,size);
  const data = imgData.data;
  const float = new Float32Array(3*size*size);
  for (let i=0, p=0; i<data.length; i+=4) {
    const r=data[i]/255, g=data[i+1]/255, b=data[i+2]/255;
    float[p] = r; float[p+size*size]=g; float[p+2*size*size]=b;
    p++; if (p >= size*size) break;
    // Actually need per pixel correctly; we interleave per channel - simpler loop per pixel:
  }
  // Correct fill: iterate pixels
  for (let y=0; y<size; y++){
    for (let x=0; x<size; x++){
      const idx=(y*size+x)*4;
      const r=data[idx]/255, g=data[idx+1]/255, b=data[idx+2]/255;
      const p=y*size+x;
      float[p]=r; float[p+size*size]=g; float[p+2*size*size]=b;
    }
  }
  return { tensor: new ort.Tensor('float32', float, [1,3,size,size]), scale, dx, dy, size };
}

function nms(boxes, iouThr=0.45){
  boxes.sort((a,b)=> b.score - a.score);
  const keep=[];
  const suppressed=new Array(boxes.length).fill(false);
  for(let i=0;i<boxes.length;i++){
    if(suppressed[i]) continue;
    keep.push(boxes[i]);
    for(let j=i+1;j<boxes.length;j++){
      if(suppressed[j]) continue;
      if(boxes[i].class !== boxes[j].class) continue;
      const a=boxes[i].bbox, b=boxes[j].bbox;
      const x1=Math.max(a[0],b[0]), y1=Math.max(a[1],b[1]);
      const x2=Math.min(a[0]+a[2], b[0]+b[2]), y2=Math.min(a[1]+a[3], b[1]+b[3]);
      const w=Math.max(0,x2-x1), h=Math.max(0,y2-y1);
      const inter=w*h, union=a[2]*a[3]+b[2]*b[3]-inter;
      const iou= union? inter/union:0;
      if(iou>iouThr) suppressed[j]=true;
    }
  }
  return keep;
}

export async function detectYolo(canvas, confTh=0.5){
  if(!yoloSession) throw new Error('YOLO not loaded');
  const {tensor, scale, dx, dy, size} = preprocessForYolo(canvas, yoloInputSize);
  const feeds={ images: tensor };
  // try common input names
  try{
    const out = await yoloSession.run(feeds);
    const key = Object.keys(out)[0];
    const outT = out[key];
    const data = outT.data;
    const dims = outT.dims; // e.g. [1,84,8400]
    let numBoxes, numAttrs;
    if(dims.length===3){
      // [1,84,8400] -> transpose
      numAttrs=dims[1]; numBoxes=dims[2];
    } else if(dims.length===2){
      numBoxes=dims[0]; numAttrs=dims[1];
    } else {
      throw new Error('Unknown YOLO output dims '+dims);
    }
    const boxes=[];
    // YOLOv8 outputs: [x_center, y_center, w, h, ...80 confidences] already sigmoid? check range
    for(let i=0;i<numBoxes;i++){
      let maxScore=0, maxCls=-1;
      // find best class
      for(let c=4;c<numAttrs;c++){
        let score;
        if(dims.length===3) score=data[c*numBoxes+i]; else score=data[i*numAttrs+c];
        if(score>maxScore){ maxScore=score; maxCls=c-4; }
      }
      if(maxScore < confTh) continue;
      let xc,yc,w,h;
      if(dims.length===3){ xc=data[0*numBoxes+i]; yc=data[1*numBoxes+i]; w=data[2*numBoxes+i]; h=data[3*numBoxes+i]; }
      else { xc=data[i*numAttrs+0]; yc=data[i*numAttrs+1]; w=data[i*numAttrs+2]; h=data[i*numAttrs+3]; }
      // map back from 640 letterboxed to original canvas
      const x0 = (xc - dx) / scale;
      const y0 = (yc - dy) / scale;
      const bw = w / scale;
      const bh = h / scale;
      const bx = x0 - bw/2;
      const by = y0 - bh/2;
      // clamp
      if(bw<8 || bh<8) continue;
      boxes.push({ bbox:[bx,by,bw,bh], class: YOLO_CLASSES[maxCls]||('cls'+maxCls), score: maxScore });
    }
    return nms(boxes, 0.45);
  } catch(e){
    // fallback: try with 'input' name
    console.warn('YOLO run error', e);
    throw e;
  }
}

export function getYoloStatus(){ return !!yoloSession; }
