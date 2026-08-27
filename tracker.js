// tracker.js - simple IOU tracker with IDs, distance estimation
export class Tracker {
  constructor(iouThreshold=0.3, maxAge=12) {
    this.tracks = []; // {id, bbox:[x,y,w,h], class, score, age, hits, dist, lastSeen}
    this.nextId = 1;
    this.iouTh = iouThreshold;
    this.maxAge = maxAge;
  }

  iou(a,b){
    const x1=Math.max(a[0],b[0]), y1=Math.max(a[1],b[1]);
    const x2=Math.min(a[0]+a[2], b[0]+b[2]), y2=Math.min(a[1]+a[3], b[1]+b[3]);
    const w=Math.max(0,x2-x1), h=Math.max(0,y2-y1);
    const inter=w*h;
    const union=a[2]*a[3]+b[2]*b[3]-inter;
    return union? inter/union : 0;
  }

  // focal length heuristic for distance estimation (pinhole model)
  // distance = (REAL_HEIGHT_m * focal_px) / bbox_height_px
  estimateDistance(cls, bboxH, frameH) {
    const REAL_HEIGHT = {
      'person':1.7, 'bicycle':1.5, 'car':1.5, 'motorcycle':1.5, 'bus':3.0, 'truck':3.0,
      'traffic light':3.0, 'stop sign':2.0, 'bench':0.8, 'chair':0.9, 'dog':0.6, 'cat':0.4
    };
    const hReal = REAL_HEIGHT[cls] ?? 1.6;
    const focal = 720; // px, tuned for 640x480
    // compensate for class scale: if detection is not full height, fudge
    const h = Math.max(12, bboxH);
    let dist = (hReal * focal) / h;
    // widen for perspective: far objects appear small; add calibration
    // clamp 1m to 150m
    dist = Math.max(1, Math.min(150, dist));
    // Smoothing: nearer objects more reliable
    return dist;
  }

  update(detections, frameH) {
    // detections: [{bbox:[x,y,w,h], class, score}]
    const matched = new Set();
    const detMatched = new Set();
    // Try to match each track to best detection via IOU + class
    for (const tr of this.tracks) {
      let bestIoU=-1, bestIdx=-1;
      for (let i=0;i<detections.length;i++){
        if (detMatched.has(i)) continue;
        if (detections[i].class !== tr.class) continue;
        const ov = this.iou(tr.bbox, detections[i].bbox);
        if (ov>bestIoU){ bestIoU=ov; bestIdx=i; }
      }
      if (bestIoU >= this.iouTh) {
        const d = detections[bestIdx];
        tr.bbox = d.bbox;
        tr.score = d.score;
        tr.age = 0;
        tr.hits++;
        tr.dist = this.estimateDistance(tr.class, d.bbox[3], frameH);
        tr.vx = (d.bbox[0] - (tr.prevX??d.bbox[0]));
        tr.prevX = d.bbox[0];
        matched.add(tr.id);
        detMatched.add(bestIdx);
      } else {
        tr.age++;
      }
    }
    // Create new tracks for unmatched detections
    for (let i=0;i<detections.length;i++){
      if (detMatched.has(i)) continue;
      const d = detections[i];
      this.tracks.push({
        id: this.nextId++,
        bbox: d.bbox,
        class: d.class,
        score: d.score,
        age:0, hits:1,
        dist: this.estimateDistance(d.class, d.bbox[3], frameH),
        prevX: d.bbox[0]
      });
    }
    // Remove stale
    this.tracks = this.tracks.filter(t=> t.age < this.maxAge);
    // Sort by distance ascending
    this.tracks.sort((a,b)=> a.dist - b.dist);
    return this.tracks;
  }
  getActive() {
    return this.tracks.filter(t=> t.age===0);
  }
}

export function hazardLevel(dist) {
  if (dist < 12) return { level:'DANGER', color:'#ef4444', bg:'rgba(239,68,68,0.25)', text:'white' };
  if (dist < 25) return { level:'WARNING', color:'#f59e0b', bg:'rgba(245,158,11,0.25)', text:'white' };
  if (dist < 45) return { level:'CAUTION', color:'#eab308', bg:'rgba(234,179,8,0.18)', text:'white' };
  return { level:'INFO', color:'#22c55e', bg:'rgba(34,197,94,0.14)', text:'white' };
}
