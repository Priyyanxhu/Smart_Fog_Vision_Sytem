// dehaze.js - Dark Channel Prior + Guided Filter approximation + CLAHE + Sharpen
// All functions operate on ImageData and return new ImageData

export function estimateFogMetrics(imageData) {
  const { data, width, height } = imageData;
  const total = width * height;
  let darkSum = 0;
  let contrastSum = 0;
  // sample stride for performance
  let minDark = 255, maxDark = 0;
  let sum = 0, sumSq = 0, count = 0;
  for (let i = 0; i < data.length; i += 16) { // every 4th pixel
    const r = data[i], g = data[i+1], b = data[i+2];
    const dark = Math.min(r,g,b);
    darkSum += dark;
    if (dark < minDark) minDark = dark;
    if (dark > maxDark) maxDark = dark;
    // luminance for contrast
    const lum = 0.299*r + 0.587*g + 0.114*b;
    sum += lum;
    sumSq += lum*lum;
    count++;
  }
  const meanDark = darkSum / count;
  const mean = sum / count;
  const variance = Math.max(0, sumSq/count - mean*mean);
  const std = Math.sqrt(variance);
  // Fog density: high dark channel + low contrast => high fog
  // Normalize dark 0-255 -> 0-100, std 0-80 typical
  const darkNorm = meanDark / 255; // 0-1
  const contrastNorm = Math.min(1, std / 65); // 1 = high contrast (clear)
  // Fog% = weighted: dark contributes 60%, low contrast 40%
  let fog = (darkNorm * 0.65 + (1 - contrastNorm) * 0.35) * 100;
  fog = Math.max(0, Math.min(100, fog));
  // Adjust: very clear images with high contrast should be low fog
  if (contrastNorm > 0.85 && darkNorm < 0.3) fog *= 0.5;
  // Visibility estimation (meteorological): empirical
  // fog 0% -> 1000m+, 50% -> ~200m, 80% -> ~50m
  let visibility = 1000 * Math.pow(1 - fog/100, 1.4);
  if (fog < 10) visibility = 800 + (10 - fog)*20;
  visibility = Math.round(visibility);
  let level = 'HIGH';
  if (fog > 70) level = 'VERY LOW';
  else if (fog > 45) level = 'LOW';
  else if (fog > 20) level = 'MODERATE';
  return { fog: Math.round(fog), contrastLoss: Math.round((1-contrastNorm)*100), darkMean: (meanDark/255).toFixed(2), visibility, level, std: Math.round(std) };
}

// Box blur for transmission smoothing (separable)
function boxBlur1D(src, dst, w, h, r) {
  // horizontal
  for (let y = 0; y < h; y++) {
    let sum = 0;
    for (let x = -r; x < w; x++) {
      if (x >= 0 && x - r -1 >= 0) sum -= src[y*w + (x - r -1)];
      if (x + r < w) sum += src[y*w + (x + r)];
      if (x >= 0) dst[y*w + x] = sum / (2*r+1);
    }
  }
}
function boxBlur(src, w, h, r) {
  const tmp = new Float32Array(w*h);
  const dst = new Float32Array(w*h);
  // horiz
  for (let y=0; y<h; y++){
    let sum=0;
    for (let x=0; x<w; x++){
      const idx = y*w+x;
      sum += src[idx];
      if (x >= r) sum -= src[idx - r - 0 - (x>=2*r+1?0:0)] // we'll do sliding properly
    }
  }
  // Instead use simple two-pass with prefix sums for correctness and speed
  // Replace with integral approach
  const horiz = new Float32Array(w*h);
  for (let y=0; y<h; y++){
    let sum=0;
    for (let x=0; x<w; x++){
      sum += src[y*w+x];
      if (x >= 2*r+1) sum -= src[y*w + x - 2*r -1];
      if (x >= r) horiz[y*w + x - r] = sum / (2*r+1);
    }
    // tail
    for (let x=w - r; x<w; x++){
      // already handled? keep as is
    }
    // left edge fill
    for (let x=0; x<r && x<w; x++) horiz[y*w+x] = horiz[y*w+r];
    for (let x=w-r; x<w; x++) horiz[y*w+x] = horiz[y*w+w-r-1];
  }
  // vertical
  for (let x=0; x<w; x++){
    let sum=0;
    for (let y=0; y<h; y++){
      sum += horiz[y*w+x];
      if (y >= 2*r+1) sum -= horiz[(y-2*r-1)*w+x];
      if (y >= r) dst[(y-r)*w+x] = sum/(2*r+1);
    }
    for (let y=0; y<r; y++) dst[y*w+x] = dst[r*w+x];
    for (let y=h-r; y<h; y++) dst[y*w+x] = dst[(h-r-1)*w+x];
  }
  return dst;
}

// Fast 3x3 or 5x5 guided filter approximation = just box blur for demo
function smoothTransmission(trans, w, h) {
  // two iterations of small box blur ~ guided filter approx
  let t = boxBlur(trans, w, h, 6);
  t = boxBlur(t, w, h, 4);
  return t;
}

export function dehazeImageData(srcImageData, opts = {}) {
  const omega = opts.omega ?? 0.95;
  const t0 = opts.t0 ?? 0.1;
  const doCLAHE = opts.clahe ?? true;
  const claheStrength = opts.claheStrength ?? 1.2;
  const doSharpen = opts.sharpen ?? true;

  const { width, height, data } = srcImageData;
  const wh = width * height;

  // --- 1. Dark channel (min over RGB) ---
  const dark = new Float32Array(wh);
  for (let i=0, p=0; i<data.length; i+=4, p++) {
    dark[p] = Math.min(data[i], data[i+1], data[i+2]);
  }

  // --- 2. Atmospheric light A: top 0.1% brightest dark pixels ---
  const numTop = Math.max(1, Math.floor(wh * 0.001));
  // Instead of full sort, use nth_element via sampling
  // Create copy and sort partial for speed: use a small array of top values
  // Quick: create array of indices sorted by dark descending via typed sort
  // For performance at 640x480 (307k), full sort is okay but we can use quickselect
  const darkCopy = Array.from(dark);
  darkCopy.sort((a,b)=>b-a);
  const threshold = darkCopy[numTop];
  let Ar=0, Ag=0, Ab=0, cntA=0;
  let maxSum = -1, bestR=220, bestG=220, bestB=220;
  for (let p=0; p<wh; p++) {
    if (dark[p] >= threshold) {
      const idx = p*4;
      const s = data[idx]+data[idx+1]+data[idx+2];
      if (s > maxSum) { maxSum=s; bestR=data[idx]; bestG=data[idx+1]; bestB=data[idx+2]; }
      Ar+=data[idx]; Ag+=data[idx+1]; Ab+=data[idx+2]; cntA++;
      if (cntA>2000) break; // enough
    }
  }
  if (cntA>0){ Ar/=cntA; Ag/=cntA; Ab/=cntA; } else { Ar=bestR; Ag=bestG; Ab=bestB; }
  // Clamp A to avoid pure white blowing out
  Ar = Math.min(225, Math.max(150, Ar));
  Ag = Math.min(225, Math.max(150, Ag));
  Ab = Math.min(225, Math.max(150, Ab));

  // --- 3. Transmission t = 1 - omega * dark/A ---
  const trans = new Float32Array(wh);
  const Aavg = (Ar+Ag+Ab)/3;
  for (let p=0; p<wh; p++) {
    // normalized dark: min(I/A)
    const idx = p*4;
    const r = data[idx]/Ar, g = data[idx+1]/Ag, b = data[idx+2]/Ab;
    const m = Math.min(r,g,b);
    let t = 1 - omega * m;
    if (t < t0) t = t0;
    if (t > 1) t = 1;
    trans[p]=t;
  }

  // --- 4. Smooth transmission ---
  const smooth = smoothTransmission(trans, width, height);

  // --- 5. Recover J = (I - A)/t + A + optional enhancements ---
  const out = new ImageData(width, height);
  const o = out.data;
  for (let p=0, i=0; p<wh; p++, i+=4) {
    const t = Math.max(t0, smooth[p]);
    let r = (data[i] - Ar)/t + Ar;
    let g = (data[i+1] - Ag)/t + Ag;
    let b = (data[i+2] - Ab)/t + Ab;
    // Apply CLAHE-like contrast stretch: simple adaptive gain
    if (doCLAHE) {
      // gamma-like boost midtones, slight
      const gain = claheStrength;
      // Normalize 0-255 then apply: out = 255 * ((in/255)^ (1/gain))
      // but cheaper: linear gain around mean 128
      r = 128 + (r - 128) * gain;
      g = 128 + (g - 128) * gain;
      b = 128 + (b - 128) * gain;
    }
    o[i]   = Math.max(0, Math.min(255, r));
    o[i+1] = Math.max(0, Math.min(255, g));
    o[i+2] = Math.max(0, Math.min(255, b));
    o[i+3] = 255;
  }

  // --- 6. Sharpen (unsharp mask) if enabled ---
  if (doSharpen) {
    return sharpenImageData(out, 0.45);
  }
  return out;
}

function sharpenImageData(img, amount) {
  const w = img.width, h = img.height;
  const src = img.data;
  // simple 3x3 blur then unsharp
  const blurred = new Uint8ClampedArray(src.length);
  // box blur 3x3
  for (let y=0; y<h; y++){
    for (let x=0; x<w; x++){
      let r=0,g=0,b=0,cnt=0;
      for (let dy=-1; dy<=1; dy++){
        for (let dx=-1; dx<=1; dx++){
          const nx=x+dx, ny=y+dy;
          if (nx>=0&&nx<w&&ny>=0&&ny<h){
            const idx=(ny*w+nx)*4;
            r+=src[idx]; g+=src[idx+1]; b+=src[idx+2]; cnt++;
          }
        }
      }
      const idx=(y*w+x)*4;
      blurred[idx]=r/cnt; blurred[idx+1]=g/cnt; blurred[idx+2]=b/cnt; blurred[idx+3]=255;
    }
  }
  const out = new ImageData(w,h);
  const d = out.data;
  for (let i=0;i<src.length;i+=4){
    d[i]   = Math.max(0,Math.min(255, src[i] + (src[i]-blurred[i])*amount));
    d[i+1] = Math.max(0,Math.min(255, src[i+1] + (src[i+1]-blurred[i+1])*amount));
    d[i+2] = Math.max(0,Math.min(255, src[i+2] + (src[i+2]-blurred[i+2])*amount));
    d[i+3] = 255;
  }
  return out;
}

// Add synthetic fog for demo (blend with white)
export function addFogEffect(imageData, intensity=0.45) {
  const d = imageData.data;
  for (let i=0;i<d.length;i+=4){
    d[i]   = d[i]*(1-intensity) + 255*intensity*0.85;
    d[i+1] = d[i+1]*(1-intensity) + 255*intensity*0.88;
    d[i+2] = d[i+2]*(1-intensity) + 255*intensity*0.92;
  }
  // reduce contrast
  for (let i=0;i<d.length;i+=4){
    d[i]   = 128 + (d[i]-128)*(1-intensity*0.5);
    d[i+1] = 128 + (d[i+1]-128)*(1-intensity*0.5);
    d[i+2] = 128 + (d[i+2]-128)*(1-intensity*0.5);
  }
  return imageData;
}
