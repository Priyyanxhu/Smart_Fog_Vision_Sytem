(function(){
// dehaze.js - Clear Dark Channel Prior + Neutral White Balance + Exposure Restore + Sharpen
// Produces bright, neutral, high-visibility output (no dark-blue crush).

function estimateFogMetrics(imageData) {
  const { data, width, height } = imageData;
  const total = width * height;
  let darkSum = 0;
  let minDark = 255, maxDark = 0;
  let sum = 0, sumSq = 0, count = 0;

  for (let i = 0; i < data.length; i += 16) { // every 4th pixel for speed
    const r = data[i], g = data[i+1], b = data[i+2];
    const dark = Math.min(r, g, b);
    darkSum += dark;
    if (dark < minDark) minDark = dark;
    if (dark > maxDark) maxDark = dark;
    const lum = 0.299 * r + 0.587 * g + 0.114 * b;
    sum += lum;
    sumSq += lum * lum;
    count++;
  }
  const meanDark = darkSum / count;
  const mean = sum / count;
  const variance = Math.max(0, sumSq / count - mean * mean);
  const std = Math.sqrt(variance);

  const darkNorm = meanDark / 255;
  const contrastNorm = Math.min(1, std / 65);
  let fog = (darkNorm * 0.65 + (1 - contrastNorm) * 0.35) * 100;
  fog = Math.max(0, Math.min(100, fog));

  if (contrastNorm > 0.85 && darkNorm < 0.3) fog *= 0.5;
  let visibility = 1000 * Math.pow(1 - fog / 100, 1.4);
  if (fog < 10) visibility = 800 + (10 - fog) * 20;
  visibility = Math.round(visibility);

  let level = 'HIGH';
  if (fog > 70) level = 'VERY LOW';
  else if (fog > 45) level = 'LOW';
  else if (fog > 20) level = 'MODERATE';

  return { fog: Math.round(fog), contrastLoss: Math.round((1 - contrastNorm) * 100), darkMean: (meanDark / 255).toFixed(2), visibility, level, std: Math.round(std) };
}

// Separable fast box blur for transmission smoothing
function boxBlur(src, w, h, r) {
  const horiz = new Float32Array(w * h);
  const dst = new Float32Array(w * h);
  const inv = 1 / (2 * r + 1);

  for (let y = 0; y < h; y++) {
    const rowOffset = y * w;
    let sum = 0;
    for (let x = 0; x < w; x++) {
      sum += src[rowOffset + x];
      if (x >= 2 * r + 1) sum -= src[rowOffset + x - 2 * r - 1];
      if (x >= r) horiz[rowOffset + x - r] = sum * inv;
    }
    for (let x = 0; x < r && x < w; x++) horiz[rowOffset + x] = horiz[rowOffset + r];
    for (let x = Math.max(0, w - r); x < w; x++) horiz[rowOffset + x] = horiz[rowOffset + Math.max(0, w - r - 1)];
  }

  for (let x = 0; x < w; x++) {
    let sum = 0;
    for (let y = 0; y < h; y++) {
      sum += horiz[y * w + x];
      if (y >= 2 * r + 1) sum -= horiz[(y - 2 * r - 1) * w + x];
      if (y >= r) dst[(y - r) * w + x] = sum * inv;
    }
    for (let y = 0; y < r && y < h; y++) dst[y * w + x] = dst[r * w + x];
    for (let y = Math.max(0, h - r); y < h; y++) dst[y * w + x] = dst[Math.max(0, h - r - 1) * w + x];
  }
  return dst;
}

function smoothTransmission(trans, w, h) {
  let t = boxBlur(trans, w, h, 6);
  t = boxBlur(t, w, h, 4);
  return t;
}

function dehazeImageData(srcImageData, opts = {}) {
  // Winter-driving defaults: strong enough for dense fog (Image1) -> clear (Image2),
  // yet safe for clear days. Dense fog auto-boosts below.
  let omega = opts.omega ?? 0.88;
  let t0 = Math.max(0.12, opts.t0 ?? 0.18);
  const doCLAHE = opts.clahe ?? true;
  let claheStrength = opts.claheStrength ?? 1.20;
  const doSharpen = opts.sharpen ?? true;
  const doWhiteBalance = opts.whiteBalance ?? true;
  let gamma = opts.gamma ?? 0.80;

  const { width, height, data } = srcImageData;
  const wh = width * height;

  // --- 1. Dark Channel ---
  const dark = new Float32Array(wh);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    dark[p] = Math.min(data[i], data[i+1], data[i+2]);
  }

  // Global fog hint early (drives dense-winter mode for car safety)
  let meanDarkN = 0;
  {
    let ds = 0, n = 0;
    for (let p = 0; p < wh; p += 8) { ds += dark[p]; n++; }
    meanDarkN = (ds / Math.max(1, n)) / 255; // 0..1, ~0.7 for Image1 dense fog
  }
  const fogHint = Math.max(0, Math.min(1, meanDarkN * 1.2));
  const denseMode = fogHint > 0.55; // Image1-style winter fog
  if (denseMode) {
    // Push through the fog: deeper transmission, stronger local contrast,
    // gamma near 1.0 so recovered dark greens/browns stay dark like Image2
    omega = Math.max(omega, 0.92);
    t0 = Math.min(t0, 0.15);
    claheStrength = Math.max(claheStrength, 1.25);
    gamma = 0.90;
  }

  // --- 2. Atmospheric Light A (top 0.1% brightest in dark channel, averaged) ---
  const hist = new Int32Array(256);
  for (let p = 0; p < wh; p++) hist[dark[p] | 0]++;

  const topCount = Math.max(1, Math.floor(wh * 0.001));
  let countAccum = 0, threshold = 220;
  for (let val = 255; val >= 0; val--) {
    countAccum += hist[val];
    if (countAccum >= topCount) { threshold = val; break; }
  }

  let Ar = 0, Ag = 0, Ab = 0, cntA = 0;
  for (let p = 0; p < wh; p++) {
    if (dark[p] >= threshold) {
      const idx = p * 4;
      Ar += data[idx]; Ag += data[idx+1]; Ab += data[idx+2];
      cntA++;
    }
  }
  if (cntA > 0) { Ar /= cntA; Ag /= cntA; Ab /= cntA; }
  else { Ar = 200; Ag = 200; Ab = 200; }

  // Neutralize the color cast in A itself (keeps recovery hue-preserving).
  // Fully neutral A (R=G=B) means J=(I-A)/t+A shifts all channels equally,
  // so no blue cast is injected at the source. This is the main colored-output fix.
  const avgA0 = (Ar + Ag + Ab) / 3;
  const neutA = Math.min(220, Math.max(150, avgA0));
  Ar = neutA; Ag = neutA; Ab = neutA;

  // --- 3. Transmission Map (adaptive floor: deep for dense winter fog, gentle for clear sky) ---
  // Dense like Image1 needs t down to ~0.15 to pull grass 190->80 (Image2).
  // Clear days keep a higher floor so sky doesn't crush.
  const floorBase = denseMode ? 0.14 : 0.30;
  const floorScale = denseMode ? 0.08 : 0.24;
  // User strength must win: at omega 0.95 the floor collapses so fog is fully cut;
  // at low omega the floor stays high to protect sky. Without this, 0.95 still looked foggy.
  const floorGain = Math.max(0.22, 1.2 - omega);
  const trans = new Float32Array(wh);
  for (let p = 0; p < wh; p++) {
    const idx = p * 4;
    const normMin = Math.min(data[idx] / Ar, data[idx+1] / Ag, data[idx+2] / Ab);
    let t = 1.0 - omega * normMin;
    const skyFloor = (floorBase + floorScale * (dark[p] / 255)) * floorGain;
    if (t < skyFloor) t = skyFloor;
    if (t < t0) t = t0;
    if (t > 1.0) t = 1.0;
    trans[p] = t;
  }

  // --- 4. Smooth Transmission ---
  const smooth = smoothTransmission(trans, width, height);

  // --- 5. Scene Recovery J = (I - A)/t + A ---
  const rArr = new Float32Array(wh);
  const gArr = new Float32Array(wh);
  const bArr = new Float32Array(wh);
  let sumR = 0, sumG = 0, sumB = 0;

  for (let p = 0, i = 0; p < wh; p++, i += 4) {
    const skyFloor = (floorBase + floorScale * (dark[p] / 255)) * floorGain;
    const t = Math.max(t0, skyFloor, smooth[p]);
    let r = (data[i] - Ar) / t + Ar;
    let g = (data[i+1] - Ag) / t + Ag;
    let b = (data[i+2] - Ab) / t + Ab;
    r = Math.max(0, Math.min(255, r));
    g = Math.max(0, Math.min(255, g));
    b = Math.max(0, Math.min(255, b));
    rArr[p] = r; gArr[p] = g; bArr[p] = b;
    sumR += r; sumG += g; sumB += b;
  }

  // --- 6. Adaptive neutral White Balance (fixes dark-blue cast, preserves real color) ---
  let scaleR = 1.0, scaleG = 1.0, scaleB = 1.0;
  if (doWhiteBalance && sumR > 0 && sumG > 0 && sumB > 0) {
    const meanR = sumR / wh, meanG = sumG / wh, meanB = sumB / wh;
    const avgGray = (meanR + meanG + meanB) / 3.0;
    // fogHint already computed early; reuse (dense winter -> near-full gray world)
    const k = 0.70 + 0.28 * fogHint; // 0.70 .. 0.98
    scaleR = (1 - k) + k * (avgGray / Math.max(1, meanR));
    scaleG = (1 - k) + k * (avgGray / Math.max(1, meanG));
    scaleB = (1 - k) + k * (avgGray / Math.max(1, meanB));
    // Extra warm correction for residual blue dominance
    const blueGap = meanB - meanR;
    if (blueGap > 12) {
      scaleB *= 0.92;
      scaleR *= 1.04;
    } else if (blueGap > 6) {
      scaleB *= 0.95;
      scaleR *= 1.02;
    }
    scaleR = Math.min(1.55, Math.max(0.70, scaleR));
    scaleG = Math.min(1.40, Math.max(0.75, scaleG));
    scaleB = Math.min(1.55, Math.max(0.60, scaleB));
  }

  // Apply WB + auto-exposure.
  // Dense winter: target ~108 (Image2 natural daylight, dark greens preserved).
  // Light fog: target ~122 (bright, no crush).
  for (let p = 0; p < wh; p++) {
    rArr[p] = Math.max(0, Math.min(255, rArr[p] * scaleR));
    gArr[p] = Math.max(0, Math.min(255, gArr[p] * scaleG));
    bArr[p] = Math.max(0, Math.min(255, bArr[p] * scaleB));
  }
  // Exposure restore: target mean luminance ~128 so output never looks crushed
  let lumSum = 0;
  for (let p = 0; p < wh; p += 4) {
    lumSum += 0.299 * rArr[p] + 0.587 * gArr[p] + 0.114 * bArr[p];
  }
  const lumMean = lumSum / Math.ceil(wh / 4);
  const expTarget = denseMode ? 108 : 122;
  const expFloor = denseMode ? 95 : 118;
  let gain = 1.0;
  if (lumMean < expFloor && lumMean > 1) {
    gain = Math.min(denseMode ? 1.45 : 1.35, expTarget / lumMean);
    for (let p = 0; p < wh; p++) {
      rArr[p] = Math.max(0, Math.min(255, rArr[p] * gain));
      gArr[p] = Math.max(0, Math.min(255, gArr[p] * gain));
      bArr[p] = Math.max(0, Math.min(255, bArr[p] * gain));
    }
  }

  // --- 7. Gamma + vibrance (dense: restore greens like Image2) ---
  const gammaLUT = new Float32Array(256);
  for (let v = 0; v < 256; v++) gammaLUT[v] = 255.0 * Math.pow(v / 255.0, gamma);

  const lumArr = new Float32Array(wh);
  const vib = denseMode ? 1.15 : 1.10;
  for (let p = 0; p < wh; p++) {
    let r = gammaLUT[Math.max(0, Math.min(255, rArr[p] | 0))];
    let g = gammaLUT[Math.max(0, Math.min(255, gArr[p] | 0))];
    let b = gammaLUT[Math.max(0, Math.min(255, bArr[p] | 0))];
    const gray = 0.299 * r + 0.587 * g + 0.114 * b;
    r = gray + (r - gray) * vib;
    g = gray + (g - gray) * vib;
    b = gray + (b - gray) * vib;
    rArr[p] = r; gArr[p] = g; bArr[p] = b;
    lumArr[p] = Math.max(0, Math.min(255, gray));
  }

  // --- 8. Gentle Auto-Levels (1% / 99% — avoids clipping that crushed shadows) ---
  const lumHist = new Int32Array(256);
  for (let p = 0; p < wh; p++) lumHist[lumArr[p] | 0]++;
  const lowThreshCount = Math.floor(wh * 0.01);
  const highThreshCount = Math.floor(wh * 0.99);
  let acc = 0, pLow = 0, pHigh = 255;
  for (let v = 0; v < 256; v++) {
    acc += lumHist[v];
    if (acc >= lowThreshCount && pLow === 0) pLow = v;
    if (acc >= highThreshCount) { pHigh = v; break; }
  }
  if (pHigh <= pLow + 10) { pLow = 0; pHigh = 255; }
  const range = Math.max(1, pHigh - pLow);

  const out = new ImageData(width, height);
  const o = out.data;

  for (let p = 0, i = 0; p < wh; p++, i += 4) {
    let r = ((rArr[p] - pLow) / range) * 255.0;
    let g = ((gArr[p] - pLow) / range) * 255.0;
    let b = ((bArr[p] - pLow) / range) * 255.0;

    if (doCLAHE && claheStrength !== 1.0) {
      // Mild S-curve, blended 80% curve + 20% linear so shadows never crush
      const s = claheStrength;
      const denom = 2.0 * Math.tanh(s);
      const k2 = s * 1.8;
      const curve = (v) => {
        const n = Math.max(0, Math.min(1, v / 255));
        const c = 255 * (0.5 + Math.tanh((n - 0.5) * k2) / denom);
        return c * 0.8 + v * 0.2;
      };
      r = curve(r); g = curve(g); b = curve(b);
    }

    o[i]   = Math.max(0, Math.min(255, r));
    o[i+1] = Math.max(0, Math.min(255, g));
    o[i+2] = Math.max(0, Math.min(255, b));
    o[i+3] = 255;
  }

  // --- 8b. Final cast removal: force neutral output (colored-image guarantee) ---
  // If the frame is still blue-dominant (like the navy screenshot), shift R up / B down
  // globally so grass/road render natural instead of navy.
  {
    let sr = 0, sg = 0, sb = 0;
    for (let i = 0; i < o.length; i += 16) { sr += o[i]; sg += o[i+1]; sb += o[i+2]; }
    const n4 = o.length / 16;
    const mR = sr / n4, mG = sg / n4, mB = sb / n4;
    const gap = mB - mR;
    if (gap > 12) {
      const shift = Math.min(38, gap * 0.45);
      for (let i = 0; i < o.length; i += 4) {
        o[i]   = Math.max(0, Math.min(255, o[i] + shift));
        o[i+2] = Math.max(0, Math.min(255, o[i+2] - shift));
      }
    }
  }

  // --- 9. Mild sharpening (0.32 — less fog-noise amplification) ---
  if (doSharpen) return sharpenImageData(out, 0.32);
  return out;
}

function sharpenImageData(img, amount) {
  const w = img.width, h = img.height;
  const src = img.data;
  const out = new ImageData(w, h);
  const d = out.data;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let r = 0, g = 0, b = 0, cnt = 0;
      for (let dy = -1; dy <= 1; dy++) {
        for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx >= 0 && nx < w && ny >= 0 && ny < h) {
            const idx = (ny * w + nx) * 4;
            r += src[idx]; g += src[idx+1]; b += src[idx+2]; cnt++;
          }
        }
      }
      const idx = (y * w + x) * 4;
      const blR = r / cnt, blG = g / cnt, blB = b / cnt;
      d[idx]   = Math.max(0, Math.min(255, src[idx] + (src[idx] - blR) * amount));
      d[idx+1] = Math.max(0, Math.min(255, src[idx+1] + (src[idx+1] - blG) * amount));
      d[idx+2] = Math.max(0, Math.min(255, src[idx+2] + (src[idx+2] - blB) * amount));
      d[idx+3] = 255;
    }
  }
  return out;
}

// Add synthetic fog for demo (blend with white)
function addFogEffect(imageData, intensity = 0.45) {
  const d = imageData.data;
  for (let i = 0; i < d.length; i += 4) {
    d[i]   = d[i] * (1 - intensity) + 255 * intensity * 0.85;
    d[i+1] = d[i+1] * (1 - intensity) + 255 * intensity * 0.88;
    d[i+2] = d[i+2] * (1 - intensity) + 255 * intensity * 0.92;
  }
  for (let i = 0; i < d.length; i += 4) {
    d[i]   = 128 + (d[i] - 128) * (1 - intensity * 0.5);
    d[i+1] = 128 + (d[i+1] - 128) * (1 - intensity * 0.5);
    d[i+2] = 128 + (d[i+2] - 128) * (1 - intensity * 0.5);
  }
  return imageData;
}

window.SFVMods = window.SFVMods || {};
window.SFVMods.dehaze = { estimateFogMetrics, dehazeImageData, addFogEffect };
})();
