// Canvas charts for the sidebar. Every chart marks zeros the same way the
// graph does: red diagonal hatch + red marker, never colour alone.
"use strict";

export const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

/** Compact variant for table cells: at most ~6 characters. */
export function fmtShort(v) {
  if (v == null || Number.isNaN(v)) return "–";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e3 || a < 1e-2) return v.toExponential(1).replace("e-", "e−");
  return v.toPrecision(2);
}

export function fmt(v) {
  if (v == null || Number.isNaN(v)) return "–";
  const a = Math.abs(v);
  if (a === 0) return "0";
  if (a >= 1e4 || a < 1e-3) return v.toExponential(2);
  if (a >= 100) return v.toFixed(1);
  return v.toPrecision(3);
}

function setupCanvas(canvas, w, h) {
  const dpr = window.devicePixelRatio || 1;
  canvas.width = Math.round(w * dpr);
  canvas.height = Math.round(h * dpr);
  canvas.style.width = w + "px";
  canvas.style.height = h + "px";
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  return ctx;
}

let _hatchCache = null;
export function hatch(ctx) {
  const key = css("--zero") + css("--zero-soft");
  if (!_hatchCache || _hatchCache.key !== key) {
    const c = document.createElement("canvas");
    c.width = c.height = 6;
    const x = c.getContext("2d");
    x.fillStyle = css("--zero-soft");
    x.fillRect(0, 0, 6, 6);
    x.strokeStyle = css("--zero");
    x.lineWidth = 1.4;
    x.beginPath();
    x.moveTo(-1, 7); x.lineTo(7, -1);
    x.moveTo(-1, 1); x.lineTo(1, -1);
    x.moveTo(5, 7); x.lineTo(7, 5);
    x.stroke();
    _hatchCache = { key, canvas: c };
  }
  return ctx.createPattern(_hatchCache.canvas, "repeat");
}

/**
 * Per-filter bar chart.
 *  values   number[]       score per filter (index = filter id)
 *  opts.order     number[] display order of filter ids
 *  opts.zero      bool[]   zero filters (hatched column + red tick under the axis)
 *  opts.keep      Set      filters inside the top-k selection (others greyed)
 *  opts.selected  filter id
 *  opts.signed    draw around a centre baseline
 * Returns hitTest(clientX) -> filter id.
 */
export function barChart(canvas, values, opts = {}) {
  const n = values.length;
  const order = opts.order || values.map((_, i) => i);
  const minBar = 3;
  const avail = opts.width || 460;
  const w = Math.max(avail, n * minBar);
  const h = opts.height || 130;
  const ctx = setupCanvas(canvas, w, h);
  const padB = 10; // zero-marker strip
  const plotH = h - padB - 2;
  const bw = w / Math.max(1, n);
  const maxAbs = Math.max(1e-30, ...values.map((v) => Math.abs(v)));
  const signed = opts.signed && values.some((v) => v < 0);
  const base = signed ? plotH / 2 + 1 : plotH + 1;
  const imp = css("--imp"), neg = css("--neg"), zero = css("--zero"), faint = css("--border");

  ctx.fillStyle = css("--hist-bg");
  ctx.fillRect(0, 0, w, plotH + 2);
  ctx.strokeStyle = css("--hist-border");
  ctx.lineWidth = 1;
  ctx.strokeRect(0.5, 0.5, w - 1, plotH + 1);
  order.forEach((f, i) => {
    const x = i * bw;
    const isZero = opts.zero && opts.zero[f];
    if (isZero) {
      ctx.fillStyle = hatch(ctx);
      ctx.fillRect(x, 0, Math.max(1, bw), plotH + 2);
      ctx.fillStyle = zero;
      ctx.fillRect(x, h - padB + 2, Math.max(1.5, bw - (bw > 3 ? 1 : 0)), padB - 3);
    }
    const v = values[f];
    const len = (Math.abs(v) / maxAbs) * (signed ? plotH / 2 - 1 : plotH - 1);
    const kept = !opts.keep || opts.keep.has(f);
    ctx.fillStyle = !kept ? faint : v < 0 ? neg : imp;
    const bwDraw = Math.max(1, bw - (bw > 4 ? 1 : 0));
    if (v >= 0) ctx.fillRect(x, base - len, bwDraw, len);
    else ctx.fillRect(x, signed ? base : base - len, bwDraw, len);
  });
  if (signed) {
    ctx.fillStyle = faint;
    ctx.fillRect(0, base, w, 1);
  }
  if (bw >= 16) {
    ctx.fillStyle = css("--muted");
    ctx.font = "10px sans-serif";
    ctx.textAlign = "center";
    order.forEach((f, i) => { if (!(opts.zero && opts.zero[f])) ctx.fillText(String(f), i * bw + bw / 2, h - 1); });
  }
  if (opts.selected != null) {
    const i = order.indexOf(opts.selected);
    if (i >= 0) {
      ctx.strokeStyle = css("--accent");
      ctx.lineWidth = 2;
      ctx.strokeRect(i * bw + 1, 1, Math.max(2, bw - 2), plotH);
    }
  }
  return (offsetX) => {
    const i = Math.floor(offsetX / bw);
    return i >= 0 && i < n ? order[i] : null;
  };
}

// Importance ramp: cyan (#00FFFF, least important) -> blue (#0000FF) ->
// magenta (#FF00FF, most important). Multi-hue, so "almost unimportant" and
// "a little important" differ in hue, not just lightness; magenta stays
// distinguishable from the red hatch used for zeros.
const RAMP = [[0, 255, 255], [0, 0, 255], [255, 0, 255]];

/** Colour-scale mapping for importance: "linear" or "log" (3 decades). */
let _scale = "linear";
export function setColorScale(s) { _scale = s; }
export function impT(v, max) {
  const t = max > 0 ? Math.min(1, Math.abs(v) / max) : 0;
  if (_scale !== "log") return t;
  return t <= 0 ? 0 : Math.max(0, 1 + Math.log10(t) / 3); // 1e-3 of max -> 0, max -> 1
}

/** Sequential (importance) colour for t in [0,1] -> "rgb(...)". */
export function seqColor(t) {
  const x = Math.max(0, Math.min(1, t)) * (RAMP.length - 1);
  const i = Math.min(RAMP.length - 2, Math.floor(x)), f = x - i;
  const c = RAMP[i].map((a, k) => Math.round(a + (RAMP[i + 1][k] - a) * f));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
/** Readable text colour on top of seqColor(t). */
export const seqText = (t) => (t < 0.22 ? "#1a1a1a" : "#fff"); // dark on cyan, white on blue/magenta
/** CSS gradient of the importance colour scale (legend). */
export const seqGradient = () => `linear-gradient(90deg, ${RAMP.map((c) => `rgb(${c})`).join(",")})`;
/** Diverging colour for t in [-1,1] (signed values: weights, signed importance). */
export function divColor(t) {
  const c = t < 0 ? css("--neg") : css("--imp");
  return `color-mix(in srgb, ${c} ${Math.round(100 * Math.min(1, Math.abs(t)))}%, ${css("--panel")})`;
}
// canvas fillStyle doesn't accept color-mix() in every browser -> let CSS resolve it to rgb()
const _colorCache = new Map();
export function resolveColor(c) {
  if (_colorCache.has(c)) return _colorCache.get(c);
  const el = document.createElement("span");
  el.style.color = c;
  document.body.appendChild(el);
  const rgb = getComputedStyle(el).color;
  el.remove();
  if (_colorCache.size > 4000) _colorCache.clear();
  _colorCache.set(c, rgb);
  return rgb;
}
export function clearColorCache() { _colorCache.clear(); }

/**
 * Grid of kernels (one cell per input channel of the selected filter).
 *  cells[i] = {value, zero, kernel?: number[][] (weights, for "weights" mode), zeroMask?: number[][]}
 *  opts.mode "importance" | "weights"
 * Returns hitTest(x, y) -> cell index.
 */
export function cellGrid(canvas, cells, opts = {}) {
  const n = cells.length;
  const avail = opts.width || 460;
  const cell = n > 512 ? 10 : n > 128 ? 16 : n > 32 ? 24 : 34;
  const gap = cell >= 24 ? 3 : 1;
  const cols = Math.max(1, Math.floor((avail + gap) / (cell + gap)));
  const rows = Math.ceil(n / cols);
  const w = Math.min(avail, cols * (cell + gap));
  const h = rows * (cell + gap);
  const ctx = setupCanvas(canvas, w, h);
  const maxV = Math.max(1e-30, ...cells.map((c) => Math.abs(c.value)));
  let maxW = 1e-30;
  if (opts.mode === "weights") for (const c of cells) for (const r of c.kernel || []) for (const v of r) maxW = Math.max(maxW, Math.abs(v));
  const zero = css("--zero");

  cells.forEach((c, i) => {
    const x = (i % cols) * (cell + gap), y = Math.floor(i / cols) * (cell + gap);
    if (opts.mode === "weights" && c.kernel) {
      const kr = c.kernel.length, kc = c.kernel[0].length;
      const pw = cell / kc, ph = cell / kr;
      for (let a = 0; a < kr; a++)
        for (let b = 0; b < kc; b++) {
          const z = c.zeroMask && c.zeroMask[a][b];
          ctx.fillStyle = z ? hatch(ctx) : resolveColor(divColor(c.kernel[a][b] / maxW));
          ctx.fillRect(x + b * pw, y + a * ph, Math.ceil(pw), Math.ceil(ph));
        }
    } else {
      ctx.fillStyle = c.zero ? hatch(ctx) : seqColor(impT(c.value, maxV));
      ctx.fillRect(x, y, cell, cell);
    }
    if (c.zero) {
      ctx.strokeStyle = zero;
      ctx.lineWidth = cell >= 16 ? 2 : 1;
      ctx.strokeRect(x + 1, y + 1, cell - 2, cell - 2);
      if (cell >= 16) {
        ctx.beginPath();
        ctx.moveTo(x + 4, y + 4); ctx.lineTo(x + cell - 4, y + cell - 4);
        ctx.moveTo(x + cell - 4, y + 4); ctx.lineTo(x + 4, y + cell - 4);
        ctx.stroke();
      }
    }
    if (opts.selected === i) {
      ctx.strokeStyle = css("--accent");
      ctx.lineWidth = 2.5;
      ctx.strokeRect(x - 0.5, y - 0.5, cell + 1, cell + 1);
    }
  });
  return (px, py) => {
    const ci = Math.floor(px / (cell + gap)), ri = Math.floor(py / (cell + gap));
    const i = ri * cols + ci;
    return ci < cols && i >= 0 && i < n ? i : null;
  };
}

/** Small 2-D/1-D input renderer for the Input node. data: [C,H,W] | [C,L] | [L]. */
export function renderInputThumb(canvas, data, shape, size = 72) {
  if (shape.length === 3) {
    const [C, H, W] = shape;
    canvas.width = W; canvas.height = H;
    const s = Math.max(1, Math.floor(size / Math.max(H, W)));
    canvas.style.width = W * s + "px"; canvas.style.height = H * s + "px";
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(W, H);
    let mn = Infinity, mx = -Infinity;
    for (const ch of data) for (const row of ch) for (const v of row) { if (v < mn) mn = v; if (v > mx) mx = v; }
    const rg = mx - mn || 1;
    for (let y = 0; y < H; y++)
      for (let x = 0; x < W; x++) {
        const k = (y * W + x) * 4;
        for (let c = 0; c < 3; c++) img.data[k + c] = ((data[C >= 3 ? c : 0][y][x] - mn) / rg) * 255;
        img.data[k + 3] = 255;
      }
    ctx.putImageData(img, 0, 0);
    return;
  }
  const series = shape.length === 2 ? data : [data];
  const ctx = setupCanvas(canvas, size * 2, size);
  const flat = series.flat();
  const mn = Math.min(...flat), mx = Math.max(...flat), rg = mx - mn || 1;
  const L = series[0].length;
  series.slice(0, 6).forEach((s, i) => {
    ctx.strokeStyle = i === 0 ? css("--imp") : css("--muted");
    ctx.beginPath();
    s.forEach((v, j) => {
      const x = (j / Math.max(1, L - 1)) * size * 2, y = size - ((v - mn) / rg) * (size - 4) - 2;
      j ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
    });
    ctx.stroke();
  });
}
