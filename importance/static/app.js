// Importance viewer: Netron-style dataflow graph (left) + property / drill-down
// sidebar (right). Plain ES modules, no build step, no CDN -- everything comes
// from importance/serve.py's JSON endpoints:
//   /api/manifest  /api/graph  /api/sparsity  /api/overview?output=&metric=
//   /api/layer/<id>/unit/<filter>?output=&metric=   /api/samples/<name>?index=
"use strict";

import { layoutGraph } from "./layout.js";
import { barChart, cellGrid, css, fmt, fmtShort, seqColor, seqText, seqGradient, impT, setColorScale, divColor, resolveColor, clearColorCache, renderInputThumb } from "./charts.js";

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const store = {
  get(k, d) { try { const v = localStorage.getItem("imp-viewer:" + k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem("imp-viewer:" + k, JSON.stringify(v)); } catch { /* private mode etc. */ } },
};

const state = {
  manifest: null,
  graph: null,
  sparsity: {},          // layerId -> {info, mask_filter, ...}
  overview: null,        // {layers: {layerId: {metric, filter, layer}}}
  layerById: new Map(),  // manifest entries
  nodeById: new Map(),
  layout: null,
  output: 0,
  metric: "auto",
  histOrder: store.get("histOrder", "sorted"),
  histAxis: store.get("histAxis", "filter"), // node histograms: "filter" | "input"
  axis: store.get("axis", "filter"),         // sidebar chart: "filter" | "input"
  colorScale: store.get("colorScale", "linear"),
  inputSel: new Map(),   // layerId -> selected real input channel
  orient: store.get("orient", "TB"),
  edgeShapes: store.get("edgeShapes", true),
  zerosOnly: false,
  selected: null,        // node id
  unit: new Map(),       // layerId -> {filter, kernel}
  chartSort: store.get("chartSort", "index"),
  topk: 100,
  gridMode: store.get("gridMode", "importance"),
  sample: 0,
  collapsed: new Set(store.get("collapsed", ["props"])),
  view: { x: 0, y: 0, k: 1 },
};

const api = async (url) => {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url} -> HTTP ${r.status}`);
  return r.json();
};
const unitWord = (layer) => (layer && layer.kind === "linear" ? "neuron" : "filter");
const subWord = (layer) => (layer && layer.kind === "linear" ? "input weight" : "kernel");
const shapeStr = (s) => (s && s.length ? s.join("×") : "scalar");
const perSample = (s) => (s && s.length > 1 ? s.slice(1) : s);

// ======================================================================
// boot
// ======================================================================
async function init() {
  const theme = store.get("theme", null);
  if (theme) document.documentElement.dataset.theme = theme;
  const [manifest, graph, sparsity] = await Promise.all([api("/api/manifest"), api("/api/graph"), api("/api/sparsity")]);
  state.manifest = manifest;
  state.graph = graph;
  state.sparsity = sparsity;
  for (const l of manifest.layers) state.layerById.set(l.id, l);
  for (const n of graph.nodes) state.nodeById.set(n.id, n);
  const sw = store.get("sidebarWidth", null);
  if (sw) $("sidebar").style.width = sw + "px";

  renderTopbar();
  renderSparsitySummary();
  await loadOverview();
  relayout(true);
  wireGraphInteraction();
  wireControls();
  if (graph.fallback) $("graph-status").textContent = "No traced graph stored with this result -- showing analyzed layers as a chain.";
}

async function loadOverview() {
  state.overview = await api(`/api/overview?output=${state.output}&metric=${state.metric}`);
}

function renderTopbar() {
  const m = state.manifest, ms = m.model_summary;
  $("model-summary").textContent =
    `${ms.class_name} · ${ms.num_parameters.toLocaleString()} params · ${ms.num_analyzed_layers} analyzed layers · ` +
    `${m.dataset_info.n_samples} samples · input ${shapeStr(m.dataset_info.input_shape)}`;
  const sel = $("output-select");
  sel.innerHTML = m.output_features.names
    .map((n, i) => `<option value="${i}">${n === "__all__" ? "all outputs (mean)" : esc(n)}</option>`).join("");
  sel.value = state.output;
  $("metric-select").value = state.metric;
  $("hist-order").value = state.histOrder;
  $("hist-axis").value = state.histAxis;
  $("color-scale").value = state.colorScale;
  setColorScale(state.colorScale);
  document.querySelector(".sw-imp").style.background = seqGradient();
  $("edge-shapes").checked = state.edgeShapes;
  for (const b of $("orient-seg").children) b.classList.toggle("on", b.dataset.v === state.orient);
}

function renderSparsitySummary() {
  let zf = 0, zk = 0, zw = 0, nw = 0, layersWithZeros = 0, nq = 0;
  const units = new Set();
  for (const [id, sp] of Object.entries(state.sparsity)) {
    const i = sp.info;
    zf += i.zero_filters; zk += i.zero_kernels; zw += i.zero_weights; nw += i.n_weights;
    if (i.zero_weights) layersWithZeros++;
    if (i.method === "quantized") nq++;
    units.add(unitWord(state.layerById.get(id)));
  }
  const n = Object.keys(state.sparsity).length;
  const pct = nw ? ((100 * zw) / nw).toFixed(2) : "0";
  const unitLabel = [...units].map((u) => u + (zf === 1 ? "" : "s")).join("/");
  $("sparsity-summary").innerHTML =
    `<div><span class="zero-chip">∅</span> Zero: <b>${zf}</b> ${unitLabel} · <b>${zk}</b> kernel${zk === 1 ? "" : "s"} · <b>${zw.toLocaleString()}</b> weight${zw === 1 ? "" : "s"} (${pct}%)</div>` +
    `<div class="muted">in ${layersWithZeros} of ${n} analyzed layers · ${nq} layer${nq === 1 ? "" : "s"} with exact quantized zeros, ` +
    `${n - nq} with the float tolerance rule</div>`;
  $("sparsity-summary").title = "Click to toggle 'zeros only'";
  $("sparsity-summary").style.cursor = "pointer";
  $("sparsity-summary").onclick = () => { $("zeros-only").checked = !$("zeros-only").checked; state.zerosOnly = $("zeros-only").checked; renderNodes(); };
}

// ======================================================================
// graph: sizing, layout, rendering
// ======================================================================
const HDR = 20, HIST_H = 34, NODE_W = 214;

function nodeLines(n) {
  // Netron-style attribute lines: "W ⟨64×32×3×3⟩", "B ⟨64⟩", hyper-params
  const lines = [];
  if (n.kind === "module") {
    for (const p of n.params || []) {
      const k = p.name === "weight" ? "W" : p.name === "bias" ? "B" : p.name;
      lines.push([k, `⟨${shapeStr(p.shape)}⟩`]);
    }
    const a = n.attributes || {};
    const hp = [];
    const tup = (v) => (Array.isArray(v) ? (v.every((x) => x === v[0]) ? v[0] : v.join("×")) : v);
    if (a.kernel_size != null) hp.push(`k${tup(a.kernel_size)}`);
    if (a.stride != null && tup(a.stride) !== 1) hp.push(`s${tup(a.stride)}`);
    if (a.groups != null && a.groups !== 1) hp.push(`g${a.groups}`);
    if (a.dilation != null && tup(a.dilation) !== 1) hp.push(`d${tup(a.dilation)}`);
    if (hp.length) lines.push(["", hp.join(" ")]);
  }
  return lines.slice(0, 3);
}

function nodeSize(n) {
  if (n.kind === "input" || n.kind === "output") return { w: 120, h: 34 };
  if (n.kind === "op") return { w: Math.max(54, n.type.length * 7 + 26), h: 24 };
  const analyzed = n.analyzed && state.layerById.has(n.module);
  const lines = nodeLines(n).length;
  if (analyzed) return { w: NODE_W, h: HDR + 16 + lines * 12 + 6 + HIST_H + 18 + 6 };
  const w = Math.max(120, Math.min(200, Math.max(n.type.length * 7.2 + 24, (n.name || "").length * 5.9 + 16)));
  return { w, h: HDR + 16 + lines * 12 + 2 };
}

function relayout(fit) {
  const nodes = state.graph.nodes.map((n) => ({ id: n.id, ...nodeSize(n) }));
  state.layout = layoutGraph(nodes, state.graph.edges, { orient: state.orient });
  renderEdges();
  renderNodes();
  if (fit) initialView();
}

function catColor(cat) {
  const v = css(`--cat-${cat}`);
  return v || css("--cat-other");
}

function zeroBadge(sp, layer) {
  if (!sp) return null;
  const i = sp.info;
  const pl = (n, w) => `${n} ${w}${n === 1 ? "" : "s"}`;
  if (i.zero_filters) return `∅ ${pl(i.zero_filters, unitWord(layer))}`;
  if (i.zero_kernels && layer.kind !== "linear") return `∅ ${pl(i.zero_kernels, "kernel")}`;
  if (i.zero_weights) return `∅ ${pl(i.zero_weights, "weight")}`;
  return null;
}

function weightBits(n) {
  const q = n.quantization && n.quantization.weight_quant;
  if (!q || q.enabled === false) return null;
  return q.bit_width != null ? `W${Math.round(q.bit_width)}` : "Wq";
}

/** Scores along one axis of a layer: per filter/neuron (output side) or per
 *  input channel / input feature (summed over the filters that read it). */
function axisData(layer, axis) {
  const ov = state.overview.layers[layer.id];
  const sp = state.sparsity[layer.id];
  if (axis === "input" && ov.inputs) {
    return { axis, vals: ov.inputs, metric: ov.input_metric, zero: sp ? sp.mask_input : null,
      word: layer.kind === "linear" ? "input" : "input channel", zeroWord: "unused" };
  }
  return { axis: "filter", vals: ov.filter, metric: ov.metric, zero: sp ? sp.mask_filter : null,
    word: unitWord(layer), zeroWord: "zero" };
}

// mini histogram: one bar per filter (or input channel), binned to fit; zeros get a red tick under their bar
function histogramSvg(layer, x, y, w, h) {
  const ov = state.overview && state.overview.layers[layer.id];
  if (!ov) return `<rect class="hist-bg" x="${x}" y="${y}" width="${w}" height="${h}" rx="2"/>`;
  const ad = axisData(layer, state.histAxis);
  const vals = ad.vals;
  const zero = ad.zero || vals.map(() => 0);
  if (!vals.some((v) => v !== 0)) {
    return `<rect class="hist-bg" x="${x}" y="${y}" width="${w}" height="${h - 4}" rx="2"/>` +
      `<text class="footer" x="${x + w / 2}" y="${y + h / 2}" text-anchor="middle">no influence on this output</text>`;
  }
  let order = vals.map((_, i) => i);
  if (state.histOrder === "sorted") order.sort((a, b) => Math.abs(vals[b]) - Math.abs(vals[a]) || zero[a] - zero[b]);
  const maxBars = Math.floor(w / 2);
  const per = Math.max(1, Math.ceil(order.length / maxBars));
  const nb = Math.ceil(order.length / per);
  const bw = w / nb;
  const maxAbs = Math.max(1e-30, ...vals.map(Math.abs));
  const signed = vals.some((v) => v < 0);
  const plotH = h - 4;
  const base = signed ? y + plotH / 2 : y + plotH;
  let pos = "", neg = "", zer = "";
  for (let b = 0; b < nb; b++) {
    let best = 0, z = false;
    for (let k = b * per; k < Math.min(order.length, (b + 1) * per); k++) {
      const v = vals[order[k]];
      if (Math.abs(v) > Math.abs(best)) best = v;
      if (zero[order[k]]) z = true;
    }
    const len = Math.max(Math.abs(best) > 0 ? 0.8 : 0, (Math.abs(best) / maxAbs) * (signed ? plotH / 2 : plotH));
    const bx = (x + b * bw).toFixed(2), bwd = Math.max(0.6, bw - (bw > 3 ? 0.6 : 0)).toFixed(2);
    if (len > 0) {
      const by = best >= 0 ? base - len : base;
      const seg = `M${bx},${by.toFixed(2)}h${bwd}v${len.toFixed(2)}h-${bwd}z`;
      if (best < 0) neg += seg; else pos += seg;
    }
    if (z) zer += `M${bx},${y + plotH + 1}h${Math.max(1.2, bw - 0.3).toFixed(2)}v3h-${Math.max(1.2, bw - 0.3).toFixed(2)}z`;
  }
  return `<rect class="hist-bg" x="${x}" y="${y}" width="${w}" height="${plotH}" rx="2"/>` +
    `<path class="hist-bar" d="${pos}"/><path class="hist-neg" d="${neg}"/><path class="hist-zero" d="${zer}"/>`;
}

function maxLayerScore() {
  let m = 1e-30;
  if (state.overview) for (const v of Object.values(state.overview.layers)) m = Math.max(m, v.layer);
  return m;
}

function renderNodes() {
  const L = state.layout;
  const maxScore = maxLayerScore();
  let html = "";
  for (const n of state.graph.nodes) {
    const p = L.positions.get(n.id);
    if (!p) continue;
    const x = p.x - p.w / 2, y = p.y - p.h / 2;
    const layer = n.kind === "module" && n.analyzed ? state.layerById.get(n.module) : null;
    const sp = layer ? state.sparsity[layer.id] : null;
    const hasZero = sp && sp.info.zero_weights > 0;
    const cls = ["node", n.kind === "op" ? "op" : "", n.kind === "input" || n.kind === "output" ? "io" : "",
      state.selected === n.id ? "selected" : "", hasZero ? "has-zero" : "",
      state.zerosOnly && !hasZero ? "dim" : ""].join(" ");
    html += `<g class="${cls}" data-id="${n.id}" transform="translate(${x.toFixed(1)},${y.toFixed(1)})">`;
    if (n.kind === "input" || n.kind === "output") {
      const s = n.kind === "input" ? (n.output_shapes || [])[0] : (n.input_shapes || [])[0];
      html += `<rect class="body" width="${p.w}" height="${p.h}" rx="${p.h / 2}"/>` +
        `<text x="${p.w / 2}" y="14" text-anchor="middle">${esc(n.name)}</text>` +
        `<text x="${p.w / 2}" y="27" text-anchor="middle" style="font:9.5px var(--mono);opacity:.85">${s ? esc(shapeStr(perSample(s))) : ""}</text>`;
    } else if (n.kind === "op") {
      html += `<rect class="body" width="${p.w}" height="${p.h}" rx="12" style="fill:${catColor(n.category)};stroke:none"/>` +
        `<text class="op-text" x="${p.w / 2}" y="16" text-anchor="middle">${esc(n.type)}</text>`;
    } else {
      const color = catColor(n.category);
      html += `<rect class="body" width="${p.w}" height="${p.h}" rx="5"/>` +
        `<path d="M0,5a5,5 0 0 1 5,-5h${p.w - 10}a5,5 0 0 1 5,5v${HDR - 5}h-${p.w}z" style="fill:${color}"/>` +
        `<text class="hdr-text" x="7" y="14">${esc(n.type)}</text>`;
      let bx = p.w - 5;
      if (layer) {
        const zb = zeroBadge(sp, layer);
        if (zb) {
          const bw = zb.length * 5.6 + 10;
          bx -= bw;
          html += `<g class="zero-badge"><rect x="${bx}" y="3" width="${bw}" height="14" rx="7"/><text x="${bx + bw / 2}" y="13.5" text-anchor="middle">${esc(zb)}</text></g>`;
          bx -= 4;
        }
      }
      const wb = weightBits(n);
      if (wb) {
        const bw = wb.length * 6 + 8;
        bx -= bw;
        html += `<g class="q-badge"><rect x="${bx}" y="3" width="${bw}" height="14" rx="3"/><text x="${bx + bw / 2}" y="13.5" text-anchor="middle">${wb}</text></g>`;
      }
      const name = n.name.length * 5.9 > p.w - 12 ? "…" + n.name.slice(-Math.floor((p.w - 18) / 5.9)) : n.name;
      html += `<text class="name" x="7" y="${HDR + 12}">${esc(name)}</text>`;
      let ty = HDR + 26;
      for (const [k, v] of nodeLines(n)) {
        html += `<text class="attr" x="7" y="${ty}">${k ? `<tspan class="k">${esc(k)}</tspan> ` : ""}${esc(v)}</text>`;
        ty += 12;
      }
      if (layer) {
        const hy = ty - 4;
        html += histogramSvg(layer, 7, hy, p.w - 14, HIST_H);
        const ad = axisData(layer, state.histAxis);
        const nz = ad.zero ? ad.zero.reduce((a, b) => a + b, 0) : 0;
        const zwp = sp && sp.info.zero_weights ? ` · w=0 ${((100 * sp.info.zero_weights) / sp.info.n_weights).toFixed(1)}%` : "";
        html += `<text class="footer" x="7" y="${hy + HIST_H + 11}">${ad.vals.length} ${ad.word}s` +
          (nz ? ` · <tspan class="z">${nz} ${ad.zeroWord}</tspan>` : "") + `${zwp}</text>`;
        const ov = state.overview && state.overview.layers[layer.id];
        const frac = ov ? ov.layer / maxScore : 0;
        html += `<rect class="meter-bg" x="0" y="${p.h - 4}" width="${p.w}" height="3"/>` +
          `<rect class="meter" x="0" y="${p.h - 4}" width="${(p.w * frac).toFixed(1)}" height="3"><title>layer importance ${fmt(ov ? ov.layer : 0)} (${(100 * frac).toFixed(0)}% of the most important layer)</title></rect>`;
      }
    }
    html += `</g>`;
  }
  $("nodes-layer").innerHTML = html;
}

function edgePath(pts, TB) {
  let d = `M${pts[0].x.toFixed(1)},${pts[0].y.toFixed(1)}`;
  for (let i = 1; i < pts.length; i++) {
    const a = pts[i - 1], b = pts[i];
    if (TB) { const m = (b.y - a.y) / 2; d += `C${a.x.toFixed(1)},${(a.y + m).toFixed(1)} ${b.x.toFixed(1)},${(b.y - m).toFixed(1)} ${b.x.toFixed(1)},${b.y.toFixed(1)}`; }
    else { const m = (b.x - a.x) / 2; d += `C${(a.x + m).toFixed(1)},${a.y.toFixed(1)} ${(b.x - m).toFixed(1)},${b.y.toFixed(1)} ${b.x.toFixed(1)},${b.y.toFixed(1)}`; }
  }
  return d;
}

function renderEdges() {
  const L = state.layout, TB = L.orient === "TB";
  // spread several edges entering/leaving one node across its border (ports)
  const ins = new Map(), outs = new Map();
  L.edgeRoutes.forEach((r, i) => {
    (outs.get(r.source) || outs.set(r.source, []).get(r.source)).push(i);
    (ins.get(r.target) || ins.set(r.target, []).get(r.target)).push(i);
  });
  const routes = L.edgeRoutes.map((r) => ({ ...r, points: r.points.map((p) => ({ ...p })) }));
  const portOffset = (list, node, getNeighbour) => {
    const p = L.positions.get(node);
    const span = (TB ? p.w : p.h) - 24;
    const sorted = [...list].sort((a, b) => getNeighbour(routes[a]) - getNeighbour(routes[b]));
    const step = sorted.length > 1 ? Math.min(16, span / (sorted.length - 1)) : 0;
    sorted.forEach((ri, k) => routes[ri]._off = (k - (sorted.length - 1) / 2) * step);
  };
  const along = (pt) => (TB ? pt.x : pt.y);
  for (const [node, list] of outs) {
    portOffset(list, node, (r) => along(r.points[1]));
    for (const ri of list) {
      const r = routes[ri], p = L.positions.get(node);
      r.points[0] = TB ? { x: p.x + r._off, y: p.y + p.h / 2 } : { x: p.x + p.w / 2, y: p.y + r._off };
    }
  }
  for (const [node, list] of ins) {
    portOffset(list, node, (r) => along(r.points[r.points.length - 2]));
    for (const ri of list) {
      const r = routes[ri], p = L.positions.get(node);
      r.points[r.points.length - 1] = TB ? { x: p.x + r._off, y: p.y - p.h / 2 - 1 } : { x: p.x - p.w / 2 - 1, y: p.y + r._off };
    }
  }
  let html = "";
  const labelled = new Map(); // source node -> shapes already labelled
  routes.forEach((r) => {
    html += `<path class="edge" data-s="${r.source}" data-t="${r.target}" marker-end="url(#arrow)" d="${edgePath(r.points, TB)}"/>`;
    if (!state.edgeShapes || !r.edge.shape) return;
    const label = shapeStr(perSample(r.edge.shape));
    const done = labelled.get(r.source) || labelled.set(r.source, []).get(r.source);
    if (done.includes(label)) return;
    done.push(label);
    const p = L.positions.get(r.source), k = done.length - 1;
    // anchor at the node's border centre, next to its outgoing edges, stacked if shapes differ
    html += TB ? `<text class="edge-label" x="${(p.x + 5).toFixed(1)}" y="${(p.y + p.h / 2 + 12 + 11 * k).toFixed(1)}">${esc(label)}</text>`
      : `<text class="edge-label" x="${(p.x + p.w / 2 + 5).toFixed(1)}" y="${(p.y - 4 - 11 * k).toFixed(1)}">${esc(label)}</text>`;
  });
  $("edges-layer").innerHTML = html;
  highlightEdges();
}

function highlightEdges() {
  for (const e of $("edges-layer").querySelectorAll(".edge")) {
    e.classList.toggle("hl", !!state.selected && (e.dataset.s === state.selected || e.dataset.t === state.selected));
  }
}

// ======================================================================
// pan / zoom
// ======================================================================
function applyView() {
  const v = state.view;
  $("viewport").setAttribute("transform", `translate(${v.x.toFixed(1)},${v.y.toFixed(1)}) scale(${v.k.toFixed(4)})`);
}

function fitView() {
  const r = $("graph-svg").getBoundingClientRect(), L = state.layout;
  const k = Math.min(r.width / L.width, r.height / L.height, 1.4) * 0.96;
  state.view = { k, x: (r.width - L.width * k) / 2, y: (r.height - L.height * k) / 2 };
  applyView();
}

function initialView() {
  const r = $("graph-svg").getBoundingClientRect(), L = state.layout;
  const kAll = Math.min(r.width / L.width, r.height / L.height);
  if (kAll >= 0.55) return fitView();
  // big graph: readable zoom, start at the input end (like Netron)
  const k = Math.min(1, Math.max(0.55, (state.orient === "TB" ? r.width / L.width : r.height / L.height) * 0.95));
  state.view = state.orient === "TB" ? { k, x: (r.width - L.width * k) / 2, y: 10 } : { k, x: 10, y: (r.height - L.height * k) / 2 };
  applyView();
}

function centerOn(nodeId, animate = true) {
  const p = state.layout.positions.get(nodeId);
  if (!p) return;
  const r = $("graph-svg").getBoundingClientRect();
  const k = Math.max(state.view.k, 0.8);
  const target = { k, x: r.width / 2 - p.x * k, y: r.height / 2 - p.y * k };
  if (!animate) { state.view = target; applyView(); return; }
  const from = { ...state.view }, t0 = performance.now(), dur = 260;
  const step = (t) => {
    const u = Math.min(1, (t - t0) / dur), e = 1 - Math.pow(1 - u, 3);
    state.view = { k: from.k + (target.k - from.k) * e, x: from.x + (target.x - from.x) * e, y: from.y + (target.y - from.y) * e };
    applyView();
    if (u < 1) requestAnimationFrame(step);
  };
  requestAnimationFrame(step);
}

function wireGraphInteraction() {
  const svg = $("graph-svg");
  svg.addEventListener("wheel", (ev) => {
    ev.preventDefault();
    const r = svg.getBoundingClientRect(), mx = ev.clientX - r.left, my = ev.clientY - r.top;
    const v = state.view;
    const k = Math.min(4, Math.max(0.05, v.k * Math.exp(-ev.deltaY * (ev.ctrlKey ? 0.01 : 0.0015))));
    state.view = { k, x: mx - ((mx - v.x) * k) / v.k, y: my - ((my - v.y) * k) / v.k };
    applyView();
  }, { passive: false });
  let drag = null;
  svg.addEventListener("pointerdown", (ev) => {
    if (ev.button !== 0) return;
    drag = { x: ev.clientX, y: ev.clientY, vx: state.view.x, vy: state.view.y, moved: false };
    svg.setPointerCapture(ev.pointerId);
  });
  svg.addEventListener("pointermove", (ev) => {
    if (drag) {
      const dx = ev.clientX - drag.x, dy = ev.clientY - drag.y;
      if (!drag.moved && Math.hypot(dx, dy) > 3) { drag.moved = true; svg.classList.add("panning"); hideTip(); }
      if (drag.moved) { state.view.x = drag.vx + dx; state.view.y = drag.vy + dy; applyView(); }
      return;
    }
    const g = ev.target.closest && ev.target.closest(".node");
    if (g) showTip(ev, nodeTooltip(state.nodeById.get(g.dataset.id))); else hideTip();
  });
  svg.addEventListener("pointerup", (ev) => {
    if (!drag) return;
    svg.classList.remove("panning");
    const wasClick = !drag.moved;
    drag = null;
    if (!wasClick) return;
    const el = document.elementFromPoint(ev.clientX, ev.clientY);
    const g = el && el.closest && el.closest(".node");
    selectNode(g ? g.dataset.id : null, false);
  });
  svg.addEventListener("pointerleave", hideTip);
  svg.addEventListener("dblclick", (ev) => {
    const g = ev.target.closest && ev.target.closest(".node");
    if (g) centerOn(g.dataset.id);
  });
  new ResizeObserver(() => applyView()).observe(svg);
}

function nodeTooltip(n) {
  if (!n) return "";
  let s = `${n.type}  ${n.name}`;
  if (n.output_shapes && n.output_shapes.length) s += `\nout: ${n.output_shapes.map((x) => shapeStr(perSample(x))).join(", ")}`;
  const layer = n.analyzed ? state.layerById.get(n.module) : null;
  if (layer) {
    const sp = state.sparsity[layer.id];
    const ov = state.overview.layers[layer.id];
    s += `\nimportance (${ov.metric}): ${fmt(ov.layer)}`;
    if (sp) s += `\nzero: ${sp.info.zero_filters} ${unitWord(layer)}s, ${sp.info.zero_kernels} kernels, ${sp.info.zero_weights}/${sp.info.n_weights} weights`;
  }
  return s;
}

// ======================================================================
// tooltip
// ======================================================================
function showTip(ev, text) {
  const t = $("tooltip");
  if (!text) return hideTip();
  t.textContent = text;
  t.hidden = false;
  const w = t.offsetWidth, h = t.offsetHeight;
  let x = ev.clientX + 14, y = ev.clientY + 14;
  if (x + w > innerWidth - 4) x = ev.clientX - w - 10;
  if (y + h > innerHeight - 4) y = ev.clientY - h - 10;
  t.style.left = x + "px";
  t.style.top = y + "px";
}
function hideTip() { $("tooltip").hidden = true; }

// ======================================================================
// selection + sidebar
// ======================================================================
function selectNode(id, center = true) {
  state.selected = id;
  for (const g of $("nodes-layer").querySelectorAll(".node")) g.classList.toggle("selected", g.dataset.id === id);
  highlightEdges();
  if (id && center) centerOn(id);
  renderSidebar();
}

function section(key, title, bodyHtml, extraCls = "") {
  const c = state.collapsed.has(key) ? " collapsed" : "";
  return `<section class="sb${c} ${extraCls}" data-key="${key}"><h3>${title}</h3><div class="sb-body" id="sec-${key}">${bodyHtml}</div></section>`;
}

function renderSidebar() {
  const n = state.selected ? state.nodeById.get(state.selected) : null;
  $("sidebar-empty").hidden = !!n;
  const root = $("sidebar-content");
  root.hidden = !n;
  if (!n) return;
  const layer = n.kind === "module" && n.analyzed ? state.layerById.get(n.module) : null;
  const color = n.kind === "input" || n.kind === "output" ? css("--cat-io") : catColor(n.category);
  let html = `<div class="sb-head"><div class="actions"><button id="sb-center" title="Center in graph">⌖</button></div>` +
    `<span class="type" style="background:${color}">${esc(n.type)}</span>` +
    `<div class="name">${esc(n.name)}</div>`;
  if (layer) {
    const ov = state.overview.layers[layer.id];
    html += `<div class="sub">metric <b>${esc(ov.metric)}</b>${layer.has_bn_after ? " (followed by BatchNorm → weight scores are scale-invariant, act_filter is the reliable one)" : ""}</div>`;
  }
  html += `</div>`;

  if (layer) {
    html += section("sparsity", "Sparsity", sparsityHtml(layer));
    html += section("importance", `Importance`, "");
    html += section("unit", `Detail`, "");
  } else if (n.kind === "input") {
    html += section("samples", "Stored samples", "");
  } else if (n.kind === "output") {
    html += section("outputs", "Output features", "");
  } else if (n.kind === "module") {
    html += `<div class="note" style="padding:8px 16px 0">No importance arrays for this module: only Conv1d/2d/3d and Linear layers are analyzed.</div>`;
  }
  // analyzed layers have plenty above; everything else is *only* properties
  html += section(layer ? "props" : "props-node", "Properties", propsHtml(n, layer));
  root.innerHTML = html;

  for (const s of root.querySelectorAll("section.sb > h3")) {
    s.onclick = () => {
      const sec = s.parentElement, key = sec.dataset.key;
      sec.classList.toggle("collapsed");
      if (sec.classList.contains("collapsed")) state.collapsed.add(key); else state.collapsed.delete(key);
      store.set("collapsed", [...state.collapsed]);
      if (!sec.classList.contains("collapsed") && layer) { if (key === "importance") renderImportanceChart(layer); if (key === "unit") renderDetail(layer); }
    };
  }
  $("sb-center").onclick = () => centerOn(n.id);
  if (layer) { renderImportanceChart(layer); renderDetail(layer); }
  if (n.kind === "input") renderSamples();
  if (n.kind === "output") renderOutputs();
}

function propsHtml(n, layer) {
  const rows = [];
  const grp = (t) => rows.push(`<tr class="group"><td colspan="2">${t}</td></tr>`);
  const row = (k, v) => rows.push(`<tr><td class="k">${esc(k)}</td><td class="v">${esc(v)}</td></tr>`);
  grp("Node");
  row("name", n.name);
  row("type", n.type);
  if (n.scope) row("scope", n.scope);
  if (n.input_shapes && n.input_shapes.length) n.input_shapes.forEach((s, i) => row(n.input_shapes.length > 1 ? `input ${i}` : "input", shapeStr(s)));
  if (n.output_shapes && n.output_shapes.length) n.output_shapes.forEach((s, i) => row(n.output_shapes.length > 1 ? `output ${i}` : "output", shapeStr(s)));
  const attrs = n.attributes || {};
  if (Object.keys(attrs).length) {
    grp("Attributes");
    for (const [k, v] of Object.entries(attrs)) row(k, Array.isArray(v) ? `[${v.join(", ")}]` : String(v));
  }
  if (n.params && n.params.length) {
    grp("Parameters");
    for (const p of n.params) row(p.name, `${p.dtype} ⟨${shapeStr(p.shape)}⟩`);
  }
  if (n.quantization) {
    for (const [proxy, q] of Object.entries(n.quantization)) {
      grp(`Quantization · ${proxy}`);
      for (const [k, v] of Object.entries(q)) row(k, String(v));
    }
  }
  if (layer) {
    grp("Importance analysis");
    row("groups", layer.groups);
    row("followed by norm", layer.has_bn_after ? "yes" : "no");
    row("levels stored", Object.keys(layer.levels).join(", "));
    row("execution index", layer.execution_index);
  }
  return `<table class="props">${rows.join("")}</table>`;
}

function sparsityHtml(layer) {
  const sp = state.sparsity[layer.id];
  if (!sp) return `<div class="note">no sparsity data</div>`;
  const i = sp.info, U = unitWord(layer);
  const tile = (v, l, zero) => `<div class="tile${zero && v !== "0" && v !== 0 ? " zero" : ""}"><div class="v">${v}</div><div class="l">${l}</div></div>`;
  let tiles = tile(`${i.zero_filters}<span class="muted" style="font-size:12px">/${i.n_filters}</span>`, `zero ${U}s`, i.zero_filters > 0);
  if (layer.kind !== "linear") tiles += tile(`${i.zero_kernels}<span class="muted" style="font-size:12px">/${i.n_kernels}</span>`, "zero kernels", i.zero_kernels > 0);
  tiles += tile(`${((100 * i.zero_weights) / Math.max(1, i.n_weights)).toFixed(i.zero_weights && i.zero_weights / i.n_weights < 0.01 ? 2 : 1)}%`, `zero weights (${i.zero_weights.toLocaleString()})`, i.zero_weights > 0);
  tiles += tile(`${i.unused_inputs}<span class="muted" style="font-size:12px">/${i.n_inputs}</span>`, layer.kind === "linear" ? "unused inputs" : "unused input ch.", i.unused_inputs > 0);
  let rule;
  if (i.method === "quantized") {
    rule = `Weights are quantized: a weight is zero iff its <b>quantized value is exactly 0</b>.`;
  } else {
    const which = i.threshold === i.rel_bound ? "relative bound" : "mass budget";
    rule = `Float weights: zero iff <code>|w| ≤ ${fmt(i.threshold)}</code> = min(` +
      `<code>max|w|·${i.rel_tol}</code> = ${fmt(i.rel_bound)}, mass budget ${fmt(i.mass_bound)}) — the ${which} decides. ` +
      `The mass budget is the largest cut-off for which all weights below it together hold ≤ ${(100 * i.mass_tol).toFixed(2)}% of the layer's Σ|w|, so many tiny weights can't add up unnoticed.`;
  }
  let warn = "";
  if (i.zero_filters_with_bias) warn = `<div class="warn">${i.zero_filters_with_bias} zero ${U}${i.zero_filters_with_bias > 1 ? "s" : ""} still ha${i.zero_filters_with_bias > 1 ? "ve" : "s"} a non-zero bias → constant output, not removed signal.</div>`;
  return `<div class="tiles">${tiles}</div>${warn}<div class="rule">${rule}</div>`;
}

// ---- importance chart (per filter or per input channel) ------------------------
const axisSel = (layer, ad) => (ad.axis === "input" ? state.inputSel.get(layer.id) : state.unit.get(layer.id)?.filter);

function renderImportanceChart(layer) {
  const el = $("sec-importance");
  if (!el || state.collapsed.has("importance")) return;
  const ov = state.overview.layers[layer.id];
  const sp = state.sparsity[layer.id];
  const ad = axisData(layer, state.axis);
  const vals = ad.vals, W = ad.word;
  el.parentElement.querySelector("h3").textContent = `Importance per ${W}`;
  const order = vals.map((_, i) => i);
  if (state.chartSort === "importance") order.sort((a, b) => Math.abs(vals[b]) - Math.abs(vals[a]));
  const byImp = vals.map((_, i) => i).sort((a, b) => Math.abs(vals[b]) - Math.abs(vals[a]));
  const keepN = Math.round((state.topk / 100) * vals.length);
  const keep = new Set(byImp.slice(0, keepN));
  const total = vals.reduce((a, v) => a + Math.abs(v), 0) || 1;
  const kept = byImp.slice(0, keepN).reduce((a, f) => a + Math.abs(vals[f]), 0);
  const hasInputs = !!ov.inputs;
  el.innerHTML =
    `<div class="toolbar"><div class="seg" id="axis-seg"><button data-v="filter">per ${unitWord(layer)}</button>` +
    `<button data-v="input" ${hasInputs ? "" : "disabled title='no kernel-level data stored'"}>per ${layer.kind === "linear" ? "input" : "input channel"}</button></div>` +
    `order <select id="chart-sort"><option value="index">index</option><option value="importance">importance</option></select>` +
    `<span class="spacer"></span><span class="nowrap">top <input type="range" id="topk" min="1" max="100" value="${state.topk}"/> <span id="topk-l">${state.topk}%</span></span></div>` +
    `<div class="chart-wrap"><canvas class="chart" id="imp-chart"></canvas></div>` +
    `<div class="note">metric <b>${esc(ad.metric)}</b>${ad.axis === "input" ? ` (summed over every ${unitWord(layer)} that reads the ${W}${ov.metric === "act_filter" ? "; act_filter has no per-input breakdown" : ""})` : ""}. ` +
    `${keepN} of ${vals.length} ${W}s (top ${state.topk}%) carry <b>${((100 * kept) / total).toFixed(1)}%</b> of it. ` +
    `<span class="zero-chip">∅</span> hatched = ${ad.axis === "input" ? `unused ${W}s (zero kernel in every ${unitWord(layer)})` : `zero ${W}s`}.</div>` +
    (vals.some((v) => v !== 0) ? "" : `<div class="warn">Every score is 0 for this output: this layer is not on a path to it (e.g. a different detection head).</div>`);
  for (const b of $("axis-seg").children) {
    b.classList.toggle("on", b.dataset.v === ad.axis);
    b.onclick = () => { state.axis = b.dataset.v; store.set("axis", state.axis); renderImportanceChart(layer); renderDetail(layer); };
  }
  $("chart-sort").value = state.chartSort;
  const canvas = $("imp-chart");
  const hit = barChart(canvas, vals, {
    order, zero: ad.zero, keep: state.topk < 100 ? keep : null, selected: axisSel(layer, ad),
    signed: ad.metric === "mean_s" || ad.metric === "act_filter_signed", width: el.clientWidth - 2, height: 130,
  });
  canvas.onmousemove = (ev) => {
    const f = hit(ev.offsetX);
    if (f == null) return hideTip();
    let t = `${W} ${f}\n${ad.metric}: ${fmt(vals[f])}\nrank ${byImp.indexOf(f) + 1} of ${vals.length}`;
    if (sp && ad.axis === "filter") {
      t += sp.mask_filter[f] ? `\n∅ ZERO ${W} (all weights 0)` : "";
      if (sp.zero_kernels_per_filter[f] && layer.kind !== "linear") t += `\nzero kernels: ${sp.zero_kernels_per_filter[f]}`;
      if (sp.zero_weights_per_filter[f]) t += `\nzero weights: ${sp.zero_weights_per_filter[f]}`;
    } else if (sp && sp.mask_input[f]) t += `\n∅ UNUSED: zero kernel in every ${unitWord(layer)}`;
    showTip(ev, t);
  };
  canvas.onmouseleave = hideTip;
  canvas.onclick = (ev) => {
    const f = hit(ev.offsetX);
    if (f == null) return;
    if (ad.axis === "input") state.inputSel.set(layer.id, f);
    else state.unit.set(layer.id, { filter: f, kernel: null });
    renderImportanceChart(layer);
    renderDetail(layer);
  };
  $("chart-sort").onchange = (e) => { state.chartSort = e.target.value; store.set("chartSort", state.chartSort); renderImportanceChart(layer); };
  $("topk").oninput = (e) => { state.topk = +e.target.value; renderImportanceChart(layer); };
}

function renderDetail(layer) {
  const ad = axisData(layer, state.axis);
  const el = $("sec-unit");
  if (el) el.parentElement.querySelector("h3").textContent = `${ad.word} detail`;
  return ad.axis === "input" ? renderInputDetail(layer) : renderUnit(layer);
}

// ---- one input channel across all filters ------------------------------------------
async function renderInputDetail(layer) {
  const el = $("sec-unit");
  if (!el || state.collapsed.has("unit")) return;
  const c = state.inputSel.get(layer.id);
  const U = unitWord(layer), linear = layer.kind === "linear", W = linear ? "input" : "input channel";
  if (c == null) {
    el.innerHTML = `<div class="note">Click a bar above (or press <kbd>→</kbd>) to see how every ${U} uses an ${W}.</div>`;
    return;
  }
  const req = ++unitReq;
  const d = await api(`/api/layer/${encodeURIComponent(layer.id)}/input/${c}?output=${state.output}&metric=${state.metric}`);
  if (req !== unitReq) return;
  const ov = state.overview.layers[layer.id];
  const cells = d.filters.map((f, i) => ({
    value: d.kernel_importance ? d.kernel_importance[i] : 0, zero: !!d.zero_kernels[i],
    kernel: linear ? [[d.weight[i]]] : to2d(d.weight[i]), zeroMask: linear ? [[d.zero_mask[i]]] : to2d(d.zero_mask[i]),
  }));
  const nZero = d.zero_kernels.reduce((a, b) => a + b, 0);
  const sorted = [...cells.map((x) => Math.abs(x.value))].sort((a, b) => b - a);
  const tot = sorted.reduce((a, b) => a + b, 0) || 1;
  let top = 0, n80 = 0;
  for (const v of sorted) { if (top / tot >= 0.8) break; top += v; n80++; }
  el.innerHTML =
    `<div class="unit-head"><button id="u-prev" title="previous (←)">‹</button><span class="title">${W} ${c}</span>` +
    `<button id="u-next" title="next (→)">›</button>` +
    (d.unused ? `<span class="pill zero">∅ unused ${W}</span>` : `<span class="pill ok">${nZero}/${d.filters.length} ${linear ? "weights" : "kernels"} zero</span>`) + `</div>` +
    `<table class="props">` +
    `<tr><td class="k">importance (${esc(ov.input_metric)}, Σ over ${U}s)</td><td class="v">${fmt(ov.inputs[c])}</td></tr>` +
    `<tr><td class="k">read by</td><td class="v">${d.filters.length} ${U}${d.filters.length === 1 ? "" : "s"}${d.filters.length !== (ov.filter.length) ? ` (${U}s ${d.filters[0]}…${d.filters[d.filters.length - 1]}, its group)` : ""}</td></tr>` +
    `<tr><td class="k">80% of it comes from</td><td class="v">${sorted[0] > 0 ? `${n80} ${U}${n80 === 1 ? "" : "s"}` : "– (no importance)"}</td></tr>` +
    `</table>` +
    `<div class="toolbar" style="margin-top:10px"><b style="color:var(--text)">${linear ? `weight of input ${c} in each neuron` : `kernel on channel ${c} in each filter`}</b><span class="spacer"></span>colour by ` +
    `<select id="grid-mode"><option value="importance">importance</option><option value="weights">weight values</option></select></div>` +
    `<div class="chart-wrap"><canvas class="chart" id="kernel-grid"></canvas></div>` +
    `<div class="note">Click a cell to open that ${U} with this ${linear ? "input" : "kernel"} selected.</div>`;
  $("grid-mode").value = state.gridMode;
  const draw = () => {
    const hit = cellGrid($("kernel-grid"), cells, { mode: state.gridMode, width: el.clientWidth - 2 });
    const canvas = $("kernel-grid");
    canvas.onmousemove = (ev) => {
      const i = hit(ev.offsetX, ev.offsetY);
      if (i == null) return hideTip();
      let t = `${U} ${d.filters[i]}\nimportance (${d.kernel_metric || "-"}): ${fmt(cells[i].value)}`;
      if (linear) t += `\nweight: ${fmt(d.weight[i])}${d.qweight ? `  (quantized ${fmt(d.qweight[i])})` : ""}`;
      if (cells[i].zero) t += `\n∅ ZERO ${linear ? "weight" : "kernel"}`;
      showTip(ev, t);
    };
    canvas.onmouseleave = hideTip;
    canvas.onclick = (ev) => {
      const i = hit(ev.offsetX, ev.offsetY);
      if (i == null) return;
      state.axis = "filter";
      store.set("axis", state.axis);
      state.unit.set(layer.id, { filter: d.filters[i], kernel: d.local_channel });
      renderImportanceChart(layer);
      renderDetail(layer);
    };
  };
  draw();
  $("grid-mode").onchange = (e) => { state.gridMode = e.target.value; store.set("gridMode", state.gridMode); draw(); };
  $("u-prev").onclick = () => stepUnit(layer, -1);
  $("u-next").onclick = () => stepUnit(layer, +1);
}

// ---- one filter / neuron -------------------------------------------------------
let unitReq = 0;
async function renderUnit(layer) {
  const el = $("sec-unit");
  if (!el || state.collapsed.has("unit")) return;
  const unit = state.unit.get(layer.id);
  const U = unitWord(layer);
  if (!unit || unit.filter == null) {
    el.innerHTML = `<div class="note">Click a bar above (or press <kbd>→</kbd>) to inspect a ${U}.</div>`;
    return;
  }
  const req = ++unitReq;
  const d = await api(`/api/layer/${encodeURIComponent(layer.id)}/unit/${unit.filter}?output=${state.output}&metric=${state.metric}`);
  if (req !== unitReq) return; // a newer selection superseded this one
  const ov = state.overview.layers[layer.id];
  const C = d.shape[0];
  const linear = layer.kind === "linear";
  const kernels = [];
  for (let c = 0; c < C; c++) {
    const wv = linear ? [[d.weight[c]]] : to2d(d.weight[c]);
    const zm = linear ? [[d.zero_mask[c]]] : to2d(d.zero_mask[c]);
    kernels.push({ value: d.kernel_importance ? d.kernel_importance[c] : 0, zero: !!d.zero_kernels[c], kernel: wv, zeroMask: zm });
  }
  const nZeroK = d.zero_kernels.reduce((a, b) => a + b, 0);
  const nZeroW = d.zero_mask.flat(Infinity).reduce((a, b) => a + b, 0);
  const nW = d.zero_mask.flat(Infinity).length;
  let head = `<div class="unit-head"><button id="u-prev" title="previous (←)">‹</button><span class="title">${U} ${d.filter}</span>` +
    `<button id="u-next" title="next (→)">›</button>` +
    (d.is_zero_filter ? `<span class="pill zero">∅ zero ${U}</span>` : `<span class="pill ok">${nZeroW}/${nW} weights zero</span>`) + `</div>`;
  head += `<table class="props">` +
    `<tr><td class="k">importance (${esc(ov.metric)})</td><td class="v">${fmt(ov.filter[d.filter])}</td></tr>` +
    (d.bias != null ? `<tr><td class="k">bias</td><td class="v">${fmt(d.bias)}${d.bias_zero ? ' <span class="zero-chip">∅</span>' : ""}${d.bias_mean_abs_s != null ? ` · importance ${fmt(d.bias_mean_abs_s)}` : ""}</td></tr>` : "") +
    (!linear ? `<tr><td class="k">zero kernels</td><td class="v">${nZeroK} / ${C}</td></tr>` : "") +
    (d.channel_offset ? `<tr><td class="k">input channels</td><td class="v">${d.channel_offset} … ${d.channel_offset + C - 1} (group ${d.channel_offset / C})</td></tr>` : "") +
    `</table>`;
  const sub = linear ? "input weights" : "kernels (one per input channel)";
  head += `<div class="toolbar" style="margin-top:10px"><b style="color:var(--text)">${sub}</b><span class="spacer"></span>colour by ` +
    `<select id="grid-mode"><option value="importance">importance</option><option value="weights">weight values</option></select></div>` +
    `<div class="chart-wrap"><canvas class="chart" id="kernel-grid"></canvas></div><div id="weight-detail"></div>`;
  el.innerHTML = head;
  $("grid-mode").value = state.gridMode;
  const draw = () => {
    const hit = cellGrid($("kernel-grid"), kernels, { mode: state.gridMode, selected: unit.kernel, width: el.clientWidth - 2 });
    const canvas = $("kernel-grid");
    canvas.onmousemove = (ev) => {
      const c = hit(ev.offsetX, ev.offsetY);
      if (c == null) return hideTip();
      const k = kernels[c];
      let t = `${linear ? "input" : "input channel"} ${d.channel_offset + c}\nimportance (${d.kernel_metric || "-"}): ${fmt(k.value)}`;
      if (linear) t += `\nweight: ${fmt(d.weight[c])}${d.qweight ? `  (quantized ${fmt(d.qweight[c])})` : ""}`;
      const zc = k.zeroMask.flat().reduce((a, b) => a + b, 0);
      if (k.zero) t += `\n∅ ZERO ${linear ? "weight" : "kernel"}`; else if (zc) t += `\nzero weights: ${zc}`;
      const unusedIn = state.sparsity[layer.id]?.mask_input?.[d.channel_offset + c];
      if (unusedIn) t += `\n∅ this input channel is unused by every ${U}`;
      showTip(ev, t);
    };
    canvas.onmouseleave = hideTip;
    canvas.onclick = (ev) => {
      const c = hit(ev.offsetX, ev.offsetY);
      if (c == null) return;
      unit.kernel = c;
      draw();
      renderWeights(layer, d, c);
    };
  };
  draw();
  $("grid-mode").onchange = (e) => { state.gridMode = e.target.value; store.set("gridMode", state.gridMode); draw(); };
  $("u-prev").onclick = () => stepUnit(layer, -1);
  $("u-next").onclick = () => stepUnit(layer, +1);
  if (unit.kernel != null && unit.kernel < C) renderWeights(layer, d, unit.kernel);
}

function to2d(x) {
  if (!Array.isArray(x)) return [[x]];
  if (!Array.isArray(x[0])) return [x];
  if (!Array.isArray(x[0][0])) return x;
  return x.flat(1); // 3-D kernel: stack depth slices vertically
}

function renderWeights(layer, d, c) {
  const el = $("weight-detail");
  const linear = layer.kind === "linear";
  const pick = (arr) => (arr ? arr[c] : null);
  const vals = pick(d.qweight) ?? pick(d.weight);
  const floatVals = pick(d.weight);
  const imp = pick(d.weight_importance), sgn = pick(d.weight_signed), zm = pick(d.zero_mask);
  if (linear) {
    el.innerHTML = `<table class="props" style="margin-top:10px">` +
      `<tr class="group"><td colspan="2">input ${d.channel_offset + c}${zm ? ' <span class="pill zero">∅ zero</span>' : ""}</td></tr>` +
      `<tr><td class="k">weight</td><td class="v">${fmt(floatVals)}</td></tr>` +
      (d.qweight ? `<tr><td class="k">quantized weight</td><td class="v">${fmt(vals)}</td></tr>` : "") +
      (imp != null ? `<tr><td class="k">importance (${esc(d.weight_metric)})</td><td class="v">${fmt(imp)}</td></tr>` : "") +
      (sgn != null ? `<tr><td class="k">signed importance</td><td class="v">${fmt(sgn)}</td></tr>` : "") + `</table>`;
    return;
  }
  const tbl = (m, caption, kind, zeroText = true) => {
    const M = to2d(m), Z = to2d(zm);
    const mx = Math.max(1e-30, ...M.flat().map(Math.abs));
    let h = `<div class="heatmap"><div class="cap">${caption}</div><table class="hm">`;
    M.forEach((row, a) => {
      h += "<tr>";
      row.forEach((v, b) => {
        const z = Z[a] && Z[a][b];
        const t = impT(v, mx);
        const bg = kind === "seq" ? seqColor(t) : resolveColor(divColor(v / mx));
        const txt = kind === "seq" ? seqText(t) : Math.abs(v) / mx > 0.55 ? "#fff" : "";
        h += `<td class="${z ? "zero" : ""}" style="background:${bg};${txt && !z ? `color:${txt}` : ""}" title="${z ? "ZERO weight · " : ""}${v}">${z && zeroText ? "0" : fmtShort(v)}</td>`;
      });
      h += "</tr>";
    });
    return h + `</table></div>`;
  };
  let html = `<div class="note" style="margin-top:12px"><b style="color:var(--text)">kernel of input channel ${d.channel_offset + c}</b>` +
    `${d.zero_kernels[c] ? ' <span class="pill zero">∅ zero kernel</span>' : ""}</div><div class="heatmaps">`;
  html += tbl(vals, d.qweight ? "quantized weight" : "weight", "div");
  if (d.qweight) html += tbl(floatVals, "float weight (pre-quant; outlined = 0 after quant)", "div", false);
  if (imp != null) html += tbl(imp, `importance (${esc(d.weight_metric)})`, "seq");
  if (sgn != null) html += tbl(sgn, "signed importance (mean s)", "div");
  html += `</div>`;
  el.innerHTML = html;
}

function stepUnit(layer, dir) {
  const ad = axisData(layer, state.axis);
  const vals = ad.vals;
  const order = vals.map((_, i) => i);
  if (state.chartSort === "importance") order.sort((a, b) => Math.abs(vals[b]) - Math.abs(vals[a]));
  const cur = axisSel(layer, ad);
  let i = cur == null ? (dir > 0 ? 0 : order.length - 1) : order.indexOf(cur) + dir;
  i = Math.max(0, Math.min(order.length - 1, i));
  if (ad.axis === "input") state.inputSel.set(layer.id, order[i]);
  else state.unit.set(layer.id, { filter: order[i], kernel: state.unit.get(layer.id)?.kernel ?? null });
  renderImportanceChart(layer);
  renderDetail(layer);
}

// ---- input / output nodes ----------------------------------------------------
async function renderSamples() {
  const el = $("sec-samples");
  const count = (state.manifest.samples && state.manifest.samples.count) || 0;
  if (!count) { el.innerHTML = `<div class="note">No samples stored (store_samples=0).</div>`; return; }
  el.innerHTML = `<div class="sample-strip" id="sample-strip"></div><div class="note">Click a sample to see the model's outputs for it on the Output node.</div>`;
  const strip = $("sample-strip");
  for (let i = 0; i < count; i++) {
    const row = await api(`/api/samples/inputs?index=${i}`);
    const c = document.createElement("canvas");
    c.className = i === state.sample ? "sel" : "";
    c.title = `sample ${i} · ${shapeStr(row.shape)}`;
    if (row.shape.length <= 3) renderInputThumb(c, row.data, row.shape, 72);
    c.onclick = () => { state.sample = i; for (const x of strip.children) x.classList.remove("sel"); c.classList.add("sel"); };
    strip.appendChild(c);
  }
}

async function renderOutputs() {
  const el = $("sec-outputs");
  const names = state.manifest.output_features.names;
  let values = null;
  if (state.manifest.samples && state.manifest.samples.count) {
    try { values = (await api(`/api/samples/outputs?index=${state.sample}`)).data; } catch { values = null; }
  }
  const mx = values ? Math.max(1e-30, ...values.map(Math.abs)) : 1;
  el.innerHTML = `<div class="note">Importance everywhere in the graph is measured against the selected output feature. ` +
    `${values ? `Values: model output for stored sample ${state.sample}.` : ""}</div><div class="out-list">` +
    names.map((n, i) => `<div class="out-row${i === state.output ? " sel" : ""}" data-i="${i}"><span>${n === "__all__" ? "all (mean)" : esc(n)}</span>` +
      `<span>${values && values[i] != null ? `<div class="bar" style="width:${(100 * Math.abs(values[i])) / mx}%;${values[i] < 0 ? "background:var(--neg)" : ""}"></div>` : ""}</span>` +
      `<span class="num">${values && values[i] != null ? fmt(values[i]) : ""}</span></div>`).join("") + `</div>`;
  for (const r of el.querySelectorAll(".out-row")) r.onclick = () => setOutput(+r.dataset.i);
}

async function setOutput(i) {
  state.output = i;
  $("output-select").value = i;
  await loadOverview();
  renderNodes();
  renderSidebar();
}

// ======================================================================
// search, controls, keyboard
// ======================================================================
function wireSearch() {
  const input = $("search"), box = $("search-results");
  let items = [], active = 0;
  const render = () => {
    box.innerHTML = items.map((n, i) => `<div class="${i === active ? "active" : ""}" data-id="${n.id}"><span>${esc(n.name)}</span><span class="t">${esc(n.type)}</span></div>`).join("");
    box.hidden = !items.length;
    for (const d of box.children) d.onmousedown = (ev) => { ev.preventDefault(); pick(d.dataset.id); };
  };
  const pick = (id) => { box.hidden = true; input.blur(); selectNode(id); };
  input.oninput = () => {
    const q = input.value.trim().toLowerCase();
    items = q ? state.graph.nodes.filter((n) => n.name.toLowerCase().includes(q) || n.type.toLowerCase().includes(q)).slice(0, 40) : [];
    active = 0;
    render();
  };
  input.onkeydown = (ev) => {
    if (ev.key === "ArrowDown") { active = Math.min(items.length - 1, active + 1); render(); ev.preventDefault(); }
    else if (ev.key === "ArrowUp") { active = Math.max(0, active - 1); render(); ev.preventDefault(); }
    else if (ev.key === "Enter" && items[active]) pick(items[active].id);
    else if (ev.key === "Escape") { box.hidden = true; input.blur(); }
  };
  input.onblur = () => setTimeout(() => (box.hidden = true), 120);
}

function wireControls() {
  wireSearch();
  $("output-select").onchange = (e) => setOutput(+e.target.value);
  $("metric-select").onchange = async (e) => { state.metric = e.target.value; await loadOverview(); renderNodes(); renderSidebar(); };
  $("hist-order").onchange = (e) => { state.histOrder = e.target.value; store.set("histOrder", state.histOrder); renderNodes(); };
  $("hist-axis").onchange = (e) => { state.histAxis = e.target.value; store.set("histAxis", state.histAxis); renderNodes(); };
  $("color-scale").onchange = (e) => {
    state.colorScale = e.target.value;
    store.set("colorScale", state.colorScale);
    setColorScale(state.colorScale);
    renderSidebar();
  };
  $("edge-shapes").onchange = (e) => { state.edgeShapes = e.target.checked; store.set("edgeShapes", state.edgeShapes); renderEdges(); };
  $("zeros-only").onchange = (e) => { state.zerosOnly = e.target.checked; renderNodes(); };
  for (const b of $("orient-seg").children) {
    b.onclick = () => {
      state.orient = b.dataset.v;
      store.set("orient", state.orient);
      for (const x of $("orient-seg").children) x.classList.toggle("on", x === b);
      relayout(true);
      if (state.selected) centerOn(state.selected, false);
    };
  }
  $("fit-btn").onclick = fitView;
  $("export-csv").onclick = exportCSV;
  $("theme-btn").onclick = () => {
    const root = document.documentElement;
    const dark = root.dataset.theme ? root.dataset.theme === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
    root.dataset.theme = dark ? "light" : "dark";
    store.set("theme", root.dataset.theme);
    clearColorCache();
    renderNodes();
    renderSidebar();
  };
  // sidebar splitter
  const sp = $("splitter");
  sp.onpointerdown = (ev) => {
    sp.setPointerCapture(ev.pointerId);
    sp.classList.add("drag");
    const x0 = ev.clientX, w0 = $("sidebar").offsetWidth;
    sp.onpointermove = (e) => { $("sidebar").style.width = Math.max(320, Math.min(innerWidth - 300, w0 - (e.clientX - x0))) + "px"; };
    sp.onpointerup = () => {
      sp.onpointermove = null;
      sp.classList.remove("drag");
      store.set("sidebarWidth", $("sidebar").offsetWidth);
      renderSidebar();
    };
  };
  document.addEventListener("keydown", (ev) => {
    if (ev.target.matches("input, select, textarea")) return;
    const n = state.selected && state.nodeById.get(state.selected);
    const layer = n && n.analyzed ? state.layerById.get(n.module) : null;
    if (ev.key === "/") { ev.preventDefault(); $("search").focus(); }
    else if (ev.key === "f" || ev.key === "F") fitView();
    else if (ev.key === "Escape") selectNode(null, false);
    else if (layer && (ev.key === "ArrowRight" || ev.key === "ArrowLeft")) { ev.preventDefault(); stepUnit(layer, ev.key === "ArrowRight" ? 1 : -1); }
  });
}

function exportCSV() {
  const names = state.manifest.output_features.names;
  const rows = [["layer", "unit", "metric", "output", "value", "zero_unit", "zero_kernels", "zero_weights"]];
  for (const [id, ov] of Object.entries(state.overview.layers)) {
    const sp = state.sparsity[id];
    ov.filter.forEach((v, f) => rows.push([id, f, ov.metric, names[state.output], v,
      sp ? sp.mask_filter[f] : "", sp ? sp.zero_kernels_per_filter[f] : "", sp ? sp.zero_weights_per_filter[f] : ""]));
  }
  const a = document.createElement("a");
  a.href = URL.createObjectURL(new Blob([rows.map((r) => r.join(",")).join("\n")], { type: "text/csv" }));
  a.download = `importance_${names[state.output]}.csv`;
  a.click();
}

init().catch((e) => {
  document.body.innerHTML = `<pre style="color:#d92d3a;padding:20px;white-space:pre-wrap">Failed to load viewer: ${esc(e)}\n${esc(e.stack || "")}</pre>`;
});
