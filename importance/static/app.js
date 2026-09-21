// Plain-JS, no-build-step viewer for importance.analyze() results.
// Everything is driven by /api/manifest plus small per-output-row JSON
// slices fetched on demand -- see importance/serve.py.
"use strict";

const state = {
  manifest: null,
  selectedOutput: 0, // index into manifest.output_features.names
  metric: "auto",
  normalize: "per-layer",
  weightNormalize: "raw", // "raw" | "max100" -- for actual weight *values*, not importance scores
  sort: "original",
  topkPct: 100,
  search: "",
  expanded: new Set(), // layer ids
  selectedFilter: {}, // layerId -> filter index
  selectedKernel: {}, // layerId -> channel index
  sampleIndex: 0,
  cache: new Map(), // "layerId/level/metric/output" -> {shape,data}
  _globalMax: null, // shared scale for "raw"/"global" normalization, see computeGlobalMax()
};

function effectiveMetric(layerEntry) {
  if (state.metric !== "auto") return state.metric;
  return layerEntry.has_bn_after ? "act_filter" : "mean_abs_s";
}

// act_filter (and the bias_* metrics) only exist at the "filter" level --
// kernel/weight only ever have mean_abs_s/mean_s. Drilling down must fall
// back to a metric that actually has data at that level.
function metricForLevel(layerEntry, level) {
  const desired = effectiveMetric(layerEntry);
  const available = (layerEntry.levels && layerEntry.levels[level]) || {};
  if (available[desired]) return desired;
  if (available["mean_abs_s"]) return "mean_abs_s";
  const keys = Object.keys(available);
  return keys[0];
}

async function fetchJSON(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url} -> ${res.status}`);
  return res.json();
}

async function getLayerSlice(layerId, level, metric, outputIdx) {
  const key = `${layerId}/${level}/${metric}/${outputIdx}`;
  if (state.cache.has(key)) return state.cache.get(key);
  const data = await fetchJSON(`/api/layer/${layerId}/${level}/${metric}?output=${outputIdx}`);
  state.cache.set(key, data);
  return data;
}

// Actual (float) parameter values -- not an importance score, not
// output-dependent. kind is "weight" or "bias".
async function getRawArray(layerId, kind) {
  const key = `raw/${layerId}/${kind}`;
  if (state.cache.has(key)) return state.cache.get(key);
  const data = await fetchJSON(`/api/layer/${layerId}/raw/${kind}`);
  state.cache.set(key, data);
  return data;
}

function deepFlatten(x, out = []) {
  if (Array.isArray(x)) x.forEach((v) => deepFlatten(v, out));
  else out.push(x);
  return out;
}

function scaleNested(x, factor) {
  if (Array.isArray(x)) return x.map((v) => scaleNested(v, factor));
  return x * factor;
}

// Cached per-layer max |weight| for the weightNormalize="max100" mode.
const _layerWeightMaxCache = {};
async function getLayerWeightMax(layerId) {
  if (layerId in _layerWeightMaxCache) return _layerWeightMaxCache[layerId];
  const raw = await getRawArray(layerId, "weight");
  const flat = deepFlatten(raw.data);
  const maxAbs = Math.max(...flat.map((v) => Math.abs(v)), 1e-12);
  _layerWeightMaxCache[layerId] = maxAbs;
  return maxAbs;
}

// Returns {values, scaleMax} -- scaleMax is the fixed denominator drawBarChart
// should use (null means "auto: scale to this array's own max", which is what
// makes "per-layer" mode look different from "raw"/"global": those two use a
// scale shared across every layer, computed once by computeGlobalMax(), so a
// globally-unimportant layer's bars honestly look small instead of always
// being stretched to fill the chart.
function computeScale(values, mode) {
  const arr = values.map((v) => Math.abs(v));
  if (mode === "raw") {
    const gmax = state._globalMax != null ? state._globalMax : Math.max(...arr, 1e-12);
    return { values, scaleMax: gmax };
  }
  if (mode === "global") {
    const gmax = state._globalMax != null ? state._globalMax : Math.max(...arr, 1e-12);
    return { values: values.map((v) => v / gmax), scaleMax: 1 };
  }
  if (mode === "rank") {
    const idx = values.map((_, i) => i).sort((a, b) => arr[a] - arr[b]);
    const ranks = new Array(values.length);
    idx.forEach((origIdx, rankPos) => (ranks[origIdx] = rankPos / Math.max(1, values.length - 1)));
    return { values: ranks, scaleMax: 1 };
  }
  // "per-layer" (default): let drawBarChart auto-scale to this chart's own max
  return { values, scaleMax: null };
}

// Fetches every layer's filter-level values (for the currently selected
// metric/output) once, so "raw"/"global" normalization have a real shared
// scale to compare against instead of silently falling back to a per-chart
// max (which is what made those modes indistinguishable from "per-layer").
async function computeGlobalMax() {
  let maxVal = 1e-12;
  for (const layerEntry of state.manifest.layers) {
    if (!layerEntry.levels.filter) continue;
    const metric = metricForLevel(layerEntry, "filter");
    if (!metric) continue;
    try {
      const row = await getLayerSlice(layerEntry.id, "filter", metric, state.selectedOutput);
      for (const v of row.data) {
        const av = Math.abs(v);
        if (av > maxVal) maxVal = av;
      }
    } catch (e) {
      // layer has no data for this metric -- skip it
    }
  }
  state._globalMax = maxVal;
}

function drawBarChart(canvas, values, opts = {}) {
  const dpr = window.devicePixelRatio || 1;
  const w = canvas.clientWidth || 300;
  const h = canvas.clientHeight || 60;
  canvas.width = w * dpr;
  canvas.height = h * dpr;
  const ctx = canvas.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  if (!values.length) return;
  const maxAbs = opts.scaleMax != null ? opts.scaleMax : Math.max(...values.map((v) => Math.abs(v)), 1e-12);
  const barW = w / values.length;
  values.forEach((v, i) => {
    const norm = v / maxAbs;
    const barH = Math.abs(norm) * (h - 4);
    const x = i * barW;
    ctx.fillStyle = v < 0 ? getCss("--neg") : (opts.color || getCss("--accent"));
    if (opts.baseline === "center") {
      const mid = h / 2;
      const y = norm >= 0 ? mid - barH / 2 : mid;
      ctx.fillRect(x, y, Math.max(1, barW - 1), barH / 2 || 1);
    } else {
      ctx.fillRect(x, h - barH, Math.max(1, barW - 1), barH);
    }
    if (opts.selectedIndex === i) {
      ctx.strokeStyle = getCss("--accent");
      ctx.strokeRect(x + 0.5, 0.5, barW - 1, h - 1);
    }
  });
}

function getCss(varName) {
  return getComputedStyle(document.documentElement).getPropertyValue(varName).trim() || "#5ea1ff";
}

function topKKeepFraction(values) {
  const abs = values.map((v) => Math.abs(v)).sort((a, b) => b - a);
  const total = abs.reduce((a, b) => a + b, 0) || 1;
  const keepN = Math.max(0, Math.round((state.topkPct / 100) * abs.length));
  const kept = abs.slice(0, keepN).reduce((a, b) => a + b, 0);
  return { keepN, retainedFraction: kept / total };
}

async function init() {
  state.manifest = await fetchJSON("/api/manifest");
  renderHeader();
  await renderInputPanel();
  renderOutputChips();
  await renderLayers();
  wireControls();
}

function renderHeader() {
  const m = state.manifest.model_summary;
  document.getElementById("model-summary").textContent =
    `${m.class_name} — ${m.num_parameters.toLocaleString()} params, ` +
    `${m.num_analyzed_layers} analyzed layers, ${state.manifest.dataset_info.n_samples} samples ` +
    `(path: ${state.manifest.settings.path_used})`;
}

function renderOutputChips() {
  const names = state.manifest.output_features.names;
  const container = document.getElementById("output-chips");
  container.innerHTML = "";
  names.forEach((name, idx) => {
    const chip = document.createElement("div");
    chip.className = "chip" + (idx === state.selectedOutput ? " selected" : "");
    chip.textContent = name;
    chip.onclick = async () => {
      state.selectedOutput = idx;
      state.cache.clear();
      state._globalMax = null;
      renderOutputChips();
      await renderLayers();
      await renderOutputValues();
    };
    container.appendChild(chip);
  });
  renderOutputValues();
}

async function renderOutputValues() {
  const el = document.getElementById("output-values");
  if (!state.manifest.samples || !state.manifest.samples.count) {
    el.innerHTML = '<div class="muted">no stored samples</div>';
    return;
  }
  try {
    const row = await fetchJSON(`/api/samples/outputs?index=${state.sampleIndex}`);
    const names = state.manifest.output_features.names;
    let html = '<table class="value-table"><tr><th>output</th><th>value</th></tr>';
    row.data.forEach((v, i) => {
      const sel = i === state.selectedOutput ? ' style="color:var(--accent)"' : "";
      html += `<tr${sel}><td>${names[i] ?? i}</td><td>${v.toFixed(4)}</td></tr>`;
    });
    html += "</table>";
    el.innerHTML = html;
  } catch (e) {
    el.innerHTML = `<div class="muted">${e}</div>`;
  }
}

async function renderInputPanel() {
  const sel = document.getElementById("sample-select");
  const count = (state.manifest.samples && state.manifest.samples.count) || 0;
  sel.innerHTML = "";
  for (let i = 0; i < count; i++) {
    const opt = document.createElement("option");
    opt.value = i;
    opt.textContent = `sample ${i}`;
    sel.appendChild(opt);
  }
  sel.onchange = async () => {
    state.sampleIndex = parseInt(sel.value, 10);
    await renderInputImage();
    await renderOutputValues();
  };
  await renderInputImage();
}

async function renderInputImage() {
  const container = document.getElementById("input-render");
  const count = (state.manifest.samples && state.manifest.samples.count) || 0;
  if (!count) {
    container.innerHTML = '<div class="muted">no stored samples</div>';
    return;
  }
  const row = await fetchJSON(`/api/samples/inputs?index=${state.sampleIndex}`);
  const shape = row.shape; // [C, ...spatial] or [C, L] or [C]
  container.innerHTML = "";
  if (shape.length === 3) {
    // [C, H, W]
    renderImageTensor(container, row.data, shape);
  } else if (shape.length === 2) {
    // [C, L] treat as 1D signal(s)
    render1DSignal(container, row.data, shape);
  } else {
    container.textContent = `input shape ${JSON.stringify(shape)} (no renderer for this rank yet)`;
  }
}

function renderImageTensor(container, data, shape) {
  const [C, H, W] = shape;
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  canvas.style.width = Math.min(256, W * 4) + "px";
  canvas.style.height = "auto";
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(W, H);
  let mn = Infinity, mx = -Infinity;
  for (const c of data) for (const row of c) for (const v of row) { if (v < mn) mn = v; if (v > mx) mx = v; }
  const range = mx - mn || 1;
  for (let y = 0; y < H; y++) {
    for (let x = 0; x < W; x++) {
      const idx = (y * W + x) * 4;
      if (C >= 3) {
        img.data[idx] = ((data[0][y][x] - mn) / range) * 255;
        img.data[idx + 1] = ((data[1][y][x] - mn) / range) * 255;
        img.data[idx + 2] = ((data[2][y][x] - mn) / range) * 255;
      } else {
        const v = ((data[0][y][x] - mn) / range) * 255;
        img.data[idx] = img.data[idx + 1] = img.data[idx + 2] = v;
      }
      img.data[idx + 3] = 255;
    }
  }
  ctx.putImageData(img, 0, 0);
  container.appendChild(canvas);
  const label = document.createElement("div");
  label.className = "muted";
  label.textContent = `shape [${shape.join(", ")}]`;
  container.appendChild(label);
}

function render1DSignal(container, data, shape) {
  const canvas = document.createElement("canvas");
  canvas.className = "bar-chart";
  canvas.style.height = "80px";
  container.appendChild(canvas);
  drawBarChart(canvas, data[0] || [], { baseline: "bottom" });
}

async function renderLayers() {
  const container = document.getElementById("layers-list");
  if ((state.normalize === "raw" || state.normalize === "global") && state._globalMax == null) {
    container.innerHTML = '<div class="muted">computing global scale…</div>';
    await computeGlobalMax();
  }
  container.innerHTML = "";
  let layers = state.manifest.layers.filter((l) =>
    l.id.toLowerCase().includes(state.search.toLowerCase())
  );

  for (const layerEntry of layers) {
    const card = buildLayerCard(layerEntry);
    container.appendChild(card);
  }
}

function buildLayerCard(layerEntry) {
  const card = document.createElement("div");
  card.className = "layer-card" + (state.expanded.has(layerEntry.id) ? " expanded" : "");

  const header = document.createElement("div");
  header.className = "layer-card-header";
  header.innerHTML =
    `<span class="layer-name">${layerEntry.id}</span>` +
    `<span class="layer-type">${layerEntry.type}${layerEntry.has_bn_after ? " → BN" : ""}</span>` +
    `<span class="layer-shape">${(layerEntry.weight_shape || []).join("×")}</span>`;
  header.onclick = async () => {
    if (state.expanded.has(layerEntry.id)) state.expanded.delete(layerEntry.id);
    else state.expanded.add(layerEntry.id);
    await renderLayers();
  };
  card.appendChild(header);

  const isExpanded = state.expanded.has(layerEntry.id);

  const body = document.createElement("div");
  body.className = "layer-card-body";
  card.appendChild(body);

  if (!isExpanded) {
    // Collapsed preview: a small filter-level bar chart. Once expanded, the
    // interactive chart inside layer-card-body (with drill-down) takes over
    // instead of showing this same chart again.
    const canvas = document.createElement("canvas");
    canvas.className = "bar-chart";
    header.appendChild(document.createElement("br"));
    card.appendChild(canvas);

    const level = "filter" in layerEntry.levels ? "filter" : Object.keys(layerEntry.levels)[0];
    const metric = metricForLevel(layerEntry, level);
    if (level && metric) {
      getLayerSlice(layerEntry.id, level, metric, state.selectedOutput).then((row) => {
        const { values, scaleMax } = computeScale(row.data, state.normalize);
        drawBarChart(canvas, values, { scaleMax });
      }).catch(() => {});
    }
  } else {
    renderLayerBody(body, layerEntry);
  }
  return card;
}

async function renderLayerBody(body, layerEntry) {
  body.innerHTML = '<div class="muted">loading…</div>';
  const levels = layerEntry.levels;
  if (!levels.filter) {
    body.innerHTML = '<div class="muted">no filter-level data for this layer</div>';
    return;
  }
  const metric = metricForLevel(layerEntry, "filter");
  let filterRow;
  try {
    filterRow = await getLayerSlice(layerEntry.id, "filter", metric, state.selectedOutput);
  } catch (e) {
    body.innerHTML = `<div class="muted">failed to load filter data: ${e}</div>`;
    return;
  }
  let values = filterRow.data;
  let order = values.map((_, i) => i);
  if (state.sort === "importance") order.sort((a, b) => Math.abs(values[b]) - Math.abs(values[a]));

  const { keepN, retainedFraction } = topKKeepFraction(values);
  const keepSet = new Set(order.slice(0, keepN));

  body.innerHTML = "";
  const summary = document.createElement("div");
  summary.className = "muted";
  summary.textContent = `metric=${metric}, top ${state.topkPct}% of filters retain ` +
    `${(retainedFraction * 100).toFixed(1)}% of total importance`;
  body.appendChild(summary);

  const canvas = document.createElement("canvas");
  canvas.className = "bar-chart";
  canvas.style.height = "100px";
  body.appendChild(canvas);
  const orderedValues = order.map((i) => (keepSet.has(i) ? values[i] : 0));
  const { values: normOrdered, scaleMax } = computeScale(orderedValues, state.normalize);
  const selFilter = state.selectedFilter[layerEntry.id];
  const selIndex = selFilter !== undefined ? order.indexOf(selFilter) : -1;
  drawBarChart(canvas, normOrdered, { selectedIndex: selIndex, scaleMax });

  canvas.onclick = async (ev) => {
    const rect = canvas.getBoundingClientRect();
    const frac = (ev.clientX - rect.left) / rect.width;
    const i = Math.min(order.length - 1, Math.max(0, Math.floor(frac * order.length)));
    state.selectedFilter[layerEntry.id] = order[i];
    delete state.selectedKernel[layerEntry.id];
    await renderLayerBody(body, layerEntry);
  };

  const filterList = document.createElement("div");
  filterList.className = "muted";
  filterList.style.marginTop = "4px";
  filterList.textContent = "click a bar to drill into that filter's kernels";
  body.appendChild(filterList);

  if (state.selectedFilter[layerEntry.id] !== undefined && levels.kernel) {
    try {
      await renderKernelLevel(body, layerEntry, state.selectedFilter[layerEntry.id]);
    } catch (e) {
      const err = document.createElement("div");
      err.className = "muted";
      err.textContent = `failed to load kernel data: ${e}`;
      body.appendChild(err);
    }
  }
}

async function renderKernelLevel(body, layerEntry, filterIdx) {
  const breadcrumb = document.createElement("div");
  breadcrumb.className = "breadcrumb";
  const backBtn = document.createElement("button");
  backBtn.textContent = "← back to filters";
  backBtn.onclick = async () => {
    delete state.selectedFilter[layerEntry.id];
    delete state.selectedKernel[layerEntry.id];
    await renderLayerBody(body, layerEntry);
  };
  breadcrumb.textContent = `filter ${filterIdx} `;
  breadcrumb.prepend(backBtn);
  body.appendChild(breadcrumb);

  const metric = metricForLevel(layerEntry, "kernel");
  const kernelRow = await getLayerSlice(layerEntry.id, "kernel", metric, state.selectedOutput);
  // kernelRow.data shape [F, C]; take row `filterIdx` -> [C]
  const channelScores = kernelRow.data[filterIdx] || [];
  const groups = layerEntry.groups || 1;
  const inCh = layerEntry.in_channels || channelScores.length * groups;
  const outCh = layerEntry.out_channels || 1;
  const groupSize = Math.max(1, Math.floor(outCh / groups));
  const groupIdx = Math.floor(filterIdx / groupSize);
  const channelsPerGroup = channelScores.length;

  const grid = document.createElement("div");
  const maxAbs = Math.max(...channelScores.map((v) => Math.abs(v)), 1e-12);
  channelScores.forEach((v, c) => {
    const realChannel = groupIdx * channelsPerGroup + c;
    const cell = document.createElement("div");
    cell.className = "grid-cell" + (state.selectedKernel[layerEntry.id] === c ? " selected" : "");
    const alpha = Math.abs(v) / maxAbs;
    const color = v < 0 ? getCss("--neg") : getCss("--accent");
    cell.style.width = "22px";
    cell.style.height = "22px";
    cell.style.background = color;
    cell.style.opacity = Math.max(0.15, alpha);
    cell.title = `real input channel ${realChannel} (local ${c}): ${v.toFixed(5)}`;
    cell.onclick = async () => {
      state.selectedKernel[layerEntry.id] = c;
      await renderLayerBody(body, layerEntry);
    };
    grid.appendChild(cell);
  });
  body.appendChild(grid);

  if (state.selectedKernel[layerEntry.id] !== undefined && layerEntry.levels.weight) {
    try {
      await renderWeightLevel(body, layerEntry, filterIdx, state.selectedKernel[layerEntry.id]);
    } catch (e) {
      const err = document.createElement("div");
      err.className = "muted";
      err.textContent = `failed to load weight data: ${e}`;
      body.appendChild(err);
    }
  }
}

async function renderWeightLevel(body, layerEntry, filterIdx, channelIdx) {
  const metric = metricForLevel(layerEntry, "weight");
  const [weightRow, valuesRow, rawWeight] = await Promise.all([
    getLayerSlice(layerEntry.id, "weight", metric, state.selectedOutput),
    getLayerSlice(layerEntry.id, "weight", "mean_s", state.selectedOutput),
    getRawArray(layerEntry.id, "weight"),
  ]);
  // shapes: [F, C, *k]
  const kernelImportance = weightRow.data[filterIdx][channelIdx]; // scalar, 1D array, or 2D array
  const kernelSigned = valuesRow.data[filterIdx][channelIdx];
  let kernelRawWeight = rawWeight.data[filterIdx][channelIdx];

  let weightLabel = "weight value (raw)";
  if (state.weightNormalize === "max100") {
    const layerMax = await getLayerWeightMax(layerEntry.id);
    kernelRawWeight = scaleNested(kernelRawWeight, 100 / layerMax);
    weightLabel = "weight value (×100 / max|w| for this layer)";
  }

  const wrap = document.createElement("div");
  wrap.style.marginTop = "8px";
  wrap.innerHTML = `<div class="muted">weight-level (channel ${channelIdx})</div>`;
  wrap.appendChild(renderHeatmap(kernelRawWeight, weightLabel));
  wrap.appendChild(renderHeatmap(kernelImportance, `importance (${metric})`));
  wrap.appendChild(renderHeatmap(kernelSigned, "signed importance (mean s = w·mean grad)"));
  body.appendChild(wrap);
}

function renderHeatmap(values, label) {
  const container = document.createElement("div");
  container.style.display = "inline-block";
  container.style.marginRight = "16px";
  container.style.verticalAlign = "top";
  const title = document.createElement("div");
  title.className = "muted";
  title.textContent = label;
  container.appendChild(title);

  let matrix;
  if (typeof values === "number") matrix = [[values]];
  else if (Array.isArray(values[0])) matrix = values;
  else matrix = [values]; // 1D kernel -> single row

  const maxAbs = Math.max(...matrix.flat().map((v) => Math.abs(v)), 1e-12);
  const table = document.createElement("table");
  table.className = "value-table";
  matrix.forEach((row) => {
    const tr = document.createElement("tr");
    row.forEach((v) => {
      const td = document.createElement("td");
      const alpha = Math.abs(v) / maxAbs;
      td.style.background = v < 0 ? `rgba(224,97,107,${alpha})` : `rgba(94,161,255,${alpha})`;
      td.textContent = v.toFixed(3);
      tr.appendChild(td);
    });
    table.appendChild(tr);
  });
  container.appendChild(table);
  return container;
}

function wireControls() {
  document.getElementById("metric-select").onchange = async (e) => {
    state.metric = e.target.value;
    state.cache.clear();
    state._globalMax = null;
    await renderLayers();
  };
  document.getElementById("normalize-select").onchange = async (e) => {
    state.normalize = e.target.value;
    await renderLayers();
  };
  document.getElementById("weight-normalize-select").onchange = async (e) => {
    state.weightNormalize = e.target.value;
    await renderLayers();
  };
  document.getElementById("sort-select").onchange = async (e) => {
    state.sort = e.target.value;
    await renderLayers();
  };
  document.getElementById("layer-search").oninput = async (e) => {
    state.search = e.target.value;
    await renderLayers();
  };
  const topk = document.getElementById("topk-slider");
  topk.oninput = async (e) => {
    state.topkPct = parseInt(e.target.value, 10);
    document.getElementById("topk-label").textContent = `${state.topkPct}%`;
    await renderLayers();
  };
  document.getElementById("export-csv").onclick = exportCurrentViewCSV;
}

async function exportCurrentViewCSV() {
  const rows = [["layer", "filter", "metric", "output", "value"]];
  const metricGlobal = state.metric;
  for (const layerEntry of state.manifest.layers) {
    if (!layerEntry.levels.filter) continue;
    const metric = metricGlobal === "auto" ? effectiveMetric(layerEntry) : metricGlobal;
    if (!layerEntry.levels.filter[metric]) continue;
    const row = await getLayerSlice(layerEntry.id, "filter", metric, state.selectedOutput);
    row.data.forEach((v, i) => {
      rows.push([layerEntry.id, i, metric, state.manifest.output_features.names[state.selectedOutput], v]);
    });
  }
  const csv = rows.map((r) => r.join(",")).join("\n");
  const blob = new Blob([csv], { type: "text/csv" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = "importance_export.csv";
  a.click();
}

init().catch((e) => {
  document.body.innerHTML = `<pre style="color:#e0616b;padding:20px">Failed to load viewer: ${e}\n${e.stack || ""}</pre>`;
});
