// Layered DAG layout (Sugiyama-style, the same family Netron's dagre uses),
// written out here so the viewer stays dependency-free and works offline.
//
//   1. rank     longest path from the sources (inputs at rank 0)
//   2. dummies  edges spanning >1 rank get a chain of thin dummy nodes, so
//               skip connections route *around* the nodes they jump over
//   3. order    barycenter sweeps down/up, keeping the ordering with the
//               fewest crossings
//   4. place    per rank, positions along the order axis minimise the squared
//               distance to the neighbours' mean subject to "no overlap" --
//               solved exactly with pool-adjacent-violators, alternating
//               down/up passes, so chains come out straight
//
// Works in abstract (rank, order) coordinates; orient "TB" maps rank->y,
// "LR" maps rank->x.
"use strict";

export function layoutGraph(nodes, edges, opts = {}) {
  const orient = opts.orient || "TB";
  const rankSep = opts.rankSep ?? (orient === "TB" ? 34 : 56);
  const nodeSep = opts.nodeSep ?? 22;
  const dummySize = opts.dummySize ?? 6;
  const TB = orient === "TB";
  const rSize = (n) => (TB ? n.h : n.w); // extent along the rank axis
  const oSize = (n) => (TB ? n.w : n.h); // extent along the order axis

  // ---- graph structures (real + dummy nodes share one array) -----------
  const V = nodes.map((n, i) => ({ id: n.id, idx: i, w: n.w, h: n.h, dummy: false, preds: [], succs: [] }));
  const byId = new Map(V.map((v) => [v.id, v]));
  const E = [];
  for (const e of edges) {
    const s = byId.get(e.source), t = byId.get(e.target);
    if (!s || !t || s === t) continue;
    E.push({ s, t, src: e });
    s.succs.push(t);
    t.preds.push(s);
  }

  // ---- 1. ranks (Kahn topological order + longest path) ----------------
  const indeg = new Map(V.map((v) => [v, v.preds.length]));
  const topo = [];
  const queue = V.filter((v) => !v.preds.length);
  while (queue.length) {
    const v = queue.shift();
    topo.push(v);
    for (const s of v.succs) {
      indeg.set(s, indeg.get(s) - 1);
      if (indeg.get(s) === 0) queue.push(s);
    }
  }
  if (topo.length < V.length) for (const v of V) if (!topo.includes(v)) topo.push(v); // cycles: shouldn't happen
  for (const v of topo) v.rank = v.preds.length ? Math.max(...v.preds.map((p) => p.rank ?? 0)) + 1 : 0;
  // sources that only feed something deep down (constants, extra inputs) sit right above their consumer
  for (const v of V) if (!v.preds.length && v.succs.length) v.rank = Math.max(0, Math.min(...v.succs.map((s) => s.rank)) - 1);

  // ---- 2. dummy chains --------------------------------------------------
  const routes = []; // per input edge: list of vertices from s to t
  const all = V.slice();
  const linkPreds = new Map(all.map((v) => [v, []]));
  const linkSuccs = new Map(all.map((v) => [v, []]));
  const link = (a, b) => { linkSuccs.get(a).push(b); linkPreds.get(b).push(a); };
  for (const e of E) {
    const chain = [e.s];
    let prev = e.s;
    for (let r = e.s.rank + 1; r < e.t.rank; r++) {
      const d = { id: `__d${all.length}`, w: dummySize, h: dummySize, dummy: true, rank: r };
      all.push(d);
      linkPreds.set(d, []);
      linkSuccs.set(d, []);
      link(prev, d);
      chain.push(d);
      prev = d;
    }
    link(prev, e.t);
    chain.push(e.t);
    routes.push({ edge: e.src, chain });
  }

  // ---- 3. ordering -------------------------------------------------------
  const nRanks = Math.max(0, ...all.map((v) => v.rank)) + 1;
  const layers = Array.from({ length: nRanks }, () => []);
  // initial order: DFS visit order from the sources, which keeps chains together
  const seen = new Set();
  let counter = 0;
  const visit = (v) => {
    if (seen.has(v)) return;
    seen.add(v);
    v.order0 = counter++;
    for (const s of linkSuccs.get(v)) visit(s);
  };
  for (const v of topo) visit(v);
  for (const v of all) { if (!seen.has(v)) v.order0 = counter++; layers[v.rank].push(v); }
  for (const L of layers) { L.sort((a, b) => a.order0 - b.order0); L.forEach((v, i) => (v.pos = i)); }

  const crossings = () => {
    let c = 0;
    for (let r = 0; r + 1 < nRanks; r++) {
      const segs = [];
      for (const v of layers[r]) for (const s of linkSuccs.get(v)) segs.push([v.pos, s.pos]);
      for (let i = 0; i < segs.length; i++)
        for (let j = i + 1; j < segs.length; j++)
          if ((segs[i][0] - segs[j][0]) * (segs[i][1] - segs[j][1]) < 0) c++;
    }
    return c;
  };
  const snapshot = () => layers.map((L) => L.slice());
  let best = snapshot();
  let bestC = crossings();
  const sweeps = all.length > 1500 ? 4 : 16;
  for (let it = 0; it < sweeps && bestC > 0; it++) {
    const down = it % 2 === 0;
    const rs = down ? [...Array(nRanks).keys()].slice(1) : [...Array(nRanks).keys()].reverse().slice(1);
    for (const r of rs) {
      const L = layers[r];
      for (const v of L) {
        const nb = down ? linkPreds.get(v) : linkSuccs.get(v);
        v.bary = nb.length ? nb.reduce((a, u) => a + u.pos, 0) / nb.length : v.pos;
      }
      L.sort((a, b) => a.bary - b.bary || a.pos - b.pos);
      L.forEach((v, i) => (v.pos = i));
    }
    const c = crossings();
    if (c < bestC) { bestC = c; best = snapshot(); }
  }
  best.forEach((L, r) => { layers[r] = L; L.forEach((v, i) => (v.pos = i)); });

  // ---- 4. coordinates along the order axis -------------------------------
  const minGap = (a, b) => (oSize(a) + oSize(b)) / 2 + (a.dummy && b.dummy ? 6 : nodeSep);
  for (const L of layers) {
    let x = 0;
    L.forEach((v, i) => { if (i) x += minGap(L[i - 1], v); v.o = x; });
    const mid = x / 2;
    for (const v of L) v.o -= mid;
  }
  // Place layer L so that sum wt_i (o_i - want_i)^2 is minimal with o_{i+1}-o_i >= gap_i:
  // substitute y_i = o_i - c_i (c = prefix sum of gaps) -> weighted isotonic
  // regression, solved exactly by pool-adjacent-violators. Dummies get a small
  // weight, so when a real node and a skip edge compete for the same spot the
  // skip edge is the one that moves aside and the main path stays straight.
  const wt = (v) => (v.dummy ? 0.15 : 1);
  const place = (L, want) => {
    const c = [0];
    for (let i = 1; i < L.length; i++) c.push(c[i - 1] + minGap(L[i - 1], L[i]));
    const blocks = []; // {sw, w, n}
    for (let i = 0; i < L.length; i++) {
      blocks.push({ sw: wt(L[i]) * (want[i] - c[i]), w: wt(L[i]), n: 1 });
      while (blocks.length > 1) {
        const b = blocks[blocks.length - 1], a = blocks[blocks.length - 2];
        if (a.sw / a.w <= b.sw / b.w) break;
        a.sw += b.sw; a.w += b.w; a.n += b.n; blocks.pop();
      }
    }
    let i = 0;
    for (const b of blocks) for (let k = 0; k < b.n; k++, i++) L[i].o = b.sw / b.w + c[i];
  };
  // weighted mean of neighbour positions; real->real links dominate, dummy
  // chains pull hard on each other so long skip edges stay straight too
  const pull = (v, nb) => {
    if (!nb.length) return v.o;
    let s = 0, wsum = 0;
    for (const u of nb) {
      const w = u.dummy && v.dummy ? 4 : u.dummy || v.dummy ? 0.2 : 1;
      s += u.o * w; wsum += w;
    }
    return s / wsum;
  };
  const passes = all.length > 1500 ? 4 : 12;
  for (let it = 0; it < passes; it++) {
    const down = it % 2 === 0;
    const both = it > 1 && it < passes - 3; // last passes: one-sided, to settle chains
    const rs = down ? [...Array(nRanks).keys()] : [...Array(nRanks).keys()].reverse();
    for (const r of rs) {
      const L = layers[r];
      place(L, L.map((v) => {
        const nb = [...(down ? linkPreds.get(v) : linkSuccs.get(v))];
        if (both) nb.push(...(down ? linkSuccs.get(v) : linkPreds.get(v)));
        return pull(v, nb);
      }));
    }
  }

  // ---- rank axis ---------------------------------------------------------
  let cursor = 0;
  for (const L of layers) {
    const ext = Math.max(dummySize, ...L.map(rSize));
    for (const v of L) v.r = cursor + ext / 2;
    cursor += ext + rankSep;
  }

  // ---- output -------------------------------------------------------------
  const minO = Math.min(...all.map((v) => v.o - oSize(v) / 2));
  const margin = 30;
  const toXY = (v) => (TB ? { x: v.o - minO + margin, y: v.r + margin } : { x: v.r + margin, y: v.o - minO + margin });
  const positions = new Map();
  for (const v of V) positions.set(v.id, { ...toXY(v), w: v.w, h: v.h });
  const edgeRoutes = routes.map(({ edge, chain }) => ({ edge, points: chain.map((v) => toXY(v)), source: chain[0].id, target: chain[chain.length - 1].id }));
  let width = 0, height = 0;
  for (const p of positions.values()) { width = Math.max(width, p.x + p.w / 2); height = Math.max(height, p.y + p.h / 2); }
  return { positions, edgeRoutes, width: width + margin, height: height + margin, orient };
}
