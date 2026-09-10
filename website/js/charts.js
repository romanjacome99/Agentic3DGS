/* Small dependency-free SVG chart helpers (line charts, categorical strips, small multiples). */

export const COLORS = {
  agent: '#1f6fb2', base: '#d97b1e', gt: '#444',
  backend: { '3dgs': '#0072B2', 'fastergs': '#E69F00', 'dash': '#009E73', 'legs': '#CC79A7' },
  densify_mode: { off: '#dcdcdc', conservative: '#a6cee3', default: '#1f78b4', aggressive: '#08306b' },
  prune_mode: { off: '#dcdcdc', opacity_only: '#b2df8a', opacity_and_size: '#33a02c' },
  opacity_reset: { no_reset: '#dcdcdc', reset_if_plateau: '#fdbf6f', force_reset: '#e31a1c' },
  block_steps: { 50: '#dadaeb', 100: '#9e9ac8', 200: '#54278f' },
  densification_interval: { 50: '#fee0d2', 100: '#fc9272', 200: '#de2d26', 400: '#a50f15' },
  stop_intent: { continue: '#dcdcdc', stop: '#7b3294' },
};

export const ACTION_LABEL = {
  block_steps: 'block length (iters)', densify_mode: 'densify mode', densification_interval: 'densify interval (iters)',
  prune_mode: 'prune mode', opacity_reset: 'opacity reset', stop: 'stop', stop_intent: 'stop intent',
  densify_threshold_mult: 'densify grad-threshold ×', prune_opacity_threshold: 'prune opacity threshold',
  position_lr_mult: 'position lr ×', feature_lr_mult: 'SH / colour lr ×', opacity_lr_mult: 'opacity lr ×',
  scaling_lr_mult: 'scale lr ×', rotation_lr_mult: 'rotation lr ×',
};

const NS = 'http://www.w3.org/2000/svg';
export function el(tag, attrs = {}, children = []) {
  const e = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v !== undefined && v !== null) e.setAttribute(k, v);
  for (const c of children) e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
  return e;
}

export function fmtK(n) {
  if (n == null || isNaN(n)) return '–';
  if (Math.abs(n) >= 1e6) return (n / 1e6).toFixed(2) + 'M';
  if (Math.abs(n) >= 1e3) return Math.round(n / 1e3) + 'k';
  return String(Math.round(n));
}
export function fmtT(s) {
  if (s == null || isNaN(s)) return '–';
  if (s >= 600) return (s / 60).toFixed(1) + ' min';
  return (s < 10 ? s.toFixed(1) : Math.round(s)) + ' s';
}

// ---------------------------------------------------------------- tooltip
let tip = null;
export function showTip(html, x, y) {
  if (!tip) { tip = document.createElement('div'); tip.className = 'tip'; document.body.appendChild(tip); }
  tip.innerHTML = html; tip.hidden = false;
  const pad = 14;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  let left = x + pad, top = y + pad;
  if (left + w > window.innerWidth - 8) left = x - w - pad;
  if (top + h > window.innerHeight - 8) top = y - h - pad;
  tip.style.left = left + 'px'; tip.style.top = top + 'px';
}
export function hideTip() { if (tip) tip.hidden = true; }

// ---------------------------------------------------------------- scales
function niceTicks(lo, hi, n = 5) {
  if (!(hi > lo)) return [lo];
  const span = hi - lo;
  const step0 = Math.pow(10, Math.floor(Math.log10(span / n)));
  const err = span / n / step0;
  const step = err >= 7.5 ? 10 * step0 : err >= 3.5 ? 5 * step0 : err >= 1.5 ? 2 * step0 : step0;
  const out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(+v.toFixed(10));
  return out;
}
function logTicks(lo, hi) {
  const out = [];
  const a = Math.floor(Math.log10(Math.max(lo, 1e-9))), b = Math.ceil(Math.log10(hi));
  for (let e = a; e <= b; e++) for (const m of [1, 2, 5]) { const v = m * Math.pow(10, e); if (v >= lo && v <= hi) out.push(v); }
  return out;
}

/**
 * Line chart. opts: { series:[{x,y,color,label,dash,width,points,area}], xLog, yDomain, xDomain, xLabel, yLabel,
 *   height, hlines:[{y,label,color}], vlines:[{x,label,color}], marker:x, xFormat, yFormat, legend }
 */
export function lineChart(container, opts) {
  container.innerHTML = '';
  const W = Math.max(240, container.clientWidth || 600);
  const H = opts.height || 220;
  const m = Object.assign({ l: 52, r: 14, t: 12, b: 34 }, opts.margin || {});
  const iw = W - m.l - m.r, ih = H - m.t - m.b;
  const xs = [], ys = [];
  for (const s of opts.series) for (let i = 0; i < s.x.length; i++) { if (s.x[i] != null && s.y[i] != null && isFinite(s.x[i]) && isFinite(s.y[i])) { xs.push(s.x[i]); ys.push(s.y[i]); } }
  let x0 = opts.xDomain ? opts.xDomain[0] : Math.min(...xs), x1 = opts.xDomain ? opts.xDomain[1] : Math.max(...xs);
  let y0 = opts.yDomain ? opts.yDomain[0] : Math.min(...ys), y1 = opts.yDomain ? opts.yDomain[1] : Math.max(...ys);
  if (opts.hlines) for (const h of opts.hlines) { y0 = Math.min(y0, h.y); y1 = Math.max(y1, h.y); }
  if (!(y1 > y0)) { y0 -= 1; y1 += 1; }
  if (!opts.yDomain) { const p = (y1 - y0) * 0.06; y0 -= p; y1 += p; }
  if (opts.xLog) { x0 = Math.max(x0, opts.xMin || 1); }
  if (!(x1 > x0)) x1 = x0 + 1;
  const sx = opts.xLog ? (v) => m.l + (Math.log(Math.max(v, x0)) - Math.log(x0)) / (Math.log(x1) - Math.log(x0)) * iw
                       : (v) => m.l + (v - x0) / (x1 - x0) * iw;
  const sy = (v) => m.t + (1 - (v - y0) / (y1 - y0)) * ih;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: 'chart' });
  // grid + axes
  const xt = opts.xLog ? logTicks(x0, x1) : niceTicks(x0, x1, Math.max(3, Math.floor(iw / 90)));
  const yt = niceTicks(y0, y1, Math.max(3, Math.floor(ih / 40)));
  for (const v of yt) {
    svg.appendChild(el('line', { x1: m.l, x2: W - m.r, y1: sy(v), y2: sy(v), class: 'grid' }));
    svg.appendChild(el('text', { x: m.l - 6, y: sy(v) + 3.5, class: 'tick', 'text-anchor': 'end' }, [opts.yFormat ? opts.yFormat(v) : String(v)]));
  }
  for (const v of xt) {
    svg.appendChild(el('line', { x1: sx(v), x2: sx(v), y1: m.t, y2: m.t + ih, class: 'grid' }));
    svg.appendChild(el('text', { x: sx(v), y: m.t + ih + 15, class: 'tick', 'text-anchor': 'middle' }, [opts.xFormat ? opts.xFormat(v) : String(v)]));
  }
  svg.appendChild(el('line', { x1: m.l, x2: W - m.r, y1: m.t + ih, y2: m.t + ih, class: 'axis' }));
  svg.appendChild(el('line', { x1: m.l, x2: m.l, y1: m.t, y2: m.t + ih, class: 'axis' }));
  if (opts.xLabel) svg.appendChild(el('text', { x: m.l + iw / 2, y: H - 4, class: 'axis-label', 'text-anchor': 'middle' }, [opts.xLabel]));
  if (opts.yLabel) svg.appendChild(el('text', { x: 12, y: m.t + ih / 2, class: 'axis-label', 'text-anchor': 'middle', transform: `rotate(-90 12 ${m.t + ih / 2})` }, [opts.yLabel]));
  // reference lines
  for (const h of opts.hlines || []) {
    svg.appendChild(el('line', { x1: m.l, x2: W - m.r, y1: sy(h.y), y2: sy(h.y), class: 'ref', stroke: h.color }));
    if (h.label) svg.appendChild(el('text', { x: W - m.r - 2, y: sy(h.y) - 3, class: 'ref-label', 'text-anchor': 'end', fill: h.color }, [h.label]));
  }
  for (const v of opts.vlines || []) {
    if (v.x < x0 || v.x > x1) continue;
    svg.appendChild(el('line', { x1: sx(v.x), x2: sx(v.x), y1: m.t, y2: m.t + ih, class: 'ref', stroke: v.color }));
    if (v.label) svg.appendChild(el('text', { x: sx(v.x) + 3, y: m.t + 10, class: 'ref-label', fill: v.color }, [v.label]));
  }
  // series
  const clip = `clip${Math.random().toString(36).slice(2, 8)}`;
  svg.appendChild(el('defs', {}, [el('clipPath', { id: clip }, [el('rect', { x: m.l, y: m.t - 2, width: iw, height: ih + 4 })])]));
  const g = el('g', { 'clip-path': `url(#${clip})` });
  svg.appendChild(g);
  for (const s of opts.series) {
    let d = '', started = false;
    for (let i = 0; i < s.x.length; i++) {
      const vx = s.x[i], vy = s.y[i];
      if (vx == null || vy == null || !isFinite(vx) || !isFinite(vy) || (opts.xLog && vx <= 0)) { started = false; continue; }
      d += (started ? 'L' : 'M') + sx(vx).toFixed(1) + ' ' + sy(vy).toFixed(1); started = true;
    }
    if (s.area) {
      const xsA = s.x.filter((v, i) => v != null && s.y[i] != null);
      if (xsA.length) g.appendChild(el('path', { d: d + `L${sx(xsA[xsA.length - 1])} ${sy(y0)}L${sx(xsA[0])} ${sy(y0)}Z`, fill: s.color, opacity: 0.08 }));
    }
    if (s.step) { /* not used */ }
    g.appendChild(el('path', { d, fill: 'none', stroke: s.color, 'stroke-width': s.width || 2, 'stroke-dasharray': s.dash || null, 'stroke-linejoin': 'round' }));
    if (s.points) for (let i = 0; i < s.x.length; i++) if (s.x[i] != null && s.y[i] != null && isFinite(s.y[i]))
      g.appendChild(el('circle', { cx: sx(s.x[i]), cy: sy(s.y[i]), r: 3.2, fill: s.color, stroke: '#fff', 'stroke-width': 1 }));
  }
  // marker (current time)
  let markerLine = null;
  if (opts.marker != null) {
    markerLine = el('line', { x1: sx(opts.marker), x2: sx(opts.marker), y1: m.t, y2: m.t + ih, class: 'marker' });
    svg.appendChild(markerLine);
  }
  // legend
  if (opts.legend !== false) {
    let lx = m.l + 6;
    const ly = m.t + 10;
    for (const s of opts.series) {
      if (!s.label) continue;
      svg.appendChild(el('line', { x1: lx, x2: lx + 16, y1: ly, y2: ly, stroke: s.color, 'stroke-width': 2.5, 'stroke-dasharray': s.dash || null }));
      const t = el('text', { x: lx + 20, y: ly + 4, class: 'legend' }, [s.label]);
      svg.appendChild(t);
      lx += 20 + s.label.length * 6.4 + 14;
    }
  }
  // hover
  if (opts.onHover || opts.hover !== false) {
    const hoverLine = el('line', { x1: 0, x2: 0, y1: m.t, y2: m.t + ih, class: 'hover-line', visibility: 'hidden' });
    svg.appendChild(hoverLine);
    const rect = el('rect', { x: m.l, y: m.t, width: iw, height: ih, fill: 'transparent' });
    svg.appendChild(rect);
    const inv = opts.xLog ? (px) => Math.exp(Math.log(x0) + (px - m.l) / iw * (Math.log(x1) - Math.log(x0))) : (px) => x0 + (px - m.l) / iw * (x1 - x0);
    rect.addEventListener('mousemove', (ev) => {
      const b = svg.getBoundingClientRect();
      const px = (ev.clientX - b.left) * (W / b.width);
      const xv = inv(px);
      hoverLine.setAttribute('x1', px); hoverLine.setAttribute('x2', px); hoverLine.setAttribute('visibility', 'visible');
      const rows = [];
      for (const s of opts.series) {
        if (!s.label) continue;
        // nearest sample in x
        let bi = -1, bd = Infinity;
        for (let i = 0; i < s.x.length; i++) { if (s.x[i] == null || s.y[i] == null) continue; const dd = Math.abs((opts.xLog ? Math.log(Math.max(s.x[i], 1e-9)) : s.x[i]) - (opts.xLog ? Math.log(xv) : xv)); if (dd < bd) { bd = dd; bi = i; } }
        if (bi >= 0) rows.push(`<span class="sw" style="background:${s.color}"></span>${s.label}: <b>${opts.yFormat ? opts.yFormat(s.y[bi]) : s.y[bi]}</b> <span class="dim">@ ${opts.xFormat ? opts.xFormat(s.x[bi]) : s.x[bi]}</span>`);
      }
      showTip(rows.join('<br>'), ev.clientX, ev.clientY);
      if (opts.onHover) opts.onHover(xv);
    });
    rect.addEventListener('mouseleave', () => { hoverLine.setAttribute('visibility', 'hidden'); hideTip(); });
  }
  container.appendChild(svg);
  return { svg, sx, sy, setMarker(x) { if (markerLine) { markerLine.setAttribute('x1', sx(x)); markerLine.setAttribute('x2', sx(x)); } } };
}

/**
 * Categorical strips over time. opts: { rows:[{key,label,colorMap}], blocks:[{t, ...fields}], tMax, tMin, marker, height, xLog }
 * Each block row covers (t_prev, t]. Hover shows the full decoded action of the block (opts.describe(block)).
 */
export function stripChart(container, opts) {
  container.innerHTML = '';
  const W = Math.max(240, container.clientWidth || 600);
  const rowH = opts.rowH || 16, gap = 3;
  const m = { l: opts.labelWidth || 128, r: 14, t: 6, b: opts.axis === false ? 6 : 22 };
  const H = m.t + opts.rows.length * (rowH + gap) + m.b;
  const blocks = opts.blocks;
  const t0 = opts.tMin != null ? opts.tMin : 0;
  const tMax = opts.tMax || (blocks.length ? blocks[blocks.length - 1].t : 1);
  const iw = W - m.l - m.r;
  const xLog = !!opts.xLog;
  const lx0 = Math.max(t0, xLog ? (opts.xMin || 1) : t0);
  const sx = xLog ? (v) => m.l + (Math.log(Math.max(v, lx0)) - Math.log(lx0)) / (Math.log(tMax) - Math.log(lx0)) * iw
                  : (v) => m.l + (v - t0) / (tMax - t0) * iw;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, width: W, height: H, class: 'chart strips' });
  opts.rows.forEach((row, ri) => {
    const y = m.t + ri * (rowH + gap);
    svg.appendChild(el('text', { x: m.l - 8, y: y + rowH * 0.72, class: 'strip-label', 'text-anchor': 'end' }, [row.label]));
    svg.appendChild(el('rect', { x: m.l, y, width: iw, height: rowH, fill: '#f3f3f3' }));
    let prevT = t0;
    for (const b of blocks) {
      const a = Math.max(prevT, t0), z = b.t;
      prevT = b.t;
      if (z <= a) continue;
      const v = b[row.key];
      const color = row.colorMap ? (row.colorMap[v] || '#bbb') : (row.color || '#888');
      const x = sx(a), w = Math.max(0.6, sx(z) - x);
      const r = el('rect', { x, y, width: w, height: rowH, fill: color });
      r.addEventListener('mousemove', (ev) => showTip(opts.describe ? opts.describe(b, row) : `${row.label}: ${v}`, ev.clientX, ev.clientY));
      r.addEventListener('mouseleave', hideTip);
      svg.appendChild(r);
    }
  });
  if (opts.axis !== false) {
    const yA = m.t + opts.rows.length * (rowH + gap) + 2;
    const ticks = xLog ? logTicks(lx0, tMax) : niceTicks(t0, tMax, Math.max(3, Math.floor(iw / 90)));
    for (const v of ticks) {
      svg.appendChild(el('line', { x1: sx(v), x2: sx(v), y1: yA, y2: yA + 4, class: 'axis' }));
      svg.appendChild(el('text', { x: sx(v), y: yA + 14, class: 'tick', 'text-anchor': 'middle' }, [fmtT(v)]));
    }
    svg.appendChild(el('line', { x1: m.l, x2: m.l + iw, y1: yA, y2: yA, class: 'axis' }));
  }
  let marker = null;
  if (opts.marker != null) {
    marker = el('line', { x1: sx(opts.marker), x2: sx(opts.marker), y1: m.t - 2, y2: m.t + opts.rows.length * (rowH + gap), class: 'marker' });
    svg.appendChild(marker);
  }
  container.appendChild(svg);
  return { svg, setMarker(x) { if (marker) { marker.setAttribute('x1', sx(x)); marker.setAttribute('x2', sx(x)); } } };
}

/** Legend for a categorical color map. */
export function legend(container, colorMap, order) {
  const keys = order || Object.keys(colorMap);
  const d = document.createElement('div');
  d.className = 'cat-legend';
  for (const k of keys) {
    const s = document.createElement('span');
    s.innerHTML = `<span class="sw" style="background:${colorMap[k]}"></span>${k}`;
    d.appendChild(s);
  }
  container.appendChild(d);
}

/** Small multiples of the continuous multipliers vs time (one panel each). */
export function multiplesChart(container, opts) {
  container.innerHTML = '';
  const fields = opts.fields;
  const W = Math.max(240, container.clientWidth || 600);
  const cols = W > 900 ? 4 : W > 620 ? 3 : W > 400 ? 2 : 1;
  const grid = document.createElement('div');
  grid.className = 'multiples';
  grid.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
  container.appendChild(grid);
  for (const f of fields) {
    const cell = document.createElement('div');
    cell.className = 'multiple';
    const title = document.createElement('div');
    title.className = 'multiple-title';
    title.textContent = f.label;
    cell.appendChild(title);
    const ch = document.createElement('div');
    cell.appendChild(ch);
    grid.appendChild(cell);
    const x = opts.blocks.map(b => b.t), y = opts.blocks.map(b => b[f.key]);
    const yd = f.log ? [Math.log2(f.lo), Math.log2(f.hi)] : [f.lo, f.hi];
    const yv = f.log ? y.map(v => v == null ? null : Math.log2(v)) : y;
    const def = f.log ? Math.log2(f.def) : f.def;
    lineChart(ch, {
      height: 110, margin: { l: 40, r: 8, t: 8, b: 20 },
      series: [{ x, y: yv, color: opts.color || COLORS.agent, width: 1.6, label: f.short || f.label }],
      xLog: opts.xLog, xMin: opts.xMin, xDomain: opts.tMax ? [opts.xLog ? (opts.xMin || 1) : 0, opts.tMax] : undefined,
      yDomain: yd, hlines: [{ y: def, label: 'default', color: '#999' }],
      xFormat: fmtT, yFormat: f.log ? (v) => (Math.pow(2, v) >= 1 ? Math.pow(2, v).toFixed(1) : Math.pow(2, v).toFixed(2)) + '×' : (v) => v.toFixed(3),
      legend: false,
    });
  }
}
