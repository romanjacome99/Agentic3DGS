/* Depth sort worker. Holds several splat sets (by id) and returns front-to-back index orders.
 * Messages in:  {type:'set', id, positions: Float32Array}   {type:'remove', id}
 *               {type:'sort', id, view: Float32Array(16) column-major, gen}
 * Messages out: {type:'sorted', id, indices: Uint32Array, gen}
 */
const sets = new Map();
const lastKey = new Map();

function sort(id, view, gen) {
  const s = sets.get(id);
  if (!s) return;
  const n = s.count;
  const p = s.positions;
  // camera-space depth = row 2 of the view matrix (column-major storage)
  const vz0 = view[2], vz1 = view[6], vz2 = view[10], vz3 = view[14];
  const key = `${vz0.toFixed(5)},${vz1.toFixed(5)},${vz2.toFixed(5)},${vz3.toFixed(3)}`;
  if (lastKey.get(id) === key && s.indices) {
    self.postMessage({ type: 'sorted', id, indices: s.indices.slice(), gen });
    return;
  }
  lastKey.set(id, key);
  if (!s.depth || s.depth.length !== n) { s.depth = new Float32Array(n); s.keys = new Uint32Array(n); }
  const depth = s.depth;
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < n; i++) {
    const d = vz0 * p[3 * i] + vz1 * p[3 * i + 1] + vz2 * p[3 * i + 2] + vz3;
    depth[i] = d;
    if (d < mn) mn = d; if (d > mx) mx = d;
  }
  // 16-bit counting sort on depth (near first)
  const B = 65536;
  const counts = new Uint32Array(B);
  const scale = mx > mn ? (B - 1) / (mx - mn) : 0;
  const keys = s.keys;
  for (let i = 0; i < n; i++) { const k = ((depth[i] - mn) * scale) | 0; keys[i] = k; counts[k]++; }
  let acc = 0;
  for (let b = 0; b < B; b++) { const c = counts[b]; counts[b] = acc; acc += c; }
  const out = new Uint32Array(n);
  for (let i = 0; i < n; i++) out[counts[keys[i]]++] = i;
  s.indices = out;
  self.postMessage({ type: 'sorted', id, indices: out.slice(), gen });
}

self.onmessage = (e) => {
  const m = e.data;
  if (m.type === 'set') {
    sets.set(m.id, { positions: m.positions, count: m.positions.length / 3 });
    lastKey.delete(m.id);
  } else if (m.type === 'remove') {
    sets.delete(m.id); lastKey.delete(m.id);
  } else if (m.type === 'sort') {
    sort(m.id, m.view, m.gen);
  }
};
