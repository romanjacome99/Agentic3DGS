/* AGSP decoder: quantized Gaussian snapshots produced by tools/build_site_data.py.
 *
 * header 48 B: 'AGSP', u32 version(2), u32 count, u32 true_count, u32 flags, f32 lo[3], f32 hi[3], u32 n_core
 * pos   u16 x 3*n_core (quantized in [lo,hi]) then f32 x 3*(count-n_core) raw
 * scale u8  x 3*count  (log-scale, ls = q/255*16-12)
 * rot   u8  x 4*count  (w,x,y,z, q = v/255*2-1)
 * rgba  u8  x 4*count
 *
 * decodeAGSP() returns { count, trueCount, subsampled, positions: Float32Array(3N), tex: Float32Array(12N) }
 * where tex holds three RGBA32F texels per splat:
 *   [x, y, z, alpha]  [c00, c01, c02, c11]  [c12, c22, rgbPacked, 0]
 * with c** the 3-D covariance and rgbPacked = r*65536 + g*256 + b (exact in f32).
 */
export function decodeAGSP(buffer) {
  const dv = new DataView(buffer);
  const magic = String.fromCharCode(dv.getUint8(0), dv.getUint8(1), dv.getUint8(2), dv.getUint8(3));
  if (magic !== 'AGSP') throw new Error('not an AGSP file');
  const version = dv.getUint32(4, true);
  const n = dv.getUint32(8, true);
  const trueCount = dv.getUint32(12, true);
  const flags = dv.getUint32(16, true);
  const lo = [dv.getFloat32(20, true), dv.getFloat32(24, true), dv.getFloat32(28, true)];
  const hi = [dv.getFloat32(32, true), dv.getFloat32(36, true), dv.getFloat32(40, true)];
  const nCore = version >= 2 ? dv.getUint32(44, true) : n;
  const nFar = n - nCore;

  let off = 48;
  const posQ = new Uint16Array(buffer, off, 3 * nCore); off += 6 * nCore;
  // f32 block may be unaligned (6*nCore may not be a multiple of 4) -> copy through DataView when needed
  let posFar;
  if (nFar > 0) {
    if (off % 4 === 0) posFar = new Float32Array(buffer, off, 3 * nFar);
    else { posFar = new Float32Array(3 * nFar); for (let i = 0; i < 3 * nFar; i++) posFar[i] = dv.getFloat32(off + 4 * i, true); }
    off += 12 * nFar;
  }
  const scaleQ = new Uint8Array(buffer, off, 3 * n); off += 3 * n;
  const rotQ = new Uint8Array(buffer, off, 4 * n); off += 4 * n;
  const rgba = new Uint8Array(buffer, off, 4 * n); off += 4 * n;

  const positions = new Float32Array(3 * n);
  const tex = new Float32Array(12 * n);
  const sx = (hi[0] - lo[0]) / 65535, sy = (hi[1] - lo[1]) / 65535, sz = (hi[2] - lo[2]) / 65535;
  const scaleLUT = new Float32Array(256);
  for (let i = 0; i < 256; i++) scaleLUT[i] = Math.exp(i / 255 * 16 - 12);

  for (let i = 0; i < n; i++) {
    let x, y, z;
    if (i < nCore) {
      x = lo[0] + posQ[3 * i] * sx; y = lo[1] + posQ[3 * i + 1] * sy; z = lo[2] + posQ[3 * i + 2] * sz;
    } else {
      const j = 3 * (i - nCore); x = posFar[j]; y = posFar[j + 1]; z = posFar[j + 2];
    }
    positions[3 * i] = x; positions[3 * i + 1] = y; positions[3 * i + 2] = z;

    const s0 = scaleLUT[scaleQ[3 * i]], s1 = scaleLUT[scaleQ[3 * i + 1]], s2 = scaleLUT[scaleQ[3 * i + 2]];
    let qw = rotQ[4 * i] / 255 * 2 - 1, qx = rotQ[4 * i + 1] / 255 * 2 - 1, qy = rotQ[4 * i + 2] / 255 * 2 - 1, qz = rotQ[4 * i + 3] / 255 * 2 - 1;
    const qn = Math.hypot(qw, qx, qy, qz) || 1; qw /= qn; qx /= qn; qy /= qn; qz /= qn;
    // rotation matrix (row-major) from unit quaternion (w,x,y,z), same convention as 3DGS build_rotation
    const r00 = 1 - 2 * (qy * qy + qz * qz), r01 = 2 * (qx * qy - qw * qz), r02 = 2 * (qx * qz + qw * qy);
    const r10 = 2 * (qx * qy + qw * qz), r11 = 1 - 2 * (qx * qx + qz * qz), r12 = 2 * (qy * qz - qw * qx);
    const r20 = 2 * (qx * qz - qw * qy), r21 = 2 * (qy * qz + qw * qx), r22 = 1 - 2 * (qx * qx + qy * qy);
    // M = R * S ; Sigma = M * M^T
    const m00 = r00 * s0, m01 = r01 * s1, m02 = r02 * s2;
    const m10 = r10 * s0, m11 = r11 * s1, m12 = r12 * s2;
    const m20 = r20 * s0, m21 = r21 * s1, m22 = r22 * s2;
    const c00 = m00 * m00 + m01 * m01 + m02 * m02;
    const c01 = m00 * m10 + m01 * m11 + m02 * m12;
    const c02 = m00 * m20 + m01 * m21 + m02 * m22;
    const c11 = m10 * m10 + m11 * m11 + m12 * m12;
    const c12 = m10 * m20 + m11 * m21 + m12 * m22;
    const c22 = m20 * m20 + m21 * m21 + m22 * m22;

    const t = 12 * i;
    tex[t] = x; tex[t + 1] = y; tex[t + 2] = z; tex[t + 3] = rgba[4 * i + 3] / 255;
    tex[t + 4] = c00; tex[t + 5] = c01; tex[t + 6] = c02; tex[t + 7] = c11;
    tex[t + 8] = c12; tex[t + 9] = c22; tex[t + 10] = rgba[4 * i] * 65536 + rgba[4 * i + 1] * 256 + rgba[4 * i + 2]; tex[t + 11] = 0;
  }
  return { count: n, trueCount, subsampled: !!(flags & 1), positions, tex, lo, hi };
}
