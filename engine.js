// Gemma 4 E2B in the browser: GGUF reading, the gemma4 BPE tokenizer and the chat prompt.
// Mirrors llama.cpp (commit 0c1e570) so token ids match the Python server exactly.

// ---------- GGUF ----------

const GGUF_TYPES = {0: 'u8', 1: 'i8', 2: 'u16', 3: 'i16', 4: 'u32', 5: 'i32', 6: 'f32', 7: 'bool', 8: 'str', 9: 'arr', 10: 'u64', 11: 'i64', 12: 'f64'};
export const GGML = {F32: 0, F16: 1, Q4_0: 2, Q6_K: 14};

// Parse the header of a GGUF blob (File, OPFS file or a Blob of the first bytes): metadata and tensor infos.
export async function readGGUF(blob) {
  let size = 32 << 20, buf;
  for (;;) {
    buf = await blob.slice(0, Math.min(size, blob.size)).arrayBuffer();
    try { return parseGGUF(buf); } catch (e) { if (!(e instanceof RangeError) || size >= blob.size) throw e; size *= 2; }
  }
}

function parseGGUF(buf) {
  const dv = new DataView(buf), dec = new TextDecoder();
  let p = 0;
  const u32 = () => { const v = dv.getUint32(p, true); p += 4; return v; };
  const u64 = () => { const v = Number(dv.getBigUint64(p, true)); p += 8; return v; };
  const str = () => { const n = u64(); const s = dec.decode(new Uint8Array(buf, p, n)); p += n; return s; };
  const val = t => {
    switch (GGUF_TYPES[t]) {
      case 'u8': return dv.getUint8(p++);
      case 'i8': return dv.getInt8(p++);
      case 'u16': p += 2; return dv.getUint16(p - 2, true);
      case 'i16': p += 2; return dv.getInt16(p - 2, true);
      case 'u32': return u32();
      case 'i32': p += 4; return dv.getInt32(p - 4, true);
      case 'f32': p += 4; return dv.getFloat32(p - 4, true);
      case 'bool': return dv.getUint8(p++) !== 0;
      case 'str': return str();
      case 'u64': return u64();
      case 'i64': p += 8; return Number(dv.getBigInt64(p - 8, true));
      case 'f64': p += 8; return dv.getFloat64(p - 8, true);
      case 'arr': {
        const et = u32(), n = u64();
        if (GGUF_TYPES[et] === 'f32') { const a = new Float32Array(buf.slice(p, p + 4 * n)); p += 4 * n; return a; }
        if (GGUF_TYPES[et] === 'i32') { const a = new Int32Array(buf.slice(p, p + 4 * n)); p += 4 * n; return a; }
        const a = new Array(n);
        for (let i = 0; i < n; i++) a[i] = val(et);
        return a;
      }
      default: throw new Error(`GGUF value type ${t}`);
    }
  };
  if (u32() !== 0x46554747) throw new Error('not a GGUF file');
  const version = u32(), nTensors = u64(), nKV = u64(), meta = {};
  for (let i = 0; i < nKV; i++) { const k = str(); meta[k] = val(u32()); }
  const tensors = {};
  for (let i = 0; i < nTensors; i++) {
    const name = str(), nd = u32(), ne = [];
    for (let j = 0; j < nd; j++) ne.push(u64());
    tensors[name] = {name, ne, type: u32(), offset: u64()};
  }
  const align = meta['general.alignment'] ?? 32, data = Math.ceil(p / align) * align;
  for (const t of Object.values(tensors)) t.offset += data;
  return {version, meta, tensors};
}

// ---------- tokenizer (llama.cpp LLAMA_VOCAB_PRE_TYPE_GEMMA4: SPM-style BPE on raw UTF-8) ----------

const CONTROL = 3, USER_DEFINED = 4, BYTE = 6;

class MinHeap {  // bigrams ordered by (rank, left)
  constructor() { this.a = []; }
  less(x, y) { return x.rank < y.rank || (x.rank === y.rank && x.left < y.left); }
  push(x) {
    const a = this.a; a.push(x);
    for (let i = a.length - 1; i > 0;) { const j = (i - 1) >> 1; if (!this.less(a[i], a[j])) break; [a[i], a[j]] = [a[j], a[i]]; i = j; }
  }
  pop() {
    const a = this.a, top = a[0], last = a.pop();
    if (a.length) {
      a[0] = last;
      for (let i = 0; ;) {
        const l = 2 * i + 1, r = l + 1; let m = i;
        if (l < a.length && this.less(a[l], a[m])) m = l;
        if (r < a.length && this.less(a[r], a[m])) m = r;
        if (m === i) break;
        [a[i], a[m]] = [a[m], a[i]]; i = m;
      }
    }
    return top;
  }
  get size() { return this.a.length; }
}

export class Tokenizer {
  constructor(meta) {
    this.tokens = meta['tokenizer.ggml.tokens'];
    this.types = meta['tokenizer.ggml.token_type'];
    this.ids = new Map(this.tokens.map((t, i) => [t, i]));
    this.ranks = new Map(meta['tokenizer.ggml.merges'].map((m, i) => [m, i]));
    this.bos = meta['tokenizer.ggml.bos_token_id'];
    this.eos = meta['tokenizer.ggml.eos_token_id'];
    // special tokens are matched in the text before BPE, longest first
    this.special = this.tokens.map((t, i) => i).filter(i => this.types[i] === CONTROL || this.types[i] === USER_DEFINED)
      .sort((a, b) => this.tokens[b].length - this.tokens[a].length);
    this.utf8 = new TextEncoder();
  }

  encode(text) {
    let frags = [text];
    for (const id of this.special) {  // split every text fragment on this special token
      const s = this.tokens[id], next = [];
      for (const f of frags) {
        if (typeof f !== 'string' || !f.includes(s)) { next.push(f); continue; }
        const parts = f.split(s);
        parts.forEach((part, k) => { if (k) next.push(id); if (part) next.push(part); });
      }
      frags = next;
    }
    const out = [];
    for (const f of frags) {
      if (typeof f === 'number') out.push(f);
      else for (const w of f.replaceAll(' ', '▁').match(/[^\n]+|\n+/g) || []) this.bpe(w, out);
    }
    return out;
  }

  bpe(word, out) {
    if (/^\n+$/.test(word) && this.ids.has(word)) { out.push(this.ids.get(word)); return; }
    const sym = Array.from(word).map((t, i, a) => ({t, prev: i - 1, next: i + 1 < a.length ? i + 1 : -1}));
    const heap = new MinHeap();
    const add = (l, r) => {
      if (l < 0 || r < 0) return;
      const rank = this.ranks.get(sym[l].t + ' ' + sym[r].t);
      if (rank !== undefined) heap.push({left: l, right: r, rank, text: sym[l].t + sym[r].t});
    };
    for (let i = 1; i < sym.length; i++) add(i - 1, i);
    while (heap.size) {
      const b = heap.pop(), L = sym[b.left], R = sym[b.right];
      if (!L.t || !R.t || L.t + R.t !== b.text) continue;  // outdated
      L.t += R.t; R.t = '';
      L.next = R.next;
      if (R.next >= 0) sym[R.next].prev = b.left;
      add(L.prev, b.left); add(b.left, L.next);
    }
    for (const s of sym) {
      if (!s.t) continue;
      const id = this.ids.get(s.t);
      if (id !== undefined) { out.push(id); continue; }
      for (const byte of this.utf8.encode(s.t)) {  // byte fallback
        const b = this.ids.get(`<0x${byte.toString(16).toUpperCase().padStart(2, '0')}>`);
        if (b !== undefined) out.push(b);
      }
    }
  }

  // Bytes of one token as llama.cpp's token_to_piece renders it; control tokens only when `special`.
  pieceBytes(id, special) {
    const t = this.tokens[id], ty = this.types[id];
    if (ty === CONTROL) return special ? this.utf8.encode(t) : new Uint8Array(0);
    if (ty === BYTE) return new Uint8Array([parseInt(t.slice(3, 5), 16)]);
    if (ty === USER_DEFINED) return this.utf8.encode(t);
    return this.utf8.encode(t.replaceAll('▁', ' '));
  }

  piece(id, special = true) { return new TextDecoder().decode(this.pieceBytes(id, special)); }
}

// The Gemma 4 chat template for plain user / assistant turns, ending with the model's turn opener.
export function chatPrompt(messages) {
  return '<bos>' + messages.map(m => `<|turn>${m.role === 'assistant' ? 'model' : m.role}\n${m.content.trim()}<turn|>\n`).join('') + '<|turn>model\n';
}

// ---------- dequantization helpers (CPU) ----------

const F16 = (() => {  // half -> float table
  const t = new Float32Array(65536);
  for (let h = 0; h < 65536; h++) {
    const s = h & 0x8000 ? -1 : 1, e = (h >> 10) & 31, m = h & 1023;
    t[h] = e === 0 ? s * m * 2 ** -24 : e === 31 ? (m ? NaN : s * Infinity) : s * (1 + m / 1024) * 2 ** (e - 15);
  }
  return t;
})();
export const f16 = h => F16[h];

// Q4_0 (18-byte blocks: f16 d, 16 bytes of nibbles) -> aligned words: 4 u32 of nibbles per block + one f32 scale
function repackQ4(u8, nBlocks) {
  const qs = new Uint32Array(nBlocks * 4), d = new Float32Array(nBlocks), dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  for (let b = 0, o = 0; b < nBlocks; b++, o += 18) {
    d[b] = F16[dv.getUint16(o, true)];
    qs[4 * b] = dv.getUint32(o + 2, true); qs[4 * b + 1] = dv.getUint32(o + 6, true);
    qs[4 * b + 2] = dv.getUint32(o + 10, true); qs[4 * b + 3] = dv.getUint32(o + 14, true);
  }
  return {qs, d};
}

// Q6_K (210-byte blocks: ql[128] qh[64] scales[16] f16 d) -> 53 aligned words: ql 32, qh 16, scales 4, d as f32
const Q6W = 53;
function repackQ6(u8, nBlocks) {
  const out = new Uint32Array(nBlocks * Q6W), f = new Float32Array(out.buffer), dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  for (let b = 0, o = 0; b < nBlocks; b++, o += 210) {
    const w = b * Q6W;
    for (let k = 0; k < 52; k++) out[w + k] = dv.getUint32(o + 4 * k, true);
    f[w + 52] = F16[dv.getUint16(o + 208, true)];
  }
  return out;
}

// one row of a Q6_K matrix, dequantized (the per-layer token embedding is read this way, on the CPU)
function dequantQ6Row(u8, n) {
  const y = new Float32Array(n), dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  for (let b = 0; b < n / 256; b++) {
    const o = b * 210, d = F16[dv.getUint16(o + 208, true)];
    for (let v = 0; v < 256; v++) {
      const half = v >> 7, rem = v & 127, qd = rem >> 5, l = rem & 31;
      const ql = u8[o + half * 64 + l + (qd & 1) * 32], qh = u8[o + 128 + half * 32 + l];
      const q = (((ql >> ((qd >> 1) * 4)) & 15) | (((qh >> (qd * 2)) & 3) << 4)) - 32;
      const sc = (u8[o + 192 + half * 8 + (l >> 4) + qd * 2] << 24) >> 24;
      y[b * 256 + v] = d * sc * q;
    }
  }
  return y;
}

// ---------- WGSL kernels ----------
// Activations live in one f32 "arena" buffer; kernels take float offsets into it.

const WG = 256;
export const N_CTX = 4096;
const DYN =`struct Dyn { pos: u32, tok: u32, ivOff: u32, ivMode: u32, ivDelta: f32, p0: u32, p1: u32, p2: u32 }`;
const GELU = `fn gelu(x: f32) -> f32 { let t = clamp(0.7978845608 * (x + 0.044715 * x * x * x), -15.0, 15.0); return 0.5 * x * (1.0 + tanh(t)); }`;
const Q6 = `fn q6(base: u32, v: u32) -> f32 {
  let half = v >> 7u; let rem = v & 127u; let qd = rem >> 5u; let l = rem & 31u;
  let qi = half * 64u + l + (qd & 1u) * 32u; let hi = half * 32u + l; let si = half * 8u + (l >> 4u) + qd * 2u;
  let ql = (W[base + qi / 4u] >> (8u * (qi % 4u))) & 0xFFu;
  let qh = (W[base + 32u + hi / 4u] >> (8u * (hi % 4u))) & 0xFFu;
  let sb = (W[base + 48u + si / 4u] >> (8u * (si % 4u))) & 0xFFu;
  let q = i32(((ql >> ((qd >> 1u) * 4u)) & 15u) | (((qh >> (qd * 2u)) & 3u) << 4u)) - 32;
  let sc = i32(sb << 24u) >> 24u;
  return bitcast<f32>(W[base + 52u]) * f32(sc * q);
}`;

const SHADERS = {
  // y = s * W x for a Q4_0 matrix; 8 rows per workgroup, 32 threads per row; x read as vec4 (offsets are multiples of 4)
  mv_q4: `
struct P { rows: u32, nb: u32, xo: u32, yo: u32, s: f32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qs: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read> dd: array<f32>;
@group(0) @binding(3) var<storage, read_write> A: array<vec4<f32>>;
var<workgroup> part: array<f32, ${WG}>;
// 4 bytes of a block: low nibbles are values 4w..4w+3 (x0), high nibbles values 16+4w.. (x1)
fn nib(q: u32, x0: vec4<f32>, x1: vec4<f32>) -> f32 {
  let b = vec4<u32>(q & 0xFFu, (q >> 8u) & 0xFFu, (q >> 16u) & 0xFFu, q >> 24u);
  return dot(vec4<f32>(b & vec4<u32>(15u)) - 8.0, x0) + dot(vec4<f32>(b >> vec4<u32>(4u)) - 8.0, x1);
}
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let r = wg.x * 8u + li.x / 32u; let t = li.x % 32u;
  var s = 0.0;
  if (r < p.rows) {
    for (var b = t; b < p.nb; b += 32u) {
      let blk = r * p.nb + b; let q = qs[blk]; let x = (p.xo + b * 32u) / 4u;
      s += dd[blk] * (nib(q.x, A[x], A[x + 4u]) + nib(q.y, A[x + 1u], A[x + 5u]) + nib(q.z, A[x + 2u], A[x + 6u]) + nib(q.w, A[x + 3u], A[x + 7u]));
    }
  }
  part[li.x] = s;
  workgroupBarrier();
  for (var o = 16u; o > 0u; o >>= 1u) { if (t < o) { part[li.x] += part[li.x + o]; } workgroupBarrier(); }
  if (t == 0u && r < p.rows) { let y = p.yo + r; A[y / 4u][y % 4u] = p.s * part[li.x]; }
}`,
  // y = W x for a Q6_K matrix (the tied LM head), optional softcap c * tanh(y / c)
  mv_q6: `
struct P { rows: u32, nb: u32, xo: u32, yo: u32, cap: f32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read_write> A: array<f32>;
var<workgroup> part: array<f32, ${WG}>;
${Q6}
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let r = wg.x * 8u + li.x / 32u; let t = li.x % 32u;
  var s = 0.0;
  if (r < p.rows) {
    for (var b = 0u; b < p.nb; b++) {
      let base = (r * p.nb + b) * ${Q6W}u; let xb = p.xo + b * 256u;
      for (var v = t; v < 256u; v += 32u) { s += q6(base, v) * A[xb + v]; }
    }
  }
  part[li.x] = s;
  workgroupBarrier();
  for (var o = 16u; o > 0u; o >>= 1u) { if (t < o) { part[li.x] += part[li.x + o]; } workgroupBarrier(); }
  if (t == 0u && r < p.rows) {
    var y = part[li.x];
    if (p.cap > 0.0) { y = p.cap * tanh(clamp(y / p.cap, -15.0, 15.0)); }
    A[p.yo + r] = y;
  }
}`,
  // the token's row of the Q6_K embedding table, times s
  emb_q6: `
struct P { n: u32, nb: u32, yo: u32, p0: u32, s: f32, p1: u32, p2: u32, p3: u32 }
${DYN}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<u32>;
@group(0) @binding(2) var<storage, read_write> A: array<f32>;
@group(0) @binding(3) var<uniform> dyn: Dyn;
${Q6}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= p.n) { return; }
  A[p.yo + g.x] = p.s * q6((dyn.tok * p.nb + g.x / 256u) * ${Q6W}u, g.x % 256u);
}`,
  // y = s * W x for an f32 matrix
  mv_f32: `
struct P { rows: u32, cols: u32, xo: u32, yo: u32, s: f32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> W: array<f32>;
@group(0) @binding(2) var<storage, read_write> A: array<f32>;
var<workgroup> part: array<f32, ${WG}>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let r = wg.x * 8u + li.x / 32u; let t = li.x % 32u;
  var s = 0.0;
  if (r < p.rows) { for (var c = t; c < p.cols; c += 32u) { s += W[r * p.cols + c] * A[p.xo + c]; } }
  part[li.x] = s;
  workgroupBarrier();
  for (var o = 16u; o > 0u; o >>= 1u) { if (t < o) { part[li.x] += part[li.x + o]; } workgroupBarrier(); }
  if (t == 0u && r < p.rows) { A[p.yo + r] = p.s * part[li.x]; }
}`,
  // RMS norm over groups of n values (one workgroup per group), optional weight of length n
  rms: `
struct P { n: u32, xo: u32, yo: u32, hasw: u32, eps: f32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> w: array<f32>;
@group(0) @binding(2) var<storage, read_write> A: array<f32>;
var<workgroup> part: array<f32, ${WG}>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let xo = p.xo + wg.x * p.n; let yo = p.yo + wg.x * p.n;
  var s = 0.0;
  for (var i = li.x; i < p.n; i += ${WG}u) { let v = A[xo + i]; s += v * v; }
  part[li.x] = s;
  workgroupBarrier();
  for (var o = ${WG / 2}u; o > 0u; o >>= 1u) { if (li.x < o) { part[li.x] += part[li.x + o]; } workgroupBarrier(); }
  let k = 1.0 / sqrt(part[0] / f32(p.n) + p.eps);
  for (var i = li.x; i < p.n; i += ${WG}u) { A[yo + i] = A[xo + i] * k * select(1.0, w[i], p.hasw != 0u); }
}`,
  // NEOX rope on heads of hd values at position dyn.pos; theta_i = pos * base^(-2i/hd) / ff[i]
  rope: `
struct P { heads: u32, hd: u32, o: u32, hasff: u32, base: f32, p0: u32, p1: u32, p2: u32 }
${DYN}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> ff: array<f32>;
@group(0) @binding(2) var<storage, read_write> A: array<f32>;
@group(0) @binding(3) var<uniform> dyn: Dyn;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  let half = p.hd / 2u;
  if (g.x >= p.heads * half) { return; }
  let h = g.x / half; let i = g.x % half;
  var theta = f32(dyn.pos) * pow(p.base, -2.0 * f32(i) / f32(p.hd));
  if (p.hasff != 0u) { theta = theta / ff[i]; }
  let c = cos(theta); let s = sin(theta);
  let a = p.o + h * p.hd + i;
  let x0 = A[a]; let x1 = A[a + half];
  A[a] = x0 * c - x1 * s; A[a + half] = x0 * s + x1 * c;
}`,
  // element-wise: 0 y = a*sa + b*sb, 1 y = a*b*sa, 2 y = gelu(a)*b, 3 y = gelu(a)
  ew: `
struct P { n: u32, ao: u32, bo: u32, yo: u32, op: u32, sa: f32, sb: f32, p0: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> A: array<f32>;
${GELU}
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= p.n) { return; }
  let a = A[p.ao + g.x];
  var y = 0.0;
  switch p.op {
    case 0u: { y = a * p.sa + A[p.bo + g.x] * p.sb; }
    case 1u: { y = a * A[p.bo + g.x] * p.sa; }
    case 2u: { y = gelu(a) * A[p.bo + g.x]; }
    default: { y = gelu(a); }
  }
  A[p.yo + g.x] = y;
}`,
  // store this token's K and V at dyn.pos
  kv: `
struct P { hd: u32, ko: u32, vo: u32, p0: u32 }
${DYN}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> A: array<f32>;
@group(0) @binding(2) var<storage, read_write> K: array<f32>;
@group(0) @binding(3) var<storage, read_write> V: array<f32>;
@group(0) @binding(4) var<uniform> dyn: Dyn;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= p.hd) { return; }
  K[dyn.pos * p.hd + g.x] = A[p.ko + g.x];
  V[dyn.pos * p.hd + g.x] = A[p.vo + g.x];
}`,
  // attention of one head (workgroup) over keys [start, pos]: scores q.k_j (scale 1), softmax, sum of p_j v_j.
  // All heads share the one KV head.
  att: `
struct P { hd: u32, qo: u32, oo: u32, swa: u32 }
${DYN}
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> K: array<f32>;
@group(0) @binding(2) var<storage, read> V: array<f32>;
@group(0) @binding(3) var<storage, read_write> A: array<f32>;
@group(0) @binding(4) var<uniform> dyn: Dyn;
var<workgroup> sc: array<f32, ${N_CTX}>;
var<workgroup> qv: array<f32, 512>;
var<workgroup> part: array<f32, ${WG}>;
@compute @workgroup_size(${WG}) fn main(@builtin(workgroup_id) wg: vec3u, @builtin(local_invocation_id) li: vec3u) {
  let h = wg.x;
  var start = 0u;
  if (p.swa > 0u && dyn.pos >= p.swa) { start = dyn.pos - p.swa + 1u; }
  let n = dyn.pos - start + 1u;
  for (var d = li.x; d < p.hd; d += ${WG}u) { qv[d] = A[p.qo + h * p.hd + d]; }
  workgroupBarrier();
  var m = -3.4e38;
  for (var j = li.x; j < n; j += ${WG}u) {
    let k = (start + j) * p.hd;
    var s = 0.0;
    for (var d = 0u; d < p.hd; d++) { s += qv[d] * K[k + d]; }
    sc[j] = s; m = max(m, s);
  }
  part[li.x] = m;
  workgroupBarrier();
  for (var o = ${WG / 2}u; o > 0u; o >>= 1u) { if (li.x < o) { part[li.x] = max(part[li.x], part[li.x + o]); } workgroupBarrier(); }
  let mx = part[0];
  workgroupBarrier();
  var z = 0.0;
  for (var j = li.x; j < n; j += ${WG}u) { let e = exp(sc[j] - mx); sc[j] = e; z += e; }
  part[li.x] = z;
  workgroupBarrier();
  for (var o = ${WG / 2}u; o > 0u; o >>= 1u) { if (li.x < o) { part[li.x] += part[li.x + o]; } workgroupBarrier(); }
  let inv = 1.0 / part[0];
  for (var d = li.x; d < p.hd; d += ${WG}u) {
    var s = 0.0;
    for (var j = 0u; j < n; j++) { s += sc[j] * V[(start + j) * p.hd + d]; }
    A[p.oo + h * p.hd + d] = s * inv;
  }
}`,
  // max and sum of exp(x - max) of the logits (the softmax normalizer), one workgroup
  lse: `
struct P { n: u32, xo: u32, yo: u32, p0: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read_write> A: array<f32>;
var<workgroup> part: array<f32, ${WG}>;
@compute @workgroup_size(${WG}) fn main(@builtin(local_invocation_id) li: vec3u) {
  var m = -3.4e38;
  for (var i = li.x; i < p.n; i += ${WG}u) { m = max(m, A[p.xo + i]); }
  part[li.x] = m;
  workgroupBarrier();
  for (var o = ${WG / 2}u; o > 0u; o >>= 1u) { if (li.x < o) { part[li.x] = max(part[li.x], part[li.x + o]); } workgroupBarrier(); }
  let mx = part[0];
  workgroupBarrier();
  var s = 0.0;
  for (var i = li.x; i < p.n; i += ${WG}u) { s += exp(A[p.xo + i] - mx); }
  part[li.x] = s;
  workgroupBarrier();
  for (var o = ${WG / 2}u; o > 0u; o >>= 1u) { if (li.x < o) { part[li.x] += part[li.x + o]; } workgroupBarrier(); }
  if (li.x == 0u) { A[p.yo] = mx; A[p.yo + 1u] = part[0]; }
}`,
  // ablation / steering: rewrite one neuron (dyn.ivOff) to 0 or + delta
  patch: `
${DYN}
@group(0) @binding(0) var<storage, read_write> A: array<f32>;
@group(0) @binding(1) var<uniform> dyn: Dyn;
@compute @workgroup_size(1) fn main() {
  if (dyn.ivMode == 1u) { A[dyn.ivOff] = 0.0; } else if (dyn.ivMode == 2u) { A[dyn.ivOff] += dyn.ivDelta; }
}`,
  // column i of a Q4_0 matrix (an output weight), times a gain vector
  col_q4: `
struct P { rows: u32, nb: u32, i: u32, yo: u32, go: u32, p0: u32, p1: u32, p2: u32 }
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage, read> qs: array<u32>;
@group(0) @binding(2) var<storage, read> dd: array<f32>;
@group(0) @binding(3) var<storage, read_write> A: array<f32>;
@compute @workgroup_size(${WG}) fn main(@builtin(global_invocation_id) g: vec3u) {
  if (g.x >= p.rows) { return; }
  let blk = g.x * p.nb + p.i / 32u; let j = p.i % 32u;
  let by = (qs[blk * 4u + (j % 16u) / 4u] >> (8u * (j % 4u))) & 0xFFu;
  let q = select(by & 15u, by >> 4u, j >= 16u);
  A[p.yo + g.x] = dd[blk] * f32(i32(q) - 8) * A[p.go + g.x];
}`,
};

// ---------- model ----------

const BYTES ={[GGML.F32]: n => 4 * n, [GGML.F16]: n => 2 * n, [GGML.Q4_0]: n => n / 32 * 18, [GGML.Q6_K]: n => n / 256 * 210};
const F = v => ({f: v});  // marks a float in a uniform
// detail modules of the selected layer, copied out of the per-layer scratch after the layer
const DETAIL_KEYS = ['attn_norm', 'Qcur_pos', 'Kcur_pos', 'Vcur_normed', 'attn_post_norm', 'attn_out', 'ffn_norm', 'ffn_gate', 'ffn_up',
  'ffn_post_norm', 'pe_in', 'ple_gelu', 'ple_in', 'ple_mul', 'ple_proj', 'per_layer_embd_out', 'resid_ple', 'o_proj', 'ffn_out'];

export class Model {
  // blob: the GGUF (File or OPFS file); g: its parsed header
  static async load(blob, g, onProgress = () => {}) {
    if (!navigator.gpu) throw new Error('이 브라우저는 WebGPU를 지원하지 않습니다.');
    const adapter = await navigator.gpu.requestAdapter({powerPreference: 'high-performance'});
    if (!adapter) throw new Error('WebGPU 어댑터를 찾지 못했습니다.');
    const lim = adapter.limits;
    const device = await adapter.requestDevice({requiredLimits: {maxBufferSize: lim.maxBufferSize, maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
      maxComputeWorkgroupStorageSize: lim.maxComputeWorkgroupStorageSize}});  // attention keeps up to N_CTX scores in workgroup memory
    const m = new Model(device, blob, g);
    await m.init(onProgress);
    return m;
  }

  constructor(device, blob, g) {
    this.device = device; this.blob = blob; this.g = g; this.pre = new Map();
    const meta = g.meta, a = 'gemma4.';
    this.nLayer = meta[a + 'block_count']; this.nEmbd = meta[a + 'embedding_length']; this.nHead = meta[a + 'attention.head_count'];
    this.nFF = meta[a + 'feed_forward_length']; this.nPL = meta[a + 'embedding_length_per_layer_input'];
    this.swa = meta[a + 'attention.sliding_window_pattern']; this.window = meta[a + 'attention.sliding_window'];
    this.eps = meta[a + 'attention.layer_norm_rms_epsilon']; this.cap = meta[a + 'final_logit_softcapping'];
    this.nKV = this.nLayer - meta[a + 'attention.shared_kv_layers'];
    this.hd = this.swa.map(s => s ? meta[a + 'attention.key_length_swa'] : meta[a + 'attention.key_length']);
    this.base = this.swa.map(s => s ? meta[a + 'rope.freq_base_swa'] : meta[a + 'rope.freq_base']);
    this.nVocab = meta['tokenizer.ggml.tokens'].length;
    // layers without their own K / V read the last earlier layer of the same attention type
    this.kvFrom = this.swa.map((s, l) => l < this.nKV ? l : this.swa.slice(0, this.nKV).lastIndexOf(s));
  }

  async read(name) {
    const t = this.g.tensors[name], n = t.ne.reduce((x, y) => x * y, 1);
    const u8 = this.pre.has(name) ? await this.pre.get(name) : new Uint8Array(await this.blob.slice(t.offset, t.offset + BYTES[t.type](n)).arrayBuffer());
    this.pre.delete(name);
    return {t, n, u8};
  }

  // Start reading tensors that lie next to each other in the file (one layer) with a single request.
  prefetchSpan(names) {
    const ts = names.map(n => this.g.tensors[n]).filter(Boolean), size = t => BYTES[t.type](t.ne.reduce((x, y) => x * y, 1));
    const a = Math.min(...ts.map(t => t.offset)), b = Math.max(...ts.map(t => t.offset + size(t)));
    if (b - a > 1.01 * ts.reduce((s, t) => s + size(t), 0)) return;  // not contiguous: read one by one
    const buf = this.blob.slice(a, b).arrayBuffer();
    for (const t of ts) this.pre.set(t.name, buf.then(ab => new Uint8Array(ab, t.offset - a, size(t))));
  }

  buffer(data, usage = GPUBufferUsage.STORAGE) {
    const b = this.device.createBuffer({size: Math.max(16, Math.ceil(data.byteLength / 4) * 4), usage, mappedAtCreation: true});
    new Uint8Array(b.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
    b.unmap();
    return b;
  }

  async q4(name) {  // {qs, d, rows, nb}
    const {t, u8} = await this.read(name), nb = t.ne[0] / 32, rows = t.ne[1];
    const {qs, d} = repackQ4(u8, rows * nb);
    return {qs: this.buffer(qs), d: this.buffer(d), rows, nb};
  }

  async f32(name) {
    const {t, n, u8} = await this.read(name);
    if (t.type === GGML.F32) return {buf: this.buffer(new Float32Array(u8.buffer, u8.byteOffset, n)), data: new Float32Array(u8.slice().buffer)};
    const h = new Uint16Array(u8.buffer, u8.byteOffset, n), out = new Float32Array(n);
    for (let i = 0; i < n; i++) out[i] = F16[h[i]];
    return {buf: this.buffer(out), data: out};
  }

  async init(onProgress) {
    const dev = this.device, L = this.nLayer, D = this.nEmbd, NH = this.nHead, NPL = this.nPL;
    const names = Object.keys(this.g.tensors).filter(n => n !== 'per_layer_token_embd.weight');
    let done = 0;
    const total = names.reduce((s, n) => { const t = this.g.tensors[n]; return s + BYTES[t.type](t.ne.reduce((x, y) => x * y, 1)); }, 0);
    const tick = name => { const t = this.g.tensors[name]; done += BYTES[t.type](t.ne.reduce((x, y) => x * y, 1)); onProgress(done / total); };

    // pipelines
    this.pipes = {};
    await Promise.all(Object.entries(SHADERS).map(async ([k, code]) => {
      this.pipes[k] = await dev.createComputePipelineAsync({layout: 'auto', compute: {module: dev.createShaderModule({code}), entryPoint: 'main'}});
    }));

    // weights
    const tok = this.g.tensors['token_embd.weight'], rowB = tok.ne[0] / 256 * 210, nb6 = tok.ne[0] / 256;
    this.tokEmb = dev.createBuffer({size: tok.ne[1] * nb6 * Q6W * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST});
    for (let r = 0; r < tok.ne[1]; r += 16384) {  // in chunks: 330 MB on the GPU
      const n = Math.min(16384, tok.ne[1] - r);
      const u8 = new Uint8Array(await this.blob.slice(tok.offset + r * rowB, tok.offset + (r + n) * rowB).arrayBuffer());
      dev.queue.writeBuffer(this.tokEmb, r * nb6 * Q6W * 4, repackQ6(u8, n * nb6));
      done += u8.byteLength; onProgress(done / total);
    }
    const glob = {};
    for (const n of ['per_layer_model_proj.weight', 'per_layer_proj_norm.weight', 'output_norm.weight', 'rope_freqs.weight']) {
      glob[n] = await this.f32(n); tick(n);
    }
    this.plProj = glob['per_layer_model_proj.weight'].buf;
    this.plNorm = glob['per_layer_proj_norm.weight'].buf;
    this.outNorm = glob['output_norm.weight'];
    this.ropeFreqs = glob['rope_freqs.weight'].buf;
    this.ones = this.buffer(new Float32Array(512).fill(1));
    this.layers = [];
    const layerNames = l => names.filter(n => n.startsWith(`blk.${l}.`));
    this.prefetchSpan(layerNames(0)); this.prefetchSpan(layerNames(1));
    for (let l = 0; l < L; l++) {
      if (l + 2 < L) this.prefetchSpan(layerNames(l + 2));  // keep two layers in flight
      const w = {}, p = `blk.${l}.`;
      for (const k of ['attn_q', 'attn_k', 'attn_v', 'attn_output', 'ffn_gate', 'ffn_up', 'ffn_down', 'inp_gate', 'proj']) {
        if (this.g.tensors[p + k + '.weight']) { w[k] = await this.q4(p + k + '.weight'); tick(p + k + '.weight'); }
      }
      for (const k of ['attn_norm', 'attn_q_norm', 'attn_k_norm', 'post_attention_norm', 'ffn_norm', 'post_ffw_norm', 'post_norm', 'layer_output_scale']) {
        if (this.g.tensors[p + k + '.weight']) { w[k] = await this.f32(p + k + '.weight'); tick(p + k + '.weight'); }
      }
      this.layers.push(w);
    }
    const ple = this.g.tensors['per_layer_token_embd.weight'];
    this.ple = {offset: ple.offset, rowBytes: ple.ne[0] / 256 * 210, n: ple.ne[0]};
    this.rows = new Map();  // token -> raw bytes of its per-layer embedding row (a promise while reading)

    // arena layout (floats): everything the UI reads first, contiguous, then scratch
    let off = 0;
    const al = n => { const o = off; off += Math.ceil(n / 4) * 4; return o; };
    this.mlpOff = this.nFF.map(n => al(n));
    this.headsOff = this.hd.map(h => al(NH * h));
    this.residOff = this.hd.map(() => al(D));
    this.EMB = al(D); this.FINAL = al(D);
    const maxFF = Math.max(...this.nFF), maxHD = Math.max(...this.hd);
    const dsize = {Qcur_pos: NH * maxHD, Kcur_pos: maxHD, Vcur_normed: maxHD, ffn_gate: maxFF, ffn_up: maxFF, ple_gelu: NPL, ple_in: NPL, ple_mul: NPL};
    this.detailOff = Object.fromEntries(DETAIL_KEYS.map(k => [k, al(dsize[k] ?? D)]));
    this.LSE = al(2);  // logits max, sum of exp(logit - max)
    this.LOGITS = al(this.nVocab);
    this.readEnd = off;
    const S = {};
    for (const [k, n] of [['XN', D], ['Q', NH * maxHD], ['K', maxHD], ['V', maxHD], ['O', D], ['APN', D], ['ATT', D], ['FN', D], ['G', maxFF],
      ['U', maxFF], ['DN', D], ['FPN', D], ['PE', D], ['PG', NPL], ['PM', NPL], ['PP', D], ['PN', D], ['RP', D], ['PIN', NPL * L],
      ['PPROJ', NPL * L], ['PLETOK', NPL * L], ['PU', D], ['PS', this.nVocab]]) S[k] = al(n);
    this.S = S;
    this.arena = dev.createBuffer({size: off * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST});
    this.staging = dev.createBuffer({size: this.readEnd * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
    this.pstaging = dev.createBuffer({size: this.nVocab * 4, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST});
    this.dyn = dev.createBuffer({size: 32, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST});
    this.kv = [];
    for (let l = 0; l < this.nKV; l++) {
      const size = N_CTX * this.hd[l] * 4, use = GPUBufferUsage.STORAGE;
      this.kv.push({K: dev.createBuffer({size, usage: use}), V: dev.createBuffer({size, usage: use})});
    }
    this.program();
  }

  uni(...vals) {
    const ab = new ArrayBuffer(32), dv = new DataView(ab);
    vals.forEach((v, i) => typeof v === 'object' ? dv.setFloat32(4 * i, v.f, true) : dv.setUint32(4 * i, v, true));
    return this.buffer(new Uint8Array(ab), GPUBufferUsage.UNIFORM);
  }

  op(kind, bufs, wg) {
    const pipe = this.pipes[kind];
    const bg = this.device.createBindGroup({layout: pipe.getBindGroupLayout(0), entries: bufs.map((buffer, binding) => ({binding, resource: {buffer}}))});
    return {pipe, bg, wg};
  }

  // the fixed sequence of dispatches for one token (llama.cpp gemma4 graph order)
  program() {
    const L = this.nLayer, D = this.nEmbd, NH = this.nHead, NPL = this.nPL, A = this.arena, S = this.S, eps = this.eps;
    const cd = n => Math.ceil(n / WG), mv = (w, xo, yo, s = 1) => this.op('mv_q4', [this.uni(w.rows, w.nb, xo, yo, F(s)), w.qs, w.d, A], [Math.ceil(w.rows / 8)]);
    const rms = (n, groups, xo, yo, w) => this.op('rms', [this.uni(n, xo, yo, w ? 1 : 0, F(eps)), w ? w.buf : this.ones, A], [groups]);
    const ew = (op, n, ao, bo, yo, sa = 1, sb = 1) => this.op('ew', [this.uni(n, ao, bo, yo, op, F(sa), F(sb)), A], [cd(n)]);
    this.head = [
      this.op('emb_q6', [this.uni(D, D / 256, this.EMB, 0, F(Math.sqrt(D))), this.tokEmb, A, this.dyn], [cd(D)]),
      this.op('mv_f32', [this.uni(NPL * L, D, this.EMB, S.PPROJ, F(1 / Math.sqrt(D))), this.plProj, A], [Math.ceil(NPL * L / 8)]),
      rms(NPL, L, S.PPROJ, S.PPROJ, {buf: this.plNorm}),
      ew(0, NPL * L, S.PPROJ, S.PLETOK, S.PIN, Math.SQRT1_2, Math.sqrt(NPL) * Math.SQRT1_2),
    ];
    this.body = [];
    for (let l = 0; l < L; l++) {
      const w = this.layers[l], hd = this.hd[l], nff = this.nFF[l], xin = l ? this.residOff[l - 1] : this.EMB, kv = this.kv[this.kvFrom[l]];
      const ff = this.swa[l] ? 0 : 1, swa = this.swa[l] ? this.window : 0, ops = [];
      ops.push(rms(D, 1, xin, S.XN, w.attn_norm), mv(w.attn_q, S.XN, S.Q), rms(hd, NH, S.Q, S.Q, w.attn_q_norm),
        this.op('rope', [this.uni(NH, hd, S.Q, ff, F(this.base[l])), this.ropeFreqs, A, this.dyn], [cd(NH * hd / 2)]));
      if (l < this.nKV) {
        ops.push(mv(w.attn_k, S.XN, S.K), mv(w.attn_v, S.XN, S.V), rms(hd, 1, S.K, S.K, w.attn_k_norm), rms(hd, 1, S.V, S.V, null),
          this.op('rope', [this.uni(1, hd, S.K, ff, F(this.base[l])), this.ropeFreqs, A, this.dyn], [cd(hd / 2)]),
          this.op('kv', [this.uni(hd, S.K, S.V, 0), A, kv.K, kv.V, this.dyn], [cd(hd)]));
      }
      ops.push(this.op('att', [this.uni(hd, S.Q, this.headsOff[l], swa), kv.K, kv.V, A, this.dyn], [NH]),
        mv(w.attn_output, this.headsOff[l], S.O), rms(D, 1, S.O, S.APN, w.post_attention_norm), ew(0, D, S.APN, xin, S.ATT),
        rms(D, 1, S.ATT, S.FN, w.ffn_norm), mv(w.ffn_gate, S.FN, S.G), mv(w.ffn_up, S.FN, S.U), ew(2, nff, S.G, S.U, this.mlpOff[l]),
        {iv: 'mlp'},
        mv(w.ffn_down, this.mlpOff[l], S.DN), rms(D, 1, S.DN, S.FPN, w.post_ffw_norm), ew(0, D, S.FPN, S.ATT, S.PE),
        mv(w.inp_gate, S.PE, S.PG), ew(3, NPL, S.PG, S.PG, S.PG), ew(1, NPL, S.PG, S.PIN + l * NPL, S.PM),
        {iv: 'ple'},
        mv(w.proj, S.PM, S.PP), rms(D, 1, S.PP, S.PN, w.post_norm), ew(0, D, S.PE, S.PN, S.RP),
        ew(0, D, S.RP, S.RP, this.residOff[l], w.layer_output_scale.data[0], 0));
      const src = {attn_norm: [S.XN, D], Qcur_pos: [S.Q, NH * hd], attn_post_norm: [S.APN, D], attn_out: [S.ATT, D], ffn_norm: [S.FN, D],
        ffn_gate: [S.G, nff], ffn_up: [S.U, nff], ffn_post_norm: [S.FPN, D], pe_in: [S.PE, D], ple_gelu: [S.PG, NPL], ple_in: [S.PIN + l * NPL, NPL],
        ple_mul: [S.PM, NPL], ple_proj: [S.PP, D], per_layer_embd_out: [S.PN, D], resid_ple: [S.RP, D], o_proj: [S.O, D], ffn_out: [S.DN, D]};
      if (l < this.nKV) Object.assign(src, {Kcur_pos: [S.K, hd], Vcur_normed: [S.V, hd]});
      const detail = Object.entries(src).map(([k, [o, n]]) => ew(0, n, o, o, this.detailOff[k], 1, 0));
      this.body.push({ops, detail, keys: Object.fromEntries(Object.entries(src).map(([k, [, n]]) => [k, n]))});
    }
    this.tail = [rms(D, 1, this.residOff[L - 1], this.FINAL, this.outNorm),
      this.op('mv_q6', [this.uni(this.nVocab, D / 256, this.FINAL, this.LOGITS, F(this.cap)), this.tokEmb, A], [Math.ceil(this.nVocab / 8)]),
      this.op('lse', [this.uni(this.nVocab, this.LOGITS, this.LSE, 0), A], [1])];
  }

  // Evaluate one token at `pos` (its K / V go into the cache). read: return the UI values (Float32Array of the read area).
  // detail: layer whose modules are copied out. iv: {kind: 'mlp' | 'ple', layer, index, mode: 'zero' | 'add', delta}
  // The per-layer token embedding table (1.9 GB) stays in the file: rows are read when needed, and kept.
  pleRow(tok) {
    const P = this.ple;
    if (!this.rows.has(tok)) {
      if (this.rows.size > 20000) this.rows.clear();
      this.rows.set(tok, this.blob.slice(P.offset + tok * P.rowBytes, P.offset + (tok + 1) * P.rowBytes).arrayBuffer()
        .catch(e => { this.rows.delete(tok); throw e; }));
    }
    return this.rows.get(tok);
  }

  prefetch(ids) { ids.forEach(t => this.pleRow(t)); }

  async step(tok, pos, {read = false, detail = -1, iv = null} = {}) {
    const dev = this.device, q = dev.queue;
    const row = dequantQ6Row(new Uint8Array(await this.pleRow(tok)), this.ple.n);
    q.writeBuffer(this.arena, this.S.PLETOK * 4, row);
    const dyn = new DataView(new ArrayBuffer(32));
    dyn.setUint32(0, pos, true); dyn.setUint32(4, tok, true);
    if (iv) {
      dyn.setUint32(8, (iv.kind === 'mlp' ? this.mlpOff[iv.layer] : this.S.PM) + iv.index, true);
      dyn.setUint32(12, iv.mode === 'zero' ? 1 : 2, true); dyn.setFloat32(16, iv.delta || 0, true);
    }
    q.writeBuffer(this.dyn, 0, dyn.buffer);
    const enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
    const run = o => { pass.setPipeline(o.pipe); pass.setBindGroup(0, o.bg); const wg = typeof o.wg === 'function' ? o.wg(pos) : o.wg; pass.dispatchWorkgroups(...wg); };
    this.head.forEach(run);
    this.body.forEach((b, l) => {
      for (const o of b.ops) {
        if (!o.iv) run(o);
        else if (iv && iv.kind === o.iv && iv.layer === l) run(this.patchOp ??= this.op('patch', [this.arena, this.dyn], [1]));
      }
      if (l === detail) b.detail.forEach(run);
    });
    this.tail.forEach(run);
    pass.end();
    if (read) enc.copyBufferToBuffer(this.arena, 0, this.staging, 0, this.readEnd * 4);
    q.submit([enc.finish()]);
    if (!read) return null;
    await this.staging.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(this.staging.getMappedRange().slice(0));
    this.staging.unmap();
    return out;
  }

  // logits of the tied unembedding against s * (column i of an output weight) * gain * final norm gain
  async pushScores(kind, l, i) {
    const w = this.layers[l], mat = kind === 'mlp' ? w.ffn_down : w.proj, gain = kind === 'mlp' ? w.post_ffw_norm : w.post_norm;
    const D = this.nEmbd, g = new Float32Array(D);
    for (let k = 0; k < D; k++) g[k] = gain.data[k] * this.outNorm.data[k];
    this.device.queue.writeBuffer(this.arena, this.S.PU * 4, g);
    const enc = this.device.createCommandEncoder(), pass = enc.beginComputePass();
    for (const o of [this.op('col_q4', [this.uni(mat.rows, mat.nb, i, this.S.PU, this.S.PU), mat.qs, mat.d, this.arena], [Math.ceil(D / WG)]),
      this.op('mv_q6', [this.uni(this.nVocab, D / 256, this.S.PU, this.S.PS, F(0)), this.tokEmb, this.arena], [Math.ceil(this.nVocab / 8)])]) {
      pass.setPipeline(o.pipe); pass.setBindGroup(0, o.bg); pass.dispatchWorkgroups(...o.wg);
    }
    pass.end();
    enc.copyBufferToBuffer(this.arena, this.S.PS * 4, this.pstaging, 0, this.nVocab * 4);
    this.device.queue.submit([enc.finish()]);
    await this.pstaging.mapAsync(GPUMapMode.READ);
    const s = new Float32Array(this.pstaging.getMappedRange().slice(0));
    this.pstaging.unmap();
    return s;
  }
}
