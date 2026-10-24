// In-browser stand-in for server.py: speaks the same messages to index.html, running the model with engine.js.
import {readGGUF, Tokenizer, Model, chatPrompt, N_CTX, f16} from './engine.js';

export const MODEL_URL = 'https://huggingface.co/google/gemma-4-E2B-it-qat-q4_0-gguf/resolve/main/gemma-4-E2B_q4_0-it.gguf';
export const MODEL_SIZE = 3349516256;
const MODEL_NAME = 'gemma-4-E2B_q4_0-it.gguf';
const SPECIAL_RE = /^<[^<>]*>$/;  // control / unused vocabulary entries, skipped in token lists

// The GGUF kept in the origin-private file system, or null.
export async function storedModel() {
  try {
    const root = await navigator.storage.getDirectory();
    const f = await (await root.getFileHandle(MODEL_NAME)).getFile();
    return f.size === MODEL_SIZE ? f : null;
  } catch { return null; }
}

// A remote file read through HTTP Range requests, in place of a stored copy.
export class HttpBlob {
  constructor(url, size) { this.url = url; this.size = size; }
  slice(a, b) {
    return {arrayBuffer: async () => {
      const r = await fetch(this.url, {headers: {Range: `bytes=${a}-${b - 1}`}});
      if (r.status !== 206) throw new Error(`부분 다운로드 실패 (HTTP ${r.status})`);
      return r.arrayBuffer();
    }};
  }
}

export async function hasRoom() {
  const {quota = 0, usage = 0} = await navigator.storage.estimate();
  return quota - usage >= MODEL_SIZE * 1.05;
}

// Download the GGUF into the origin-private file system (once), reporting progress 0..1.
export async function downloadModel(url, onProgress) {
  await navigator.storage.persist?.();
  const root = await navigator.storage.getDirectory();
  const fh = await root.getFileHandle(MODEL_NAME, {create: true});
  const res = await fetch(url);
  if (!res.ok) throw new Error(`모델을 받지 못했습니다 (HTTP ${res.status}).`);
  const w = await fh.createWritable(), reader = res.body.getReader();
  let got = 0;
  for (;;) {
    const {done, value} = await reader.read();
    if (done) break;
    await w.write(value);
    got += value.length;
    onProgress(got / MODEL_SIZE);
  }
  await w.close();
  const f = await fh.getFile();
  if (f.size !== MODEL_SIZE) throw new Error('받은 모델 파일의 크기가 다릅니다. 다시 시도해 주세요.');
  return f;
}

// magnitudes scaled to 0..255 by the max (s), mean |v| (m) and norm (n), as server.py's payload
function vec(v) {
  const n = v.length, q = new Uint8Array(n);
  let s = 0, a = 0, ss = 0;
  for (let i = 0; i < n; i++) { const x = Math.abs(v[i]); if (x > s) s = x; a += x; ss += x * x; }
  const k = 255 / (s || 1);
  for (let i = 0; i < n; i++) q[i] = Math.round(Math.abs(v[i]) * k);
  return {q, s: s || 1, m: a / n, n: Math.sqrt(ss)};
}
const norm = v => { let s = 0; for (let i = 0; i < v.length; i++) s += v[i] * v[i]; return Math.sqrt(s); };

function softmax(x) {
  let m = -Infinity;
  for (const v of x) m = Math.max(m, v);
  const p = new Float64Array(x.length);
  let s = 0;
  for (let i = 0; i < x.length; i++) s += p[i] = Math.exp(x[i] - m);
  for (let i = 0; i < x.length; i++) p[i] /= s;
  return p;
}

function topIdx(x, k, sign = 1) {  // indices of the k largest (sign 1) or smallest (sign -1) entries
  const idx = [];
  for (let i = 0; i < x.length; i++) {
    const v = sign * x[i];
    if (idx.length < k) { idx.push(i); if (idx.length === k) idx.sort((a, b) => sign * x[b] - sign * x[a]); continue; }
    if (v <= sign * x[idx[k - 1]]) continue;
    let j = k - 1;
    while (j > 0 && v > sign * x[idx[j - 1]]) { idx[j] = idx[j - 1]; j--; }
    idx[j] = i;
  }
  return idx.length < k ? idx.sort((a, b) => sign * x[b] - sign * x[a]) : idx;
}

export class Session {
  // emit: delivers one server-style message object to the page
  constructor(emit) {
    this.emit = emit;
    this.busy = false; this.stopFlag = false; this.layer = 0; this.last = null;
    this.cache = [];  // token ids whose K / V are in the cache, in order
    this.groups = new Map();
  }

  async boot(blob, onProgress) {
    if (blob.size !== MODEL_SIZE) throw new Error(`${MODEL_NAME}(${MODEL_SIZE.toLocaleString()} 바이트, google/gemma-4-E2B-it-qat-q4_0-gguf)만 지원합니다.`);
    const [g, struct, corpus, tokens] = await Promise.all([readGGUF(blob), fetch('data/structure.json').then(r => r.json()),
      fetch('data/corpus.json').then(r => r.json()), fetch('data/corpus.bin').then(r => r.arrayBuffer())]);
    this.tok = new Tokenizer(g.meta);
    this.meta = g.meta;
    this.m = await Model.load(blob, g, onProgress);
    this.corpus = corpus; this.corpusTokens = new Uint32Array(tokens);
    this.eog = new Set([this.tok.eos, this.tok.ids.get('<turn|>'), this.tok.ids.get('<|tool_response>')]);
    // one decode, as the server's start-up does
    await this.m.step(this.tok.bos, 0);
    this.cache = [this.tok.bos];
    this.emit(struct);
    this.emit(this.corpusStatus());
  }

  corpusStatus() {
    const n = this.corpus.docs.length;
    return {type: 'corpus', done: n, total: n, tokens: this.corpus.tokens, error: null};
  }

  handle(data) {
    const err = e => this.emit({type: 'nerror', message: `${e.name}: ${e.message}`});
    const n = [data.kind || 'mlp', +data.layer, +data.index];
    switch (data.type) {
      case 'chat':
        if (this.busy) this.emit({type: 'error', message: '이미 생성 중입니다.'});
        else this.chat(data.messages).catch(e => this.emit({type: 'error', message: `${e.name}: ${e.message}`}));
        break;
      case 'stop': this.stopFlag = true; break;
      case 'select':
        this.layer = +data.layer;
        if (this.layer >= 0) this.refresh().catch(console.error);
        break;
      case 'neuron': this.neuron(...n).then(this.emit, err); break;
      case 'nprobe': this.probe(...n, data.mode, +data.k || 0).then(this.emit, err); break;
      case 'nsteer': this.steer(...n, data.mode, +data.k || 0, data.messages).then(this.emit, err); break;
    }
  }

  // evaluate tokens from the first position where they differ from the cache; returns the last read-out
  async feed(tokens, {iv = null, readLast = true} = {}) {
    let p = 0;
    while (p < this.cache.length && p < tokens.length - 1 && this.cache[p] === tokens[p]) p++;
    this.cache.length = p;
    let out = null;
    this.m.prefetch(tokens.slice(p));
    for (; p < tokens.length; p++) {
      const read = readLast && p === tokens.length - 1;
      out = await this.m.step(tokens[p], p, {read, detail: read ? this.layer : -1, iv});
      this.cache.push(tokens[p]);
    }
    return out;
  }

  // top-k -> top-p -> temperature -> draw, as llama.cpp's sampler chain; ids: the top-k logits, best first
  sample(logits, ids) {
    const M = this.meta, topP = M['general.sampling.top_p'], temp = M['general.sampling.temp'];
    const p = softmax(ids.map(i => logits[i]));
    let n = 0, c = 0;
    while (n < ids.length && (c < topP || n === 0)) c += p[n++];
    const q = softmax(ids.slice(0, n).map(i => logits[i] / temp));
    let r = Math.random();
    for (let j = 0; j < n; j++) { r -= q[j]; if (r <= 0) return ids[j]; }
    return ids[n - 1];
  }

  logits(out) { return out.subarray(this.m.LOGITS, this.m.LOGITS + this.m.nVocab); }

  async chat(messages) {
    this.busy = true; this.stopFlag = false;
    try {
      const tokens = this.tok.encode(chatPrompt(messages));
      if (tokens.length >= N_CTX - 1) throw new Error(`대화가 컨텍스트 ${N_CTX} 토큰을 넘었습니다. 새 대화를 시작하세요.`);
      let out = await this.feed(tokens), prev = tokens[tokens.length - 1];
      const dec = new TextDecoder();
      for (;;) {
        const lg = this.logits(out), ids = topIdx(lg, this.meta['general.sampling.top_k']), tok = this.sample(lg, ids);
        if (this.eog.has(tok)) break;
        const bytes = this.tok.pieceBytes(tok, false);
        this.emit({type: 'tok', text: dec.decode(bytes, {stream: true}), piece: this.tok.piece(tok, false), input: this.tok.piece(prev, true),
                   frame: this.payload(out, ids)});
        this.last = {pos: this.cache.length - 1, tok: prev};
        prev = tok;
        if (this.stopFlag || this.cache.length >= N_CTX - 1) break;
        out = await this.m.step(tok, this.cache.length, {read: true, detail: this.layer});
        this.cache.push(tok);
      }
      this.emit({type: 'done'});
    } finally {
      this.busy = false;
    }
  }

  // re-run the last shown decode so a newly selected layer gets all of its modules
  async refresh() {
    if (this.busy || !this.last) return;
    this.busy = true;
    try {
      const out = await this.m.step(this.last.tok, this.last.pos, {read: true, detail: this.layer});
      this.emit({type: 'frame', input: this.tok.piece(this.last.tok, true), frame: this.payload(out)});
    } finally {
      this.busy = false;
    }
  }

  payload(out, ids = topIdx(this.logits(out), 5)) {
    const m = this.m, L = m.nLayer, sub = (o, n) => out.subarray(o, o + n);
    const mlp = [], attn = [], resid = [];
    for (let l = 0; l < L; l++) {
      mlp.push(vec(sub(m.mlpOff[l], m.nFF[l])));
      const hd = m.hd[l];
      attn.push(Array.from({length: m.nHead}, (_, h) => +norm(sub(m.headsOff[l] + h * hd, hd)).toFixed(3)));
      resid.push(vec(sub(m.residOff[l], m.nEmbd)));
    }
    const lg = this.logits(out), mx = out[m.LSE], z = out[m.LSE + 1];  // softmax normalizer, from the GPU
    const top = ids.slice(0, 5).map(t => [this.tok.piece(t, true), Math.exp(lg[t] - mx) / z]);
    const v = {}, D = this.layer;
    if (D >= 0 && D < L) {
      for (const [k, n] of Object.entries(m.body[D].keys)) v[k] = vec(sub(m.detailOff[k], n));
      v.heads = vec(sub(m.headsOff[D], m.nHead * m.hd[D]));
    }
    return {mlp, attn, resid, top, emb: vec(sub(m.EMB, m.nEmbd)), final: vec(sub(m.FINAL, m.nEmbd)), detail: {layer: D, v}};
  }

  // ---- single-neuron interpretation ----

  async group(key) {
    if (!this.groups.has(key)) {
      const buf = await (await fetch(`data/index/${key}.bin`)).arrayBuffer(), n = this.corpus.groups[key], K = this.corpus.k;
      this.groups.set(key, {n, K, val: new Uint16Array(buf, 0, n * 2 * K), pos: new Uint16Array(buf, n * 4 * K, n * 2 * K),
                            stats: new Float32Array(buf, n * 8 * K, n * 4)});
    }
    return this.groups.get(key);
  }

  async stats(kind, l, i) {
    const G = await this.group(`${kind}${l}`), K = G.K, s = G.stats.subarray(4 * i, 4 * i + 4);
    const poles = [0, 1].map(pole => {
      const out = [];
      for (let k = 0; k < K; k++) {
        const j = (i * 2 + pole) * K + k;
        if (G.pos[j] !== 0xFFFF) out.push([f16(G.val[j]), G.pos[j]]);
      }
      return out;
    });
    return {count: this.corpus.tokens, mean: s[0], std: s[1], max: s[2], min: s[3], poles};
  }

  async neuron(kind, l, i) {
    const info = await this.stats(kind, l, i), {starts, docs} = this.corpus, T = this.corpusTokens;
    for (const pole of info.poles) {  // peak token with up to 12 tokens before and 3 after in its document
      pole.forEach(([v, g], e) => {
        let d = 0;
        while (d + 1 < starts.length && starts[d + 1] <= g) d++;
        const a = Math.max(starts[d], g - 12), b = Math.min(d + 1 < starts.length ? starts[d + 1] : T.length, g + 4), ctx = [];
        for (let p = a; p < b; p++) ctx.push([this.tok.piece(T[p], true), p === g]);
        pole[e] = {v, doc: docs[d].title, url: docs[d].url, ctx};
      });
    }
    const s = await this.m.pushScores(kind, l, i);
    const pick = ids => ids.map(t => [this.tok.piece(t, true), s[t]]).filter(([p]) => !SPECIAL_RE.test(p)).slice(0, 12);
    return {type: 'neuron', kind, layer: l, index: i, corpus: this.corpusStatus(), stats: info,
            push: {up: pick(topIdx(s, 200)), down: pick(topIdx(s, 200, -1))}};
  }

  async iv(kind, l, i, mode, k) {
    if (mode === 'zero') return {kind, layer: l, index: i, mode};
    const st = await this.stats(kind, l, i);
    return {kind, layer: l, index: i, mode, delta: k * Math.max(Math.abs(st.max), Math.abs(st.min))};
  }

  // next-token distribution at the last shown position, neuron untouched vs intervened
  async probe(kind, l, i, mode, k) {
    if (this.busy || !this.last) throw new Error('생성이 끝난 뒤, 대화가 하나 이상 있을 때 비교할 수 있습니다.');
    this.busy = true;
    try {
      const {pos, tok} = this.last, probs = [];
      for (const iv of [null, await this.iv(kind, l, i, mode, k)]) probs.push(softmax(this.logits(await this.m.step(tok, pos, {read: true, iv}))));
      this.cache.length = pos;  // the intervened position must not be reused by the next chat
      const [base, alt] = probs, ids = [...new Set([...topIdx(base, 5), ...topIdx(alt, 5)])];
      return {type: 'nprobe', input: this.tok.piece(tok, true), rows: ids.map(t => [this.tok.piece(t, true), base[t], alt[t]])};
    } finally {
      this.busy = false;
    }
  }

  // greedy continuation of the conversation without and with the intervention
  async steer(kind, l, i, mode, k, messages, n = 48) {
    if (this.busy) throw new Error('이미 생성 중입니다.');
    this.busy = true;
    try {
      const tokens = this.tok.encode(chatPrompt(messages)), orig = this.cache.slice(), texts = [];
      let P = 0;  // both runs (and the restore) recompute from where the prompt leaves the session's cache
      while (P < orig.length && P < tokens.length - 1 && orig[P] === tokens[P]) P++;
      for (const iv of [null, await this.iv(kind, l, i, mode, k)]) {
        this.cache.length = P;
        let out = await this.feed(tokens, {iv});
        const gen = [];
        for (;;) {
          const lg = this.logits(out);
          const t = topIdx(lg, 1)[0];
          if (this.eog.has(t) || gen.length === n) break;
          gen.push(t);
          out = await this.m.step(t, this.cache.length, {read: true, iv});
          this.cache.push(t);
        }
        texts.push(new TextDecoder().decode(new Uint8Array(gen.flatMap(t => [...this.tok.pieceBytes(t, false)]))));
      }
      this.cache.length = P;
      await this.feed(orig, {readLast: false});  // put the session's own tokens back into the KV cache
      return {type: 'nsteer', base: texts[0], alt: texts[1]};
    } finally {
      this.busy = false;
    }
  }
}
