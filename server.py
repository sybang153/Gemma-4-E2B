"""Chat with Gemma 4 E2B (GGUF Q4_0, llama.cpp) while watching its full structure and live activations.

Usage: python server.py  ->  open http://127.0.0.1:8800
"""
import asyncio
import base64
import bisect
import codecs
import contextlib
import ctypes
import json
import os
import re
import threading
from collections import Counter
from ctypes import POINTER, c_char, c_char_p, c_float, c_int, c_int32, c_int64, c_size_t, c_void_p

import numpy as np
import requests
from aiohttp import web
from gguf import GGUFReader, GGUFValueType
from gguf.quants import dequantize
from huggingface_hub import hf_hub_download
import llama_cpp.llama_cpp as lc
from llama_cpp import Llama
from llama_cpp.llama_chat_format import Jinja2ChatFormatter

REPO, FILE = "google/gemma-4-E2B-it-qat-q4_0-gguf", "gemma-4-E2B_q4_0-it.gguf"
HERE = os.path.dirname(os.path.abspath(__file__))
PORT = 8800
N_CTX = 8192
N_THREADS = 8  # fastest decode on the Core Ultra 5 225H (4..14 measured)
N_UBATCH = 128  # the compute buffer reserves n_ubatch x vocab logits (512 -> 512 MiB)
LAYOUT_OPS = {"NONE", "VIEW", "RESHAPE", "PERMUTE", "TRANSPOSE"}  # pure re-indexing, hidden in op lists
TAP_RE = re.compile(rb"blk\.(\d+)\.(ffn_down|attn_output|proj)\.weight$")
LAYER_RE = re.compile(r"-(\d+)(?: |$)")
NAMED_RE = re.compile(rb"([A-Za-z_]+)-(\d+)")
# outputs of these named graph nodes are shown as modules of the selected layer
DETAIL = {b"attn_norm", b"Qcur_pos", b"Kcur_pos", b"Vcur_normed", b"attn_post_norm", b"attn_out",
          b"ffn_norm", b"ffn_gate", b"ffn_up", b"ffn_post_norm", b"pe_in", b"per_layer_embd_out"}
PER_HEAD = {"Qcur_pos", "Kcur_pos", "Vcur_normed"}  # [head_dim, head, token] tensors
SPECIAL_RE = re.compile(r"<[^<>]*>")  # control / unused vocabulary entries, skipped in token lists
# interpretable neuron kinds: the matmul that reads the neurons, and the norm gain on its output before the residual
NEURONS = {"mlp": ("ffn_down", "post_ffw_norm"), "ple": ("proj", "post_norm")}
# neuron-interpretation corpus: the start of N_DOCS articles per language of wikimedia/wikipedia, in runs of 10
# spread evenly over the dump, fetched once through the Hugging Face datasets-server
ROWS = "https://datasets-server.huggingface.co/rows"
WIKI = {"ko": "20231101.ko", "en": "20231101.en"}
N_DOCS, DOC_TOKENS = 80, 200
CACHE = os.path.join(HERE, "cache")


class Tensor(ctypes.Structure):
    pass


# struct ggml_tensor from ggml.h at llama.cpp 0c1e570 (vendored by llama-cpp-python 0.3.36)
Tensor._fields_ = [("type", c_int), ("buffer", c_void_p), ("ne", c_int64 * 4), ("nb", c_size_t * 4), ("op", c_int),
                   ("op_params", c_int32 * 16), ("flags", c_int32), ("src", POINTER(Tensor) * 10),
                   ("view_src", POINTER(Tensor)), ("view_offs", c_size_t), ("data", c_void_p),
                   ("name", c_char * 64), ("extra", c_void_p), ("padding", c_char * 8)]

ggml = ctypes.CDLL(os.path.join(os.path.dirname(lc.__file__), "lib", "ggml-base.dll"))
ggml.ggml_op_desc.argtypes = [POINTER(Tensor)]
ggml.ggml_op_desc.restype = c_char_p
ggml.ggml_op_name.argtypes = [c_int]
ggml.ggml_op_name.restype = c_char_p
ggml.ggml_type_name.argtypes = [c_int]
ggml.ggml_type_name.restype = c_char_p
ggml.ggml_backend_tensor_get.argtypes = [POINTER(Tensor), c_void_p, c_size_t, c_size_t]
ggml.ggml_backend_tensor_set.argtypes = [POINTER(Tensor), c_void_p, c_size_t, c_size_t]
MUL_MAT, MUL, ADD, UNARY = (next(i for i in range(200) if ggml.ggml_op_name(i) == n) for n in (b"MUL_MAT", b"MUL", b"ADD", b"UNARY"))


def last_row(t, axis=1):
    """The last token's slice of a contiguous F32 activation; the token axis is 1, or 2 for per-head tensors."""
    out = np.empty(int(np.prod(t.ne[:axis])), np.float32)
    ggml.ggml_backend_tensor_get(ctypes.byref(t), out.ctypes.data, t.nb[axis] * (int(np.prod(t.ne[axis:])) - 1), out.nbytes)
    return out


def all_rows(t):
    """A contiguous F32 activation as (tokens, features)."""
    out = np.empty((int(np.prod(t.ne[1:])), t.ne[0]), np.float32)
    ggml.ggml_backend_tensor_get(ctypes.byref(t), out.ctypes.data, 0, out.nbytes)
    return out


class NeuronIndex:
    """For every neuron of each group (key "mlp3" = layer 3's MLP neurons, "ple3" its PLE neurons), over all corpus
    tokens: the K strongest activations of each pole (value, corpus position) and count / sum / sum of squares / max / min.
    Tokens go into a buffer that is merged when it reaches LIMIT tokens or before a query.
    `lock` guards the buffers (taken by the evaluating thread), `mlock` the merged results."""
    K, LIMIT = 16, 256
    SAVED = ("val", "pos", "sum", "sq", "max", "min")

    def __init__(self):
        self.layers, self.lock, self.mlock = {}, threading.Lock(), threading.Lock()

    def update(self, l, V, base):
        """V: (tokens, neurons) activations of group l at corpus positions base, base + 1, ..."""
        T, n = V.shape
        with self.lock:
            if l not in self.layers:
                self.layers[l] = {"val": np.full((2, n, self.K), -np.inf, np.float32), "pos": np.full((2, n, self.K), -1, np.int32),
                                  "kth": np.full((2, n), -np.inf, np.float32),  # smallest kept value: the bar to enter
                                  "buf": [], "bufpos": [], "nbuf": 0, "sum": np.zeros(n), "sq": np.zeros(n),
                                  "max": np.full(n, -np.inf, np.float32), "min": np.full(n, np.inf, np.float32), "count": 0}
            d = self.layers[l]
            d["buf"].append(V)
            d["bufpos"].append(base + np.arange(T))
            d["nbuf"] += T
            full = d["nbuf"] >= self.LIMIT
        if full:
            self._merge(d)

    def flush(self):
        for d in list(self.layers.values()):
            self._merge(d)

    def save(self, path, **extra):
        self.flush()
        np.savez(path, keys=list(self.layers), count=[d["count"] for d in self.layers.values()], **extra,
                 **{f"{k}-{l}": d[k] for l, d in self.layers.items() for k in self.SAVED})

    def load(self, path):
        """Restore a saved index; returns the archive for the extra arrays."""
        z = np.load(path)
        for l, c in zip(z["keys"].tolist(), z["count"]):
            d = {k: z[f"{k}-{l}"] for k in self.SAVED}
            self.layers[l] = {**d, "kth": d["val"].min(2), "buf": [], "bufpos": [], "nbuf": 0, "count": int(c)}
        return z

    def _merge(self, d):
        with self.mlock:
            with self.lock:
                bufs, poss, d["buf"], d["bufpos"], d["nbuf"] = d["buf"], d["bufpos"], [], [], 0
            if not bufs:
                return
            V, P = np.concatenate(bufs), np.concatenate(poss)
            d["sum"] += V.sum(0, dtype=np.float64)
            d["sq"] += np.square(V).sum(0, dtype=np.float64)
            np.maximum(d["max"], V.max(0), out=d["max"])
            np.minimum(d["min"], V.min(0), out=d["min"])
            d["count"] += len(P)
            K = self.K
            for pole, sign in ((0, 1), (1, -1)):  # pole 1 keeps the negative pole negated, so both select the largest
                val, pos, kth = d["val"][pole], d["pos"][pole], d["kth"][pole]
                # candidates: tokens that beat their neuron's K-th value, grouped by neuron
                t, j = np.nonzero(V > kth if sign > 0 else V < -kth)
                if not len(j):
                    continue
                o = np.argsort(j.astype(np.uint16), kind="stable")  # 16-bit keys: numpy uses a radix sort
                t, j = t[o], j[o]
                first = np.flatnonzero(np.r_[True, j[1:] != j[:-1]])
                cnt = np.diff(np.r_[first, len(j)])
                grp = np.repeat(np.arange(len(first)), cnt)
                col, x = np.arange(len(j)) - first[grp], V[t, j] * sign
                for g in (cnt <= K, cnt > K):  # padded to K columns, and the few busy neurons to their own max
                    if not g.any():
                        continue
                    rows, w, m = j[first[g]], int(cnt[g].max()), g[grp]
                    cv, cp = np.full((len(rows), w + K), -np.inf, np.float32), np.full((len(rows), w + K), -1, np.int32)
                    r = (np.cumsum(g) - 1)[grp[m]]
                    cv[r, col[m]], cp[r, col[m]] = x[m], P[t[m]]
                    cv[:, w:], cp[:, w:] = val[rows], pos[rows]
                    sel = np.argpartition(cv, -K, axis=1)[:, -K:]
                    val[rows], pos[rows] = np.take_along_axis(cv, sel, 1), np.take_along_axis(cp, sel, 1)
                    kth[rows] = val[rows].min(1)

    def neuron(self, l, i):
        d = self.layers.get(l)
        if not d:
            return None
        self._merge(d)
        if not d["count"]:
            return None
        n, mean = d["count"], d["sum"][i] / d["count"]
        poles = []
        for pole, sign in ((0, 1), (1, -1)):
            v, p = d["val"][pole, i], d["pos"][pole, i]
            poles.append([(float(sign * v[k]), int(p[k])) for k in np.argsort(-v) if np.isfinite(v[k])])
        return {"count": n, "mean": float(mean), "std": float(np.sqrt(max(d["sq"][i] / n - mean ** 2, 0))),
                "max": float(d["max"][i]), "min": float(d["min"][i]), "poles": poles}


def is_weight(p, name):
    """p (a src pointer) is a matmul with the named weight."""
    return bool(p) and p.contents.op == MUL_MAT and p.contents.src[0].contents.name == name


def is_ple_mul(t, l):
    """t is layer l's GELU(inp_gate · x) ⊙ per-layer embedding (unnamed), the input of its proj matmul."""
    return t.op == MUL and bool(t.src[0]) and t.src[0].contents.op == UNARY and is_weight(t.src[0].contents.src[0], b"blk.%d.inp_gate.weight" % l)


def shape(t):
    ne = list(t.ne)
    while len(ne) > 1 and ne[-1] == 1:
        ne.pop()
    return ne


class Tap:
    """ggml scheduler eval callback: records the compute graph once, then copies activations at fixed points.

    Every layer: the input of the ffn_down matmul (MLP neurons), the input of the attn_output matmul (concatenated
    head outputs), l_out (residual stream). Model-wide: the scaled embedding, the final norm and the logits.
    The selected layer (`layer`, -1 = none) also gets every module output shown in the block diagram.
    While `indexing`, every MLP and PLE neuron activation (the inputs of ffn_down and proj) of every evaluated token
    goes into `index` at corpus position `g`.
    `intervene` = (node test, neuron, fn) rewrites that neuron before the next matmul reads it (ablation / steering).
    """

    def __init__(self):
        self.graph, self._rec = None, []
        self.cur, self.frames = {}, []  # one finished frame per llama_decode
        self.layer = self.active = 0  # active: the selection, frozen for the decode in progress
        self.index, self.indexing, self.g, self.base = NeuronIndex(), False, 0, 0
        self.intervene = None
        self.c_fn = lc.ggml_backend_sched_eval_callback(self._cb)

    def _point(self, t):
        """(kind, layer, key) for a node to copy, else None."""
        name, l = t.name, self.active
        if self.intervene and self.intervene[0](t):
            return "intervene", -1, None
        if t.op == MUL_MAT and (m := TAP_RE.match(t.src[0].contents.name)):
            il, w = int(m.group(1)), m.group(2)
            return ("mlp", il, None) if w == b"ffn_down" else ("attn", il, None) if w == b"attn_output" else \
                ("ple", il, None) if self.indexing or il == l else None
        if m := NAMED_RE.fullmatch(name):
            base, il = m.group(1), int(m.group(2))
            return ("resid", il, None) if base == b"l_out" else ("detail", il, base.decode()) if il == l and base in DETAIL else None
        if name == b"result_output":
            return "logits", -1, None
        if name in (b"inp_scaled", b"result_norm"):
            return "global", -1, name.decode()
        if l < 0:
            return None
        # unnamed nodes of the per-layer-embedding branch, recognised through their inputs
        if t.op == UNARY and is_weight(t.src[0], b"blk.%d.inp_gate.weight" % l):
            return "detail", l, "ple_gelu"
        if is_ple_mul(t, l):
            return "detail", l, "ple_mul"
        if t.op == ADD and t.src[1] and t.src[1].contents.name == b"per_layer_embd_out-%d" % l:
            return "detail", l, "resid_ple"
        return None

    def _cb(self, tp, ask, _):
        t = ctypes.cast(tp, POINTER(Tensor)).contents
        pt = self._point(t)
        if ask:
            if self.graph is None:
                srcs = [t.src[i] for i in range(10) if t.src[i]]
                self._rec.append({
                    "addr": tp, "name": t.name.decode(), "op": ggml.ggml_op_desc(t).decode(), "shape": shape(t),
                    "srcs": [{"addr": ctypes.addressof(s.contents), "name": s.contents.name.decode(),
                              "type": ggml.ggml_type_name(s.contents.type).decode(), "shape": shape(s.contents)} for s in srcs]})
            return bool(pt)
        kind, il, key = pt
        detail = self.cur.setdefault("detail", {})
        if kind == "intervene":
            _, i, fn = self.intervene
            x = c_float()
            for k in range(int(np.prod(t.ne[1:]))):  # every token of the ubatch
                ggml.ggml_backend_tensor_get(ctypes.byref(t), ctypes.byref(x), k * t.nb[1] + i * 4, 4)
                x.value = fn(x.value)
                ggml.ggml_backend_tensor_set(ctypes.byref(t), ctypes.byref(x), k * t.nb[1] + i * 4, 4)
        elif kind == "logits":
            self.cur["logits"] = last_row(t)
            self.cur["detail_layer"] = self.active
            self.frames.append(self.cur)
            self.cur = {}
            self.active = self.layer
            if self.graph is None:
                self.graph = self._rec
        elif kind == "global":
            self.cur[key] = last_row(t)
        elif kind == "detail":
            detail[key] = last_row(t, 2 if key in PER_HEAD else 1)
            if key == "ple_mul":  # its second input is this layer's slice of the per-layer embedding
                detail["ple_in"] = last_row(t.src[1].contents)
        elif kind == "mlp" and self.indexing:
            V = all_rows(t.src[1].contents)
            if il == 0:  # first layer of a ubatch: its tokens get the next corpus positions
                self.base, self.g = self.g, self.g + len(V)
            self.index.update(f"mlp{il}", V, self.base)
            v = self.cur.setdefault(kind, {})[il] = V[-1].copy()
            if il == self.active:
                detail["ffn_out"] = last_row(t)
        elif kind == "ple":  # layer il's proj matmul; its input rows are the PLE neurons (after the MLP: same base)
            if self.indexing:
                self.index.update(f"ple{il}", all_rows(t.src[1].contents), self.base)
            if il == self.active:
                detail["ple_proj"] = last_row(t)
        else:
            v = last_row(t if kind == "resid" else t.src[1].contents)
            self.cur.setdefault(kind, {})[il] = v
            if il == self.active and kind != "resid":  # the matmul's input is a module output, so is its output
                detail.update({"heads": v, "o_proj": last_row(t)} if kind == "attn" else {"ffn_out": last_row(t)})
        return True


def q8(v):
    """Magnitudes scaled to 0..255 by the vector's max, base64 encoded."""
    a = np.abs(v)
    s = float(a.max()) or 1.0
    return base64.b64encode(np.round(a / s * 255).astype(np.uint8).tobytes()).decode(), s


def top5(logits):
    """Top-5 token ids and softmax probabilities; logits is overwritten."""
    # the 5 largest values lie in the 5 blocks with the largest maxima
    blocks = logits.reshape(-1, 256).max(axis=1)
    cand = (np.argpartition(blocks, -5)[-5:, None] * 256 + np.arange(256)).ravel()
    top = cand[np.argsort(-logits[cand])[:5]]
    np.subtract(logits, blocks.max(), out=logits)
    np.exp(logits, out=logits)
    return top, logits[top] / logits.sum(dtype=np.float64)


def show(v):
    return v[:300] + f" … (총 {len(v)}자)" if isinstance(v, str) and len(v) > 300 else v


def wiki_docs():
    """The corpus articles (lang, title, url, first 3000 characters), downloaded on the first run."""
    path = os.path.join(CACHE, "wiki.json")
    if not os.path.exists(path):
        langs = []
        for lang, config in WIKI.items():
            def rows(offset):
                r = requests.get(ROWS, params={"dataset": "wikimedia/wikipedia", "config": config, "split": "train",
                                               "offset": offset, "length": 10}, timeout=60)
                r.raise_for_status()
                return r.json()
            first = rows(0)
            runs = [first] + [rows(k * first["num_rows_total"] // (N_DOCS // 10)) for k in range(1, N_DOCS // 10)]
            langs.append([{"lang": lang, "title": r["row"]["title"], "url": r["row"]["url"], "text": r["row"]["text"][:3000]}
                          for run in runs for r in run["rows"]])
        docs = [d for pair in zip(*langs) for d in pair]  # alternate languages: fewer index entries change per merge
        os.makedirs(CACHE, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(docs, f, ensure_ascii=False)
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class Engine:
    def __init__(self, path):
        self.path, self.tap, self.lock = path, Tap(), threading.Lock()
        self.busy, self.last = False, None  # last: (KV position, token) of the input of the last frame sent
        orig = lc.llama_context_default_params

        def with_tap():
            p = orig()
            p.cb_eval = self.tap.c_fn
            return p

        lc.llama_context_default_params = with_tap  # Llama() builds its context params through this function
        try:
            self.llm = Llama(model_path=path, n_ctx=N_CTX, n_threads=N_THREADS, n_ubatch=N_UBATCH, flash_attn=True, verbose=False)
        finally:
            lc.llama_context_default_params = orig
        self.llm.eval([self.llm.token_bos()])  # one decode records the graph
        self.tap.frames.clear()
        # corpus: token ids of all indexed documents, in order; starts: each document's first position
        self.corpus, self.docs, self.starts, self.scan, self.listeners = [], [], [], [0, None], set()
        self.scan_error = None

        reader = GGUFReader(path)
        self.tensors, self.meta = reader.tensors, {}
        self.weight = {t.name: t for t in self.tensors}
        self._down, self._unembed, self.unembed_lock = (None, None), None, threading.Lock()
        for k, f in reader.fields.items():  # vocab-sized arrays are summarized instead of converted
            big = f.types[0] == GGUFValueType.ARRAY and len(f.data) > 64
            self.meta[k] = f"[{len(f.data)}개] {f.contents(slice(0, 3))} …" if big else f.contents()
        arch = self.meta["general.architecture"]
        self.hp = lambda k: self.meta[f"{arch}.{k}"]
        self.n_layer, self.n_head = self.hp("block_count"), self.hp("attention.head_count")
        self.tok = lambda i: self.llm.detokenize([i], special=True).decode("utf-8", "replace")
        self.fmt = Jinja2ChatFormatter(self.meta["tokenizer.chat_template"], eos_token=self.tok(self.llm.token_eos()),
                                       bos_token=self.tok(self.llm.token_bos()))

    def structure(self):
        graph, n_layer = self.tap.graph, self.n_layer
        nodes = {n["addr"]: n for n in graph}
        last = max(i for i, n in enumerate(graph) if n["name"] == f"l_out-{n_layer - 1}")
        stage, ops = {}, {"in": [], "out": [], **{i: [] for i in range(n_layer)}}
        n_layout, kv_from, used, prev = Counter(), {}, set(), -1

        def resolve(s):  # follow layout ops back to the producing compute op or leaf
            while s["addr"] in nodes and nodes[s["addr"]]["op"] in LAYOUT_OPS:
                s = nodes[s["addr"]]["srcs"][0]
            return s

        for i, n in enumerate(graph):
            blk = [int(m.group(1)) for s in n["srcs"] if (m := re.match(r"blk\.(\d+)\.", s["name"]))]
            m = LAYER_RE.search(n["name"])
            prod = [stage[s["addr"]] for s in n["srcs"] if s["addr"] in stage]
            # stage: name suffix > block weight > latest producer > position (nodes fed only by leaves, e.g. cache views)
            st = -2 if i > last else int(m.group(1)) if m else blk[0] if blk else max(prod) if prod else prev
            stage[n["addr"]], prev = st, st
            key = {-1: "in", -2: "out"}.get(st, st)
            used.update(s["name"] for s in n["srcs"] if s["addr"] not in nodes)
            if n["op"] in LAYOUT_OPS:
                n_layout[key] += 1
                continue
            m = n["op"] == "MUL_MAT" and TAP_RE.match(n["srcs"][0]["name"].encode())
            tap = {b"ffn_down": "mlp", b"attn_output": "attn"}.get(m.group(2)) if m else "resid" if n["name"].startswith("l_out-") else None
            ins = []
            for s in map(resolve, n["srcs"]):
                leaf = s["addr"] not in nodes
                if leaf and st >= 0 and (c := re.match(r"cache_k_l(\d+)$", s["name"])):
                    kv_from[st] = int(c.group(1))
                ins.append({"name": s["name"], "leaf": leaf, **({"type": s["type"], "shape": s["shape"]} if leaf else {})})
            ops[key].append({"op": n["op"], "name": n["name"], "shape": n["shape"], "inputs": ins, "tap": tap})

        types = {}
        for t in self.tensors:
            d = types.setdefault(t.tensor_type.name, {"n": 0, "bytes": 0})
            d["n"] += 1
            d["bytes"] += int(t.n_bytes)
        swa = self.hp("attention.sliding_window_pattern")
        stages = [{"id": "in", "title": "토큰 임베딩 · 층별 임베딩(PLE) 준비"}] + \
                 [{"id": i} for i in range(n_layer)] + [{"id": "out", "title": "최종 norm · LM head(임베딩 공유) · logit softcap"}]
        for s in stages:
            s["ops"], s["n_layout_ops"] = ops[s["id"]], n_layout[s["id"]]
        return {
            "summary": {
                "arch": self.meta["general.architecture"], "n_layer": n_layer, "n_embd": self.hp("embedding_length"),
                "n_head": self.n_head, "n_head_kv": self.hp("attention.head_count_kv"), "n_nodes": len(graph),
                "n_params": int(sum(int(t.n_elements) for t in self.tensors)), "file_bytes": os.path.getsize(self.path), "types": types,
            },
            "layers": [{"attn": f"sliding {self.hp('attention.sliding_window')}" if swa[i] else "global",
                        "kv_shared": kv_from.get(i, i) != i, "kv_from": kv_from.get(i, i),
                        "n_ff": self.hp("feed_forward_length")[i], "n_ple": self.hp("embedding_length_per_layer_input"),
                        "n_head": self.n_head} for i in range(n_layer)],
            "stages": stages,
            "meta": {k: show(v) for k, v in self.meta.items()},
            "tensors": [{"name": t.name, "shape": [int(x) for x in t.shape], "type": t.tensor_type.name,
                         "bytes": int(t.n_bytes), "used": t.name in used} for t in self.tensors],
        }

    def payload(self, f):
        L = self.n_layer
        mlp, attn, resid = [None] * L, [None] * L, [None] * L
        for il, v in f.get("mlp", {}).items():
            q, s = q8(v)
            mlp[il] = {"q": q, "s": s, "m": float(np.abs(v).mean()), "n": float(np.linalg.norm(v))}
        for il, v in f.get("attn", {}).items():
            attn[il] = [round(float(x), 3) for x in np.linalg.norm(v.reshape(self.n_head, -1), axis=1)]
        vec = lambda v: dict(zip("qs", q8(v)), n=float(np.linalg.norm(v)))  # s (max |a|) turns q back into values
        for il, v in f.get("resid", {}).items():
            resid[il] = vec(v)
        top, p = top5(f["logits"])
        return {"mlp": mlp, "attn": attn, "resid": resid, "top": [[self.tok(int(t)), float(x)] for t, x in zip(top, p)],
                "emb": vec(f["inp_scaled"]), "final": vec(f["result_norm"]),
                "detail": {"layer": f["detail_layer"], "v": {k: vec(v) for k, v in f["detail"].items()}}}

    def chat(self, messages, send, stop):
        with self.lock:
            self.busy = True
            try:
                prompt = self.fmt(messages=messages).prompt
                tokens = self.llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True)
                if len(tokens) >= N_CTX - 1:
                    raise ValueError(f"대화가 컨텍스트 {N_CTX} 토큰을 넘었습니다. 새 대화를 시작하세요.")
                self.tap.frames.clear()
                dec = codecs.getincrementaldecoder("utf-8")(errors="replace")
                prev = tokens[-1]  # the token whose decode produced the next frame
                for tok in self.llm.generate(tokens, temp=self.meta["general.sampling.temp"], top_k=self.meta["general.sampling.top_k"],
                                             top_p=self.meta["general.sampling.top_p"], min_p=0.0):
                    frame = self.tap.frames[-1]
                    self.tap.frames.clear()
                    if lc.llama_vocab_is_eog(self.llm._model.vocab, tok):
                        break
                    piece = self.llm.detokenize([tok])
                    send({"type": "tok", "text": dec.decode(piece), "piece": piece.decode("utf-8", "replace"), "input": self.tok(prev),
                          "frame": self.payload(frame)})
                    self.last = (self.llm.n_tokens - 1, prev)
                    prev = tok
                    if stop.is_set() or self.llm.n_tokens >= N_CTX - 1:
                        break
                send({"type": "done"})
            except Exception as e:
                send({"type": "error", "message": f"{type(e).__name__}: {e}"})
            finally:
                self.busy = False

    def refresh(self, send):
        """Re-run the last shown frame's decode so a newly selected layer gets all of its modules."""
        if self.busy or not self.last:
            return  # a running generation fills the next frame anyway
        with self.lock:
            pos, tok = self.last
            self.llm.n_tokens = pos  # eval() drops the KV cache from here on, then recomputes this token
            self.tap.active = self.tap.layer
            self.tap.frames.clear()
            self.llm.eval([tok])
            send({"type": "frame", "input": self.tok(tok), "frame": self.payload(self.tap.frames[-1])})

    # ---- single-neuron interpretation (dataset examples, output-weight logit lens, ablation / steering) ----

    def build_corpus(self):
        """Index every MLP activation over the Wikipedia corpus, each document from a fresh context, or load the
        index cached by an earlier run. Holds the engine (chat waits) until done."""
        with self.lock:
            self.busy = True
            try:
                self.docs = wiki_docs()
                path = os.path.join(CACHE, f"index-mlp-ple-{len(self.docs)}x{DOC_TOKENS}.npz")
                if os.path.exists(path):
                    z = self.tap.index.load(path)
                    self.corpus, self.starts = z["tokens"].tolist(), z["starts"].tolist()
                    self.scan = [len(self.docs)] * 2
                    return self._notify()
                self.scan = [0, len(self.docs)]
                for d in self.docs:
                    toks = self.llm.tokenize(d["text"].encode("utf-8"), add_bos=False, special=False)[:DOC_TOKENS]
                    self.llm.n_tokens = 0
                    self.llm.eval([self.llm.token_bos()])  # BOS (the attention sink) is not indexed
                    self.starts.append(len(self.corpus))
                    self.corpus += toks
                    self.tap.indexing = True
                    try:
                        self.llm.eval(toks)
                    finally:
                        self.tap.indexing = False
                    self.tap.frames.clear()
                    self.scan[0] += 1
                    self._notify()
                self.tap.index.save(path, tokens=np.array(self.corpus, np.int32), starts=np.array(self.starts, np.int32))
            except Exception as e:  # e.g. offline on the first run: chat still works, the neuron view has no examples
                self.scan_error = f"{type(e).__name__}: {e}"
                self._notify()
                raise
            finally:
                self.busy = False

    def corpus_status(self):
        return {"type": "corpus", "done": self.scan[0], "total": self.scan[1], "tokens": len(self.corpus), "error": self.scan_error}

    def _notify(self):
        for send in list(self.listeners):
            send(self.corpus_status())

    @contextlib.contextmanager
    def _side_run(self, intervene):
        """Evaluate with a neuron intervention (or none)."""
        self.tap.intervene = intervene
        try:
            yield
        finally:
            self.tap.intervene = None

    def _f32(self, name):
        t = self.weight[name]
        return dequantize(t.data, t.tensor_type).astype(np.float32)

    def pushes(self, kind, l, i, k=12):
        """Tokens whose logits the neuron's output weight raises / lowers directly: column i of ffn_down (or proj)
        through the post-FFN (post-PLE) and final norm gains onto the tied unembedding (norm scales and later layers ignored)."""
        mat, gain = NEURONS[kind]
        with self.unembed_lock:
            if self._unembed is None:  # one pass over the Q6_K table (~15 s), kept as int8 rows + scales (~400 MB)
                t = self.weight["token_embd.weight"]
                E, sc = np.empty((t.data.shape[0], self.hp("embedding_length")), np.int8), np.empty(t.data.shape[0], np.float32)
                for a in range(0, len(E), 16384):
                    w = dequantize(t.data[a:a + 16384], t.tensor_type)
                    s = np.abs(w).max(1) / 127
                    E[a:a + 16384], sc[a:a + 16384] = np.round(w / s[:, None]), s
                self._unembed = E, sc
            if self._down[0] != (kind, l):
                self._down = (kind, l), self._f32(f"blk.{l}.{mat}.weight")
        E, sc = self._unembed
        u = self._down[1][:, i] * self._f32(f"blk.{l}.{gain}.weight") * self._f32("output_norm.weight")
        s = np.concatenate([E[a:a + 32768].astype(np.float32) @ u for a in range(0, len(E), 32768)]) * sc

        def pick(ids):
            out = [[self.tok(int(t)), float(s[t])] for t in ids]
            return [x for x in out if not SPECIAL_RE.fullmatch(x[0])][:k]
        order = np.argsort(s)
        return {"up": pick(order[::-1][:200]), "down": pick(order[:200])}

    def neuron(self, kind, l, i):
        """Everything the neuron view shows except the interventions."""
        info = self.tap.index.neuron(f"{kind}{l}", i)
        if info:
            for pole in info["poles"]:  # peak token with up to 12 tokens before and 3 after in its document
                for e, (v, g) in enumerate(pole):
                    d = bisect.bisect_right(self.starts, g) - 1
                    a = max(self.starts[d], g - 12)
                    b = min(self.starts[d + 1] if d + 1 < len(self.starts) else len(self.corpus), g + 4)
                    pole[e] = {"v": v, "doc": self.docs[d]["title"], "url": self.docs[d]["url"],
                               "ctx": [[self.tok(self.corpus[p]), p == g] for p in range(a, b)]}
        return {"type": "neuron", "kind": kind, "layer": l, "index": i,
                "corpus": self.corpus_status(), "stats": info, "push": self.pushes(kind, l, i)}

    def _iv(self, kind, l, i, mode, k):
        """Intervention for the tap: 'zero' ablates, 'add' adds k times the neuron's largest |a| in the corpus."""
        node = (lambda t: t.name == b"ffn_geglu-%d" % l) if kind == "mlp" else (lambda t: is_ple_mul(t, l))
        if mode == "zero":
            return node, i, lambda v: 0.0
        st = self.tap.index.neuron(f"{kind}{l}", i)
        d = k * (max(abs(st["max"]), abs(st["min"])) if st else 1.0)
        return node, i, lambda v: v + d

    def probe(self, kind, l, i, mode, k):
        """Next-token distribution at the last shown position, neuron untouched vs intervened."""
        if self.busy or not self.last:
            raise ValueError("생성이 끝난 뒤, 대화가 하나 이상 있을 때 비교할 수 있습니다.")
        with self.lock:
            pos, tok = self.last
            probs = []
            for iv in (None, self._iv(kind, l, i, mode, k)):
                self.llm.n_tokens = pos
                with self._side_run(iv):
                    self.tap.frames.clear()
                    self.llm.eval([tok])
                    x = self.tap.frames[-1]["logits"].astype(np.float64)
                p = np.exp(x - x.max())
                probs.append(p / p.sum())
            self.llm.n_tokens = pos  # the intervened position must not be reused by the next chat
            self.tap.frames.clear()
        base, alt = probs
        ids = list(dict.fromkeys([*np.argsort(-base)[:5], *np.argsort(-alt)[:5]]))
        return {"type": "nprobe", "input": self.tok(tok),
                "rows": [[self.tok(int(t)), float(base[t]), float(alt[t])] for t in ids]}

    def steer(self, kind, l, i, mode, k, messages, n=48):
        """Greedy continuation of the conversation without and with the intervention."""
        if self.busy:
            raise ValueError("이미 생성 중입니다.")
        with self.lock:
            self.busy = True
            try:
                tokens = self.llm.tokenize(self.fmt(messages=messages).prompt.encode("utf-8"), add_bos=False, special=True)
                n0, orig = self.llm.n_tokens, self.llm.input_ids[:self.llm.n_tokens].copy()
                P = next((j for j, (a, b) in enumerate(zip(orig, tokens[:-1])) if a != b), min(n0, len(tokens) - 1))
                texts = []
                for iv in (None, self._iv(kind, l, i, mode, k)):
                    self.llm.n_tokens = P
                    out = []
                    with self._side_run(iv):
                        for tok in self.llm.generate(tokens, temp=0.0):
                            if lc.llama_vocab_is_eog(self.llm._model.vocab, tok) or len(out) == n:
                                break
                            out.append(tok)
                    texts.append(self.llm.detokenize(out).decode("utf-8", "replace"))
                self.llm.n_tokens = P
                if n0 > P:  # put the session's own tokens back into the KV cache
                    self.llm.eval(orig[P:n0].tolist())
                self.tap.frames.clear()
                return {"type": "nsteer", "base": texts[0], "alt": texts[1]}
            finally:
                self.busy = False


async def index(_):
    return web.FileResponse(os.path.join(HERE, "index.html"))


async def ws_handler(request):
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    loop, q, stop = asyncio.get_running_loop(), asyncio.Queue(), threading.Event()
    send = lambda obj: loop.call_soon_threadsafe(q.put_nowait, obj)

    async def pump():  # single writer keeps websocket frames in order
        while True:
            await ws.send_json(await q.get())

    pumper = asyncio.create_task(pump())
    send({"type": "structure", **STRUCT})
    send(ENGINE.corpus_status())
    ENGINE.listeners.add(send)
    try:
        async for msg in ws:
            data = msg.json()
            if data["type"] == "chat":
                if ENGINE.busy:
                    done, total = ENGINE.scan
                    send({"type": "error", "message": "이미 생성 중입니다." if done == total else
                          f"위키백과 말뭉치를 색인하는 중입니다 ({done}/{total or '?'} 문서). 처음 한 번만 하며, 끝나면 대화할 수 있습니다."})
                    continue
                stop.clear()
                loop.run_in_executor(None, ENGINE.chat, data["messages"], send, stop)
            elif data["type"] == "stop":
                stop.set()
            elif data["type"] == "select":  # layer whose modules get captured from the next decode on
                ENGINE.tap.layer = int(data["layer"])
                if ENGINE.tap.layer >= 0:
                    loop.run_in_executor(None, ENGINE.refresh, send)
            elif data["type"] in ("neuron", "nprobe", "nsteer"):
                n = (data.get("kind", "mlp"), int(data["layer"]), int(data["index"]))
                args = {"neuron": (ENGINE.neuron, *n),
                        "nprobe": (ENGINE.probe, *n, data.get("mode"), float(data.get("k", 0))),
                        "nsteer": (ENGINE.steer, *n, data.get("mode"), float(data.get("k", 0)), data.get("messages"))}[data["type"]]

                def answer(fn, *a):
                    try:
                        send(fn(*a))
                    except Exception as e:
                        send({"type": "nerror", "message": f"{type(e).__name__}: {e}"})
                loop.run_in_executor(None, answer, *args)
    finally:
        ENGINE.listeners.discard(send)
        stop.set()
        pumper.cancel()
    return ws


if __name__ == "__main__":
    ENGINE = Engine(hf_hub_download(REPO, FILE))
    STRUCT = ENGINE.structure()
    threading.Thread(target=ENGINE.build_corpus, daemon=True).start()
    app = web.Application()
    app.add_routes([web.get("/", index), web.get("/ws", ws_handler)])
    print(f"ready: http://127.0.0.1:{PORT}", flush=True)
    web.run_app(app, host="127.0.0.1", port=PORT, print=None)
