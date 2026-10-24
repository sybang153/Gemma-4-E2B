"""Static data for the in-browser version (index.html served without server.py, e.g. GitHub Pages):

data/structure.json   the structure message server.py sends (graph, layers, metadata, tensors)
data/corpus.json      corpus documents (title, url, lang), document starts, index layout
data/corpus.bin       corpus token ids (uint32)
data/index/<g>.bin    per neuron group (mlp0..mlp34, ple0..ple34) of n neurons:
                      f16 values [n][2 poles][K], u16 corpus positions [n][2][K] (0xFFFF = none),
                      f32 stats [n][mean, std, max, min]; poles sorted strongest first, values signed

Needs the corpus index built by server.py (cache/).
Usage: python build_web_data.py
"""
import json
import os

import numpy as np
from huggingface_hub import hf_hub_download

import server

K = 12  # examples per pole shown by the neuron view
OUT = os.path.join(server.HERE, "data")


def main():
    engine = server.Engine(hf_hub_download(server.REPO, server.FILE))
    engine.build_corpus()  # loads the cached index
    os.makedirs(os.path.join(OUT, "index"), exist_ok=True)
    with open(os.path.join(OUT, "structure.json"), "w", encoding="utf-8") as f:
        json.dump({"type": "structure", **engine.structure()}, f, ensure_ascii=False, separators=(",", ":"))
    np.array(engine.corpus, np.uint32).tofile(os.path.join(OUT, "corpus.bin"))
    index = engine.tap.index
    index.flush()
    groups = {}
    for key, d in index.layers.items():
        n = len(d["max"])
        order = np.argsort(-d["val"], axis=2)[:, :, :K]  # strongest first within each pole
        val = np.take_along_axis(d["val"], order, 2) * np.array([1, -1])[:, None, None]  # pole 1 is stored negated
        pos = np.take_along_axis(d["pos"], order, 2)
        pos = np.where(np.isfinite(val), pos, 0xFFFF).astype(np.uint16)
        val = np.where(np.isfinite(val), val, 0).astype(np.float16)
        mean = d["sum"] / d["count"]
        stats = np.stack([mean, np.sqrt(np.maximum(d["sq"] / d["count"] - mean ** 2, 0)), d["max"], d["min"]], 1).astype(np.float32)
        with open(os.path.join(OUT, "index", f"{key}.bin"), "wb") as f:
            f.write(val.transpose(1, 0, 2).tobytes())  # [n][pole][K]
            f.write(pos.transpose(1, 0, 2).tobytes())
            f.write(stats.tobytes())
        groups[key] = n
    with open(os.path.join(OUT, "corpus.json"), "w", encoding="utf-8") as f:
        json.dump({"docs": [{k: d[k] for k in ("lang", "title", "url")} for d in engine.docs], "starts": engine.starts,
                   "tokens": len(engine.corpus), "k": K, "groups": groups}, f, ensure_ascii=False)
    size = sum(os.path.getsize(os.path.join(r, x)) for r, _, fs in os.walk(OUT) for x in fs)
    print(f"wrote {OUT}: {len(groups)} groups, {size / 2 ** 20:.1f} MiB")


if __name__ == "__main__":
    main()
