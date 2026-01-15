# obfuscation_translation.py
from __future__ import annotations

from typing import List, Tuple, Dict, Optional
import re
import sqlite3
import os


# -----------------------------
# Segmentation (runtime + quality)
# -----------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"\s+")


def _normalize_ws(t: str) -> str:
    return _WS_RE.sub(" ", t).strip()


def _split_into_segments(text: str, max_chars: int = 220) -> List[str]:
    """
    Split long slice text into short segments so MT is fast and stable.
    - First split into sentences
    - Then further chunk any long sentence into max_chars pieces
    """
    text = _normalize_ws(text)
    if not text:
        return []

    sents = _SENT_SPLIT_RE.split(text)
    segs: List[str] = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if len(s) <= max_chars:
            segs.append(s)
        else:
            # hard chunk long sentences
            for i in range(0, len(s), max_chars):
                segs.append(s[i:i + max_chars])
    return segs


def _join_segments(segs: List[str]) -> str:
    # join with a space; keep it simple and robust
    return _normalize_ws(" ".join(segs))


# -----------------------------
# Persistent cache (SQLite)
# -----------------------------
class TranslationCache:
    """
    Disk-backed cache: (model_name, input_text) -> output_text
    This makes reruns MUCH faster.
    """
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self._init_db()

    def _init_db(self) -> None:
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                model TEXT NOT NULL,
                src   TEXT NOT NULL,
                tgt   TEXT NOT NULL,
                input TEXT NOT NULL,
                output TEXT NOT NULL,
                PRIMARY KEY (model, src, tgt, input)
            )
            """
        )
        self.conn.commit()

    def get_many(self, model: str, src: str, tgt: str, inputs: List[str]) -> Dict[str, str]:
        if not inputs:
            return {}
        cur = self.conn.cursor()
        out: Dict[str, str] = {}
        # SQLite parameter limit is finite; chunk queries
        chunk = 800
        for i in range(0, len(inputs), chunk):
            batch = inputs[i:i + chunk]
            placeholders = ",".join(["?"] * len(batch))
            cur.execute(
                f"SELECT input, output FROM cache WHERE model=? AND src=? AND tgt=? AND input IN ({placeholders})",
                [model, src, tgt] + batch
            )
            for inp, oup in cur.fetchall():
                out[inp] = oup
        return out

    def set_many(self, model: str, src: str, tgt: str, pairs: List[Tuple[str, str]]) -> None:
        if not pairs:
            return
        cur = self.conn.cursor()
        cur.executemany(
            "INSERT OR REPLACE INTO cache(model, src, tgt, input, output) VALUES (?, ?, ?, ?, ?)",
            [(model, src, tgt, inp, out) for inp, out in pairs]
        )
        self.conn.commit()

    def close(self) -> None:
        try:
            self.conn.close()
        except Exception:
            pass


# -----------------------------
# MT model loading + batching
# -----------------------------
def _require_transformers():
    try:
        from transformers import MarianTokenizer, MarianMTModel  # noqa
        import torch  # noqa
    except Exception as e:
        raise RuntimeError(
            "Translation requires `transformers` + `torch` + `sentencepiece`.\n"
            "Install them or disable translation in your pipeline."
        ) from e


def _get_device(device: str):
    import torch
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


# In-memory cache for loaded models (fast)
_MODEL_CACHE: Dict[Tuple[str, str, bool], Tuple[object, object]] = {}


def _get_tokenizer_model(model_name: str, dev, fp16: bool):
    from transformers import MarianTokenizer, MarianMTModel
    import torch

    key = (model_name, str(dev), bool(fp16 and dev.type == "cuda"))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    tok = MarianTokenizer.from_pretrained(model_name)
    mod = MarianMTModel.from_pretrained(model_name, use_safetensors=True)
    mod.to(dev)
    if fp16 and dev.type == "cuda":
        mod = mod.half()
    mod.eval()

    _MODEL_CACHE[key] = (tok, mod)
    return tok, mod


def _translate_unique_segments(
    segments: List[str],
    model_name: str,
    dev,
    max_len: int,
    batch_size: int,
    num_beams: int,
    fp16: bool,
) -> List[str]:
    """
    Translate a list of unique segments with batching.
    """
    from tqdm import tqdm
    import torch

    tok, mod = _get_tokenizer_model(model_name, dev, fp16=fp16)
    use_amp = fp16 and dev.type == "cuda"

    out: List[str] = []
    step = max(1, int(batch_size))

    it = range(0, len(segments), step)
    it = tqdm(it, desc=f"MT {model_name}", total=(len(segments) + step - 1) // step)

    for start in it:
        batch = segments[start:start + step]
        enc = tok(batch, return_tensors="pt", truncation=True, max_length=max_len, padding=True)
        enc = {k: v.to(dev) for k, v in enc.items()}

        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    gen = mod.generate(**enc, max_length=max_len, num_beams=num_beams, early_stopping=True)
            else:
                gen = mod.generate(**enc, max_length=max_len, num_beams=num_beams, early_stopping=True)

        out.extend(tok.batch_decode(gen, skip_special_tokens=True))

    return out


# -----------------------------
# Public API
# -----------------------------
def roundtrip_translate_transformers_batch(
    texts: List[str],
    # Backwards-compatible params from your current file :contentReference[oaicite:1]{index=1}
    src2tgt: str = "Helsinki-NLP/opus-mt-en-de",
    tgt2src: str = "Helsinki-NLP/opus-mt-de-en",
    max_len: int = 128,
    batch_size: int = 16,
    num_beams: int = 1,
    device: str = "auto",
    fp16: bool = True,
    # New params (speed + chain + cache + segmentation)
    chain: Optional[List[Tuple[str, str]]] = None,
    cache_path: str = "results_si/mt_cache.sqlite",
    max_chars_per_segment: int = 220,
    min_output_chars: int = 20,
) -> List[str]:
    """
    Runtime-friendly round-trip translation obfuscation.

    - Splits each input into short segments (sentences/chunks).
    - Applies MT hop-by-hop (default EN->DE->EN), or a multi-hop chain like EN->DE->FR->EN.
    - Caches each segment translation in a persistent SQLite cache to accelerate reruns.

    Parameters:
      chain:
        If provided, overrides src2tgt/tgt2src and defines the hop languages as tuples:
          [("en","de"), ("de","fr"), ("fr","en")]   # EN->DE->FR->EN
        Models are assumed to be "Helsinki-NLP/opus-mt-{src}-{tgt}".

      max_chars_per_segment:
        Smaller = faster + more stable MT, but more boundary artifacts.
        180–260 is a good range for speed.
    """
    if not texts:
        return []

    _require_transformers()
    import torch

    dev = _get_device(device)

    # Default chain: keep compatibility with your previous behavior
    if chain is None:
        # Infer src/tgt languages from model strings if possible; otherwise keep the two provided models
        # We’ll treat these model names as authoritative for hop 1 and hop 2.
        hops = [
            ("MODEL", src2tgt),
            ("MODEL", tgt2src),
        ]
    else:
        # Convert language chain into model names
        hops = []
        for src, tgt in chain:
            hops.append(("LANG", f"Helsinki-NLP/opus-mt-{src}-{tgt}"))

    # Prepare cache
    cache = TranslationCache(cache_path)

    # Segment all texts
    segmented: List[List[str]] = [_split_into_segments(t, max_chars=max_chars_per_segment) for t in texts]

    # Flatten segments with mapping back to each text
    flat_segments: List[str] = []
    offsets: List[Tuple[int, int]] = []
    idx = 0
    for segs in segmented:
        offsets.append((idx, idx + len(segs)))
        flat_segments.extend(segs)
        idx += len(segs)

    # Early exit for empty content
    if not flat_segments:
        cache.close()
        return texts

    # Apply hop-by-hop translation to flat segments
    current = flat_segments

    for hop_i, (mode, model_name) in enumerate(hops):
        # Determine src/tgt labels for cache keys
        if mode == "LANG":
            # parse "...opus-mt-{src}-{tgt}"
            m = re.search(r"opus-mt-([a-z]{2,3})-([a-z]{2,3})$", model_name)
            src_lang = m.group(1) if m else "src"
            tgt_lang = m.group(2) if m else "tgt"
        else:
            # unknown, but include hop index
            src_lang = f"hop{hop_i}_src"
            tgt_lang = f"hop{hop_i}_tgt"

        # Cache lookup for UNIQUE segments
        uniq = list(dict.fromkeys(current))  # stable unique
        cached = cache.get_many(model_name, src_lang, tgt_lang, uniq)

        missing = [u for u in uniq if u not in cached]

        if missing:
            translated_missing = _translate_unique_segments(
                missing,
                model_name=model_name,
                dev=dev,
                max_len=max_len,
                batch_size=batch_size,
                num_beams=num_beams,
                fp16=fp16,
            )
            cache.set_many(model_name, src_lang, tgt_lang, list(zip(missing, translated_missing)))
            # update cached dict
            for inp, outp in zip(missing, translated_missing):
                cached[inp] = outp

        # Reconstruct in original order
        current = [cached[s] for s in current]

    # Rebuild per-text outputs
    outputs: List[str] = []
    for (lo, hi), orig in zip(offsets, texts):
        segs_out = current[lo:hi]
        joined = _join_segments(segs_out)
        outputs.append(joined if len(joined.strip()) >= min_output_chars else orig)

    cache.close()
    return outputs
