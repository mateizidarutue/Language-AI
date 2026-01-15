# obfuscation_paraphrase.py
from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import os
import re
import sqlite3
import hashlib

import numpy as np


# -----------------------------
# Segmentation utilities (runtime + stability)
# -----------------------------
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
_WS_RE = re.compile(r"\s+")
_URL_RE = re.compile(r"https?://\S+|www\.\S+")


def _normalize_ws(t: str) -> str:
    return _WS_RE.sub(" ", t).strip()


def _safe_segment_text(text: str, max_chars: int = 220) -> List[str]:
    """
    Split long 'slice' text into sentence-like segments, then chunk long sentences.
    This makes paraphrasing:
      - faster
      - less likely to produce garbage
      - less likely to hit model max_length issues
    """
    text = _normalize_ws(text)
    if not text:
        return []

    # Protect URLs (paraphrase models can mangle them badly)
    text = _URL_RE.sub(" URL ", text)

    sents = _SENT_SPLIT_RE.split(text)
    segs: List[str] = []
    for s in sents:
        s = s.strip()
        if not s:
            continue
        if len(s) <= max_chars:
            segs.append(s)
        else:
            for i in range(0, len(s), max_chars):
                segs.append(s[i:i + max_chars])
    return segs


def _join_segments(segs: List[str]) -> str:
    return _normalize_ws(" ".join(segs))


def _looks_like_code_or_gibberish(s: str) -> bool:
    """
    Quick heuristics to skip paraphrasing segments that are likely to break.
    """
    if len(s) < 20:
        return True
    if s.count("{") + s.count("}") + s.count(";") > 6:
        return True
    if any(tok in s.lower() for tok in ["__label__", "http", "www", "url"]):
        # we already replace urls with URL token; avoid paraphrasing those segments
        return True
    return False


# -----------------------------
# Persistent cache (SQLite)
# -----------------------------
class ParaphraseCache:
    """
    (model, params_hash, input_segment) -> output_segment
    """
    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self._init_db()

    def _init_db(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cache (
                model TEXT NOT NULL,
                params TEXT NOT NULL,
                input TEXT NOT NULL,
                output TEXT NOT NULL,
                PRIMARY KEY (model, params, input)
            )
            """
        )
        self.conn.commit()

    def get_many(self, model: str, params: str, inputs: List[str]) -> Dict[str, str]:
        if not inputs:
            return {}
        cur = self.conn.cursor()
        out: Dict[str, str] = {}
        chunk = 800
        for i in range(0, len(inputs), chunk):
            batch = inputs[i:i + chunk]
            placeholders = ",".join(["?"] * len(batch))
            cur.execute(
                f"SELECT input, output FROM cache WHERE model=? AND params=? AND input IN ({placeholders})",
                [model, params] + batch
            )
            for inp, oup in cur.fetchall():
                out[inp] = oup
        return out

    def set_many(self, model: str, params: str, pairs: List[Tuple[str, str]]):
        if not pairs:
            return
        cur = self.conn.cursor()
        cur.executemany(
            "INSERT OR REPLACE INTO cache(model, params, input, output) VALUES (?, ?, ?, ?)",
            [(model, params, inp, out) for inp, out in pairs]
        )
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


# -----------------------------
# Transformers paraphrasing core
# -----------------------------
_MODEL_CACHE: Dict[Tuple[str, str, bool], Tuple[object, object]] = {}


def _require_transformers():
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM  # noqa
        import torch  # noqa
    except Exception as e:
        raise RuntimeError(
            "Paraphrasing requires `transformers` + `torch`.\n"
            "Install them or disable paraphrasing."
        ) from e


def _get_device(device: str):
    import torch
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _get_t5(model_name: str, dev, fp16: bool):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    use_amp = fp16 and dev.type == "cuda"
    key = (model_name, str(dev), bool(use_amp))
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, use_safetensors=True)
    model.to(dev)
    if use_amp:
        model = model.half()
    model.eval()

    _MODEL_CACHE[key] = (tokenizer, model)
    return tokenizer, model


def _batch_generate(
    inputs: List[str],
    model_name: str,
    dev,
    max_len: int,
    batch_size: int,
    num_beams: int,
    num_return_sequences: int,
    temperature: float,
    fp16: bool,
    seed: int,
) -> List[List[str]]:
    """
    Returns list-of-candidate-lists (len = len(inputs), each inner list has num_return_sequences paraphrases).
    """
    import torch

    tokenizer, model = _get_t5(model_name, dev, fp16=fp16)
    use_amp = fp16 and dev.type == "cuda"

    # Seed for repeatability
    torch.manual_seed(seed)
    if dev.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    batch_size = max(1, int(batch_size))
    all_out: List[List[str]] = []

    try:
        from tqdm import tqdm
        it = tqdm(range(0, len(inputs), batch_size), desc="paraphrase", total=(len(inputs) + batch_size - 1) // batch_size)
    except Exception:
        it = range(0, len(inputs), batch_size)

    for start in it:
        batch = inputs[start:start + batch_size]

        # T5 paraphrase prompt style (common pattern)
        prompts = [f"paraphrase: {t} </s>" for t in batch]
        enc = tokenizer(prompts, return_tensors="pt", truncation=True, max_length=max_len, padding=True)
        enc = {k: v.to(dev) for k, v in enc.items()}

        with torch.no_grad():
            if use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    out = model.generate(
                        **enc,
                        max_length=max_len,
                        num_beams=num_beams,
                        num_return_sequences=num_return_sequences,
                        do_sample=(temperature > 0.0),
                        temperature=temperature if temperature > 0.0 else 1.0,
                        early_stopping=True,
                    )
            else:
                out = model.generate(
                    **enc,
                    max_length=max_len,
                    num_beams=num_beams,
                    num_return_sequences=num_return_sequences,
                    do_sample=(temperature > 0.0),
                    temperature=temperature if temperature > 0.0 else 1.0,
                    early_stopping=True,
                )

        dec = tokenizer.batch_decode(out, skip_special_tokens=True)

        # Split into chunks per input (HF returns flat list)
        # If num_return_sequences = r, outputs are grouped by input in order.
        r = num_return_sequences
        for i in range(len(batch)):
            cand = dec[i * r:(i + 1) * r]
            cand = [_normalize_ws(c) for c in cand]
            all_out.append(cand)

    return all_out


# -----------------------------
# Candidate selection (quality control)
# -----------------------------
def _jaccard(a: str, b: str) -> float:
    sa = set(a.lower().split())
    sb = set(b.lower().split())
    if not sa or not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb)


def _load_embedder(embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """
    Optional: semantic similarity scoring (recommended if you generate multiple candidates).
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "Semantic scoring requires sentence-transformers.\n"
            "Install it: pip install sentence-transformers"
        ) from e
    return SentenceTransformer(embedder_name)


def _select_best_candidate(
    original: str,
    candidates: List[str],
    min_semantic_sim: float,
    max_lexical_overlap: float,
    embedder=None,
) -> str:
    """
    Pick a paraphrase that:
      - remains semantically close to original (semantic cosine sim high)
      - reduces lexical overlap (jaccard low)
    If embedder is None, falls back to lexical-only selection.
    """
    # Filter trivial empties / too short
    cands = [c for c in candidates if len(c.strip()) >= 15]
    if not cands:
        return original

    if embedder is None:
        # lexical-only: prefer lower overlap but not too different in length
        best = original
        best_score = -1e9
        for c in cands:
            overlap = _jaccard(original, c)
            if overlap > max_lexical_overlap:
                continue
            length_penalty = abs(len(c) - len(original)) / max(1, len(original))
            score = (1.0 - overlap) - 0.2 * length_penalty
            if score > best_score:
                best = c
                best_score = score
        return best if best_score > -1e8 else original

    # semantic scoring
    import numpy as np

    emb = embedder.encode([original] + cands, normalize_embeddings=True)
    o = emb[0]
    cs = emb[1:]
    sims = (cs @ o).astype(float)  # cosine sim

    best = original
    best_score = -1e9
    for c, sim in zip(cands, sims):
        if sim < min_semantic_sim:
            continue
        overlap = _jaccard(original, c)
        if overlap > max_lexical_overlap:
            continue
        # Prefer semantic similarity + lower lexical overlap
        score = sim + (1.0 - overlap)
        if score > best_score:
            best = c
            best_score = score

    return best if best_score > -1e8 else original


# -----------------------------
# Public API: comprehensive paraphrase obfuscation pipeline
# -----------------------------
def paraphrase_obfuscate(
    texts: List[str],
    model_name: str = "Vamsi/T5_Paraphrase_Paws",
    device: str = "auto",
    fp16: bool = True,
    seed: int = 42,
    # segmentation
    max_chars_per_segment: int = 220,
    # generation
    max_len: int = 128,
    batch_size: int = 8,
    num_beams: int = 4,
    num_return_sequences: int = 3,
    temperature: float = 0.0,
    # selection / quality
    use_semantic_rerank: bool = True,
    embedder_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    min_semantic_sim: float = 0.80,
    max_lexical_overlap: float = 0.90,
    # caching
    cache_path: str = "results_si/paraphrase_cache.sqlite",
    # safety
    min_output_chars: int = 20,
) -> List[str]:
    """
    Paraphrasing obfuscation pipeline for long SOBR slices.

    Steps:
      1) Segment long slices into short segments (fast, stable).
      2) Skip segments likely to break paraphrasing (URLs, code-like, too short).
      3) Translate ONLY unique segments (speed) and cache results on disk (speed across reruns).
      4) Optionally generate multiple candidates and select best using semantic similarity + reduced lexical overlap.
      5) Reassemble segments back to the slice.

    This follows the obfuscation principle emphasized in the obfuscation literature: preserve meaning and fluency while reducing
    stylistic/lexical signals that enable profiling. :contentReference[oaicite:2]{index=2}
    """
    if not texts:
        return []

    _require_transformers()
    dev = _get_device(device)

    # Param hash for cache keying (so different settings don't collide)
    params = dict(
        model=model_name,
        max_len=max_len,
        num_beams=num_beams,
        num_return_sequences=num_return_sequences,
        temperature=temperature,
        max_chars=max_chars_per_segment,
        rerank=use_semantic_rerank,
        min_sem=min_semantic_sim,
        max_overlap=max_lexical_overlap,
    )
    params_hash = hashlib.md5(str(sorted(params.items())).encode("utf-8")).hexdigest()

    cache = ParaphraseCache(cache_path)

    # Segment all texts
    segmented = [_safe_segment_text(t, max_chars=max_chars_per_segment) for t in texts]

    # Flatten segments with offsets for reconstruction
    flat: List[str] = []
    offsets: List[Tuple[int, int]] = []
    idx = 0
    for segs in segmented:
        offsets.append((idx, idx + len(segs)))
        flat.extend(segs)
        idx += len(segs)

    if not flat:
        cache.close()
        return texts

    # Decide which segments to actually paraphrase
    mask = [not _looks_like_code_or_gibberish(s) for s in flat]
    to_para = [s for s, m in zip(flat, mask) if m]

    # Unique segments only (big speed boost)
    uniq = list(dict.fromkeys(to_para))
    cached = cache.get_many(model_name, params_hash, uniq)
    missing = [u for u in uniq if u not in cached]

    # Optional semantic reranker (only needed if >1 candidates)
    embedder = None
    if use_semantic_rerank and num_return_sequences > 1:
        embedder = _load_embedder(embedder_name)

    if missing:
        # Generate candidates for missing segments
        cand_lists = _batch_generate(
            missing,
            model_name=model_name,
            dev=dev,
            max_len=max_len,
            batch_size=batch_size,
            num_beams=num_beams,
            num_return_sequences=num_return_sequences,
            temperature=temperature,
            fp16=fp16,
            seed=seed,
        )

        # Select final output per segment
        out_pairs: List[Tuple[str, str]] = []
        for orig, cands in zip(missing, cand_lists):
            best = _select_best_candidate(
                original=orig,
                candidates=cands,
                min_semantic_sim=min_semantic_sim,
                max_lexical_overlap=max_lexical_overlap,
                embedder=embedder,
            )
            if len(best.strip()) < min_output_chars:
                best = orig
            out_pairs.append((orig, best))

        cache.set_many(model_name, params_hash, out_pairs)
        for inp, outp in out_pairs:
            cached[inp] = outp

    # Reconstruct paraphrased segments in original order
    # For segments that were not paraphrased (mask False), keep as-is.
    paraphrased_flat: List[str] = []
    it = iter([cached[s] for s in to_para])  # values in order
    for s, m in zip(flat, mask):
        paraphrased_flat.append(next(it) if m else s)

    # Rebuild each original slice
    outputs: List[str] = []
    for (lo, hi), orig_text in zip(offsets, texts):
        seg_out = paraphrased_flat[lo:hi]
        joined = _join_segments(seg_out)
        outputs.append(joined if len(joined.strip()) >= min_output_chars else orig_text)

    cache.close()
    return outputs


# Backwards-compatible wrapper (so your main runner doesn't break)
def paraphrase_transformers_batch(
    texts: List[str],
    model_name: str = "Vamsi/T5_Paraphrase_Paws",
    max_len: int = 128,
    batch_size: int = 8,
    num_beams: int = 2,
    device: str = "auto",
    fp16: bool = True
) -> List[str]:
    """
    Compatibility function: behaves like your current file but calls the new pipeline
    with conservative defaults (fast).
    """
    return paraphrase_obfuscate(
        texts=texts,
        model_name=model_name,
        device=device,
        fp16=fp16,
        seed=42,
        max_chars_per_segment=220,
        max_len=max_len,
        batch_size=batch_size,
        num_beams=max(1, num_beams),
        num_return_sequences=1,      # keep old behavior: single output
        temperature=0.0,
        use_semantic_rerank=False,   # old behavior had no reranking
        cache_path="results_si/paraphrase_cache.sqlite",
    )
