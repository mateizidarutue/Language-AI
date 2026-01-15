# obfuscation_high_weight.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict, Iterable
import os
import re
import math
import sqlite3
import hashlib

import numpy as np

from text_utils import WS_RE  # your existing whitespace regex


# -----------------------------
# Basic tokenization helpers
# -----------------------------
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'-]{1,}")  # conservative "word" tokens
MASK_TOKEN = "[MASK]"


def _normalize_ws(t: str) -> str:
    return WS_RE.sub(" ", t).strip()


def simple_tokenize(text: str) -> List[str]:
    return WORD_RE.findall(text)


def _replace_nth_word(text: str, word: str, n: int, repl: str) -> str:
    """
    Replace the nth occurrence of `word` (case-insensitive, word-boundary) with `repl`.
    """
    pattern = re.compile(rf"\b{re.escape(word)}\b", flags=re.IGNORECASE)
    i = 0

    def _sub(m: re.Match) -> str:
        nonlocal i
        i += 1
        if i == n:
            return repl
        return m.group(0)

    return pattern.sub(_sub, text)


def _all_word_positions(tokens: List[str]) -> Dict[str, List[int]]:
    """
    Map token -> list of 1-based occurrence indices (for nth occurrence replacement).
    """
    pos: Dict[str, List[int]] = {}
    counts: Dict[str, int] = {}
    for t in tokens:
        key = t.lower()
        counts[key] = counts.get(key, 0) + 1
        pos.setdefault(key, []).append(counts[key])
    return pos


# -----------------------------
# POS tagging (optional)
# -----------------------------
def _pos_tag(tokens: List[str]) -> List[Tuple[str, str]]:
    """
    Best-effort POS tags. Uses spaCy if available (better), else NLTK.
    Returns list of (token, POS_tag).
    """
    # spaCy preferred (more consistent tags)
    try:
        import spacy
        # you might already have spacy installed; model may or may not be
        # available; try small english
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except Exception:
            # fallback: blank english tokenization + tagger not available
            raise RuntimeError("spaCy model missing")
        doc = nlp(" ".join(tokens))
        return [(t.text, t.pos_) for t in doc]
    except Exception:
        # NLTK fallback
        try:
            import nltk
            from nltk import pos_tag
        except Exception:
            # no POS available
            return [(t, "") for t in tokens]
        try:
            return [(w, p) for w, p in pos_tag(tokens)]
        except Exception:
            return [(t, "") for t in tokens]


def _pos_compatible(pos_a: str, pos_b: str) -> bool:
    """
    Coarse POS compatibility check.
    - If tags unavailable, allow.
    - Otherwise require same coarse class: NOUN/VERB/ADJ/ADV/PROPN, etc.
    """
    if not pos_a or not pos_b:
        return True
    return pos_a == pos_b


# -----------------------------
# Substitute model interface (LR + vectorizer)
# -----------------------------
def _lr_logit_for_label(lr_model, X, label_index: int) -> float:
    """
    Return logit-like score for a class.
    Works for binary LR (decision_function -> 1 value).
    """
    # scikit-learn LR:
    # - binary: decision_function returns shape (n_samples,)
    # - multi: decision_function returns (n_samples, n_classes)
    s = lr_model.decision_function(X)
    if s.ndim == 1:
        # binary: positive class is classes_[1]
        # map label_index to + or -
        if label_index == 1:
            return float(s[0])
        else:
            return float(-s[0])
    return float(s[0, label_index])


def _predict_label_and_conf(lr_model, X) -> Tuple[int, float]:
    """
    Returns (predicted_label_index, confidence_for_pred_label).
    Confidence is probability if available.
    """
    if hasattr(lr_model, "predict_proba"):
        proba = lr_model.predict_proba(X)[0]
        pred = int(np.argmax(proba))
        return pred, float(proba[pred])
    # fallback: use decision_function magnitude mapped through sigmoid
    s = lr_model.decision_function(X)
    if np.ndim(s) == 1:
        # binary
        p = 1.0 / (1.0 + np.exp(-float(s[0])))
        pred = 1 if p >= 0.5 else 0
        conf = p if pred == 1 else (1.0 - p)
        return pred, float(conf)
    pred = int(np.argmax(s[0]))
    # softmax-ish confidence
    ex = np.exp(s[0] - np.max(s[0]))
    pr = ex / np.sum(ex)
    return pred, float(pr[pred])


# -----------------------------
# Importance scoring (paper-inspired)
# -----------------------------
def _omission_importance(
    text: str,
    tokens: List[str],
    lr_model,
    vectorizer,
    target_label_index: int,
    max_targets: int = 30,
) -> List[Tuple[str, float]]:
    """
    Rank tokens by an omission-style score: drop one token (one occurrence)
    and see how much target-label score decreases (Equation 1 spirit).
    Practical approximation for our setting: for each unique token, remove its
    first occurrence and measure delta logit for target label.

    This mirrors the paper's "omission score" ranking :contentReference[oaicite:3]{index=3}.
    """
    base_X = vectorizer.transform([text])
    base = _lr_logit_for_label(lr_model, base_X, target_label_index)

    uniq = sorted(set(t.lower() for t in tokens))
    scored: List[Tuple[str, float]] = []
    for w in uniq:
        # remove first occurrence by replacing with "" (space)
        t2 = _replace_nth_word(text, w, 1, "")
        X2 = vectorizer.transform([t2])
        s2 = _lr_logit_for_label(lr_model, X2, target_label_index)
        # positive importance means: removing w lowers target score (good target to attack)
        imp = base - s2
        scored.append((w, float(imp)))

    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:max_targets]


# -----------------------------
# Candidate generation
# -----------------------------
def try_wordnet_synonyms(token: str, max_cands: int = 10) -> List[str]:
    """
    WordNet synonyms (baseline "WS" style).
    """
    if not token.isalpha() or len(token) < 4:
        return []
    try:
        from nltk.corpus import wordnet as wn
    except Exception:
        return []

    try:
        synsets = wn.synsets(token)
    except LookupError:
        return []

    out: List[str] = []
    for ss in synsets[:5]:
        for lemma in ss.lemmas()[:10]:
            cand = lemma.name().replace("_", " ")
            if " " in cand:
                continue
            if cand.lower() == token.lower():
                continue
            out.append(cand)
            if len(out) >= max_cands:
                return out
    return out


def masked_bert_candidates(
    text: str,
    target_word: str,
    occurrence: int,
    model_name: str = "bert-base-uncased",
    top_k: int = 10,
) -> List[str]:
    """
    Paper-style contextual substitution using masked LM ("MB" idea) :contentReference[oaicite:4]{index=4}.
    We mask the target occurrence and ask BERT for top-k predictions.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        import torch
    except Exception:
        return []

    # Replace target occurrence with [MASK]
    masked_text = _replace_nth_word(text, target_word, occurrence, MASK_TOKEN)

    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModelForMaskedLM.from_pretrained(model_name)
    mod.eval()

    # tokenize
    enc = tok(masked_text, return_tensors="pt")
    mask_id = tok.mask_token_id
    mask_positions = (enc["input_ids"][0] == mask_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        return []

    with torch.no_grad():
        out = mod(**enc).logits[0]

    # take first mask position
    pos = int(mask_positions[0])
    scores = out[pos]
    top = torch.topk(scores, k=top_k).indices.tolist()
    cands = []
    for tid in top:
        w = tok.decode([tid]).strip()
        # filter junk
        if not w or w.startswith("##"):
            continue
        if re.search(r"\W", w):
            continue
        if w.lower() == target_word.lower():
            continue
        cands.append(w)
    return cands


def dropout_bert_candidates(
    text: str,
    target_word: str,
    occurrence: int,
    model_name: str = "bert-base-uncased",
    top_k: int = 10,
    dropout_p: float = 0.3,
    n_samples: int = 6,
) -> List[str]:
    """
    Dropout BERT ("DB" idea): apply dropout at inference by enabling train mode
    and sampling several times, then aggregate top candidates :contentReference[oaicite:5]{index=5}.
    This approximates the paper's dropout substitution idea in a practical way.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForMaskedLM
        import torch
    except Exception:
        return []

    tok = AutoTokenizer.from_pretrained(model_name)
    mod = AutoModelForMaskedLM.from_pretrained(model_name)
    # enable dropout
    mod.train()

    masked_text = _replace_nth_word(text, target_word, occurrence, MASK_TOKEN)
    enc = tok(masked_text, return_tensors="pt")
    mask_id = tok.mask_token_id
    mask_positions = (enc["input_ids"][0] == mask_id).nonzero(as_tuple=True)[0]
    if len(mask_positions) == 0:
        mod.eval()
        return []
    pos = int(mask_positions[0])

    # apply dropout probability by setting all dropout layers (best-effort)
    for m in mod.modules():
        if m.__class__.__name__.lower().startswith("dropout"):
            try:
                m.p = dropout_p
            except Exception:
                pass

    counts: Dict[str, int] = {}
    with torch.no_grad():
        for _ in range(n_samples):
            logits = mod(**enc).logits[0, pos]
            top = torch.topk(logits, k=top_k).indices.tolist()
            for tid in top:
                w = tok.decode([tid]).strip()
                if not w or w.startswith("##"):
                    continue
                if re.search(r"\W", w):
                    continue
                if w.lower() == target_word.lower():
                    continue
                counts[w] = counts.get(w, 0) + 1

    mod.eval()
    # rank by frequency across samples
    return [w for w, _ in sorted(counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]]


# -----------------------------
# Semantic preservation (optional)
# -----------------------------
def _sent_embedder(name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError("Install sentence-transformers for semantic filtering.") from e
    return SentenceTransformer(name)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


# -----------------------------
# Cache for candidate generation
# -----------------------------
class _SqlCache:
    """
    Cache expensive candidate generation calls:
      (kind, model, params, context_hash) -> list of candidates
    """
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path)
        self._init()

    def _init(self):
        cur = self.conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS cand_cache (
                kind TEXT NOT NULL,
                model TEXT NOT NULL,
                params TEXT NOT NULL,
                ctx TEXT NOT NULL,
                cands TEXT NOT NULL,
                PRIMARY KEY(kind, model, params, ctx)
            )
            """
        )
        self.conn.commit()

    def get(self, kind: str, model: str, params: str, ctx: str) -> Optional[List[str]]:
        cur = self.conn.cursor()
        cur.execute(
            "SELECT cands FROM cand_cache WHERE kind=? AND model=? AND params=? AND ctx=?",
            (kind, model, params, ctx)
        )
        row = cur.fetchone()
        if not row:
            return None
        return row[0].split("\n") if row[0] else []

    def set(self, kind: str, model: str, params: str, ctx: str, cands: List[str]) -> None:
        cur = self.conn.cursor()
        cur.execute(
            "INSERT OR REPLACE INTO cand_cache(kind, model, params, ctx, cands) VALUES (?, ?, ?, ?, ?)",
            (kind, model, params, ctx, "\n".join(cands))
        )
        self.conn.commit()

    def close(self):
        try:
            self.conn.close()
        except Exception:
            pass


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class SubstitutionConfig:
    max_targets: int = 30         # how many important words to try
    max_changes: int = 30         # how many substitutions allowed in total
    top_k_cands: int = 10         # candidate list size per method
    synonym_only: bool = False    # if True: only WordNet candidates
    use_masked_bert: bool = True
    use_dropout_bert: bool = True
    bert_model: str = "bert-base-uncased"
    dropout_p: float = 0.3
    dropout_samples: int = 6

    # filtering / reranking
    require_pos_match: bool = True
    use_semantic_filter: bool = False
    semantic_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    min_sem_sim: float = 0.80

    # caching
    cache_path: str = "results_si/subst_cand_cache.sqlite"


# -----------------------------
# Main attack (TextFooler-inspired loop)
# -----------------------------
def obfuscate_high_weight_substitution(
    text: str,
    lr_model,
    vectorizer,
    target_label: int,
    cfg: Optional[SubstitutionConfig] = None,
) -> str:
    """
    Paper-inspired lexical substitution attack for stylometric obfuscation :contentReference[oaicite:6]{index=6}.

    High-level loop (Algorithm 1 spirit):
      1) Rank words by importance using omission-style score (substitute model).
      2) For each important word:
           a) generate candidate replacements (WordNet / Masked BERT / Dropout BERT)
           b) optional POS + semantic similarity filtering
           c) pick replacement that most reduces target-label confidence/logit
      3) apply iteratively until prediction changes or budget exhausted.

    Inputs:
      - text: a slice (long Reddit text)
      - lr_model + vectorizer: your substitute model pipeline
      - target_label: the label you want to obfuscate away from (e.g., the predicted class)

    Returns:
      - obfuscated text
    """
    cfg = cfg or SubstitutionConfig()

    t = _normalize_ws(text)
    if not t:
        return t

    # Prepare cache for expensive candidate generation
    cache = _SqlCache(cfg.cache_path)

    # Tokenize + POS tags (on initial)
    tokens = simple_tokenize(t)
    if not tokens:
        cache.close()
        return t

    pos_tags = dict((w.lower(), p) for w, p in _pos_tag(tokens))

    # Compute importance ranking (omission score)
    ranked = _omission_importance(
        text=t,
        tokens=tokens,
        lr_model=lr_model,
        vectorizer=vectorizer,
        target_label_index=target_label,
        max_targets=cfg.max_targets,
    )

    # Baseline prediction / confidence
    X0 = vectorizer.transform([t])
    pred0, conf0 = _predict_label_and_conf(lr_model, X0)

    # Optional semantic embedder for doc-level check
    embedder = None
    base_emb = None
    if cfg.use_semantic_filter:
        embedder = _sent_embedder(cfg.semantic_model)
        base_emb = embedder.encode([t], normalize_embeddings=True)[0]

    # Track word occurrence indices
    tok_positions = _all_word_positions(tokens)

    changes = 0
    current = t

    for w, imp in ranked:
        if changes >= cfg.max_changes:
            break
        if imp <= 0:
            # removing it didn't reduce target score; skip
            continue

        occurrences = tok_positions.get(w, [])
        if not occurrences:
            continue

        # We try one occurrence at a time (first occurrence is often enough, faster)
        occ = occurrences[0]

        # Candidate generation
        candidates: List[str] = []

        # 1) WordNet (baseline)
        wn_cands = try_wordnet_synonyms(w, max_cands=cfg.top_k_cands)
        candidates.extend(wn_cands)

        if not cfg.synonym_only:
            # Build context hash for caching (text + target + occ)
            ctx = hashlib.md5(f"{current}||{w}||{occ}".encode("utf-8")).hexdigest()

            # 2) Masked BERT candidates (MB)
            if cfg.use_masked_bert:
                params = f"topk={cfg.top_k_cands}"
                cached = cache.get("mb", cfg.bert_model, params, ctx)
                if cached is None:
                    mb = masked_bert_candidates(
                        current, w, occ, model_name=cfg.bert_model, top_k=cfg.top_k_cands
                    )
                    cache.set("mb", cfg.bert_model, params, ctx, mb)
                    cached = mb
                candidates.extend(cached)

            # 3) Dropout BERT candidates (DB)
            if cfg.use_dropout_bert:
                params = f"topk={cfg.top_k_cands}|p={cfg.dropout_p}|n={cfg.dropout_samples}"
                cached = cache.get("db", cfg.bert_model, params, ctx)
                if cached is None:
                    db = dropout_bert_candidates(
                        current, w, occ,
                        model_name=cfg.bert_model,
                        top_k=cfg.top_k_cands,
                        dropout_p=cfg.dropout_p,
                        n_samples=cfg.dropout_samples,
                    )
                    cache.set("db", cfg.bert_model, params, ctx, db)
                    cached = db
                candidates.extend(cached)

        # Deduplicate, keep order
        seen = set()
        uniq_cands = []
        for c in candidates:
            c2 = c.strip()
            if not c2:
                continue
            if c2.lower() == w.lower():
                continue
            if c2.lower() in seen:
                continue
            seen.add(c2.lower())
            uniq_cands.append(c2)

        if not uniq_cands:
            continue

        # Filter candidates by POS tag compatibility (paper suggests POS-based checks) :contentReference[oaicite:7]{index=7}
        if cfg.require_pos_match:
            pos_w = pos_tags.get(w, "")
            filtered = []
            for c in uniq_cands:
                # tag single word candidate
                c_pos = _pos_tag([c])[0][1] if c else ""
                if _pos_compatible(pos_w, c_pos):
                    filtered.append(c)
            if filtered:
                uniq_cands = filtered

        # Evaluate each candidate by how much it reduces target label logit/prob
        best_text = None
        best_score = None  # lower is better (target score)
        best_candidate = None

        for c in uniq_cands:
            # Apply substitution at the selected occurrence
            trial = _replace_nth_word(current, w, occ, c)

            # Optional semantic doc-level similarity filter (fast enough for small candidate lists)
            if cfg.use_semantic_filter and embedder is not None and base_emb is not None:
                emb = embedder.encode([trial], normalize_embeddings=True)[0]
                sim = float(np.dot(base_emb, emb))
                if sim < cfg.min_sem_sim:
                    continue

            Xt = vectorizer.transform([trial])
            score = _lr_logit_for_label(lr_model, Xt, target_label)

            if best_score is None or score < best_score:
                best_score = score
                best_text = trial
                best_candidate = c

        if best_text is None:
            continue

        # Apply best change if it helps
        before_X = vectorizer.transform([current])
        before_score = _lr_logit_for_label(lr_model, before_X, target_label)

        if best_score is not None and best_score < before_score:
            current = best_text
            changes += 1

            # Stop early if target label is no longer predicted
            Xn = vectorizer.transform([current])
            pred, conf = _predict_label_and_conf(lr_model, Xn)
            if pred != target_label:
                break

    cache.close()
    return _normalize_ws(current)
