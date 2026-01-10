"""
S/I Stylometry + Adversarial Obfuscation (SOBR-style) — End-to-end pipeline

WHAT THIS SCRIPT DOES
1) Loads your slice-based S/I dataset from sensing_intuitive.xlsx
   Expected columns (based on your file): auhtor_ID (typo in file), post, sensing (0/1)
2) Splits data at AUTHOR LEVEL into train/dev/test (prevents author leakage)
3) Trains two baseline models:
   - Logistic Regression with TF–IDF n-grams (white-box, interpretable)
   - fastText supervised (black-box-ish, strong baseline)
4) Implements obfuscation techniques:
   - Lexical cue removal (MBTI/self-identification/jargon patterns)
   - High-weight feature substitution (uses LR coefficients + optional WordNet synonyms)
   - Character-level & visual perturbations (homoglyph/diacritics/noise)
   - Paraphrasing (optional, Transformers)  [optional dependency]
   - Round-trip translation EN->DE->EN (optional, Transformers) [optional dependency]
5) Evaluates (macro-F1 + accuracy) on unmodified and obfuscated test sets
6) Produces qualitative artifacts:
   - Top LR features for each class
   - Before/after example showcase with predictions + confidence
   - Post-obfuscation error analysis themes (lightweight, rule-based tags)

HOW TO RUN (minimal)
    python si_obfuscation_pipeline.py --data sensing_intuitive.xlsx

COMMON OPTIONS
    --run_paraphrase            Enable paraphrasing (requires `transformers`, `torch`)
    --run_translation           Enable round-trip translation (requires `transformers`, `torch`)
    --out_dir results

Dependencies (minimal): pandas, numpy, scikit-learn, openpyxl, fasttext
Optional:
  - nltk (WordNet): pip install nltk  (then python -c "import nltk; nltk.download('wordnet'); nltk.download('omw-1.4')")
  - transformers + torch for paraphrase/translation: pip install transformers torch sentencepiece

NOTE: This is designed for a course project: reproducible, interpretable, and practical.
"""

import argparse
import os
import re
import random
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# -----------------------------
# Reproducibility
# -----------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

# -----------------------------
# Data loading + author-level split
# -----------------------------
@dataclass
class SplitData:
    X_train: List[str]
    y_train: np.ndarray
    a_train: np.ndarray

    X_dev: List[str]
    y_dev: np.ndarray
    a_dev: np.ndarray

    X_test: List[str]
    y_test: np.ndarray
    a_test: np.ndarray

def load_si_xlsx(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, engine="openpyxl")

    # Column normalization: your file uses "auhtor_ID" (typo)
    cols = {c.lower().strip(): c for c in df.columns}
    author_col = None
    for cand in ["author_id", "auhtor_id", "author", "user_id"]:
        if cand in cols:
            author_col = cols[cand]
            break
    if author_col is None:
        raise ValueError(f"Could not find an author id column. Columns: {list(df.columns)}")

    post_col = None
    for cand in ["post", "text", "content"]:
        if cand in cols:
            post_col = cols[cand]
            break
    if post_col is None:
        raise ValueError(f"Could not find a post/text column. Columns: {list(df.columns)}")

    label_col = None
    for cand in ["sensing", "label", "y"]:
        if cand in cols:
            label_col = cols[cand]
            break
    if label_col is None:
        raise ValueError(f"Could not find the sensing label column. Columns: {list(df.columns)}")

    out = df[[author_col, post_col, label_col]].copy()
    out.columns = ["author_id", "text", "label"]

    # Clean basic types
    out["author_id"] = out["author_id"].astype(str)
    out["text"] = out["text"].fillna("").astype(str)
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int)

    # Remove empty texts
    out = out[out["text"].str.strip().astype(bool)].reset_index(drop=True)
    return out

def author_level_split(df: pd.DataFrame, seed: int = 42, train_size=0.8, dev_size=0.1, test_size=0.1) -> SplitData:
    assert abs(train_size + dev_size + test_size - 1.0) < 1e-9

    authors = df["author_id"].unique()
    authors_train, authors_tmp = train_test_split(authors, test_size=(1 - train_size), random_state=seed, shuffle=True)

    # Split remaining into dev/test
    tmp_frac = (dev_size + test_size)
    dev_frac_of_tmp = dev_size / tmp_frac
    authors_dev, authors_test = train_test_split(authors_tmp, test_size=(1 - dev_frac_of_tmp), random_state=seed, shuffle=True)

    def subset(authors_set: np.ndarray) -> Tuple[List[str], np.ndarray, np.ndarray]:
        sub = df[df["author_id"].isin(authors_set)]
        return sub["text"].tolist(), sub["label"].values, sub["author_id"].values

    X_train, y_train, a_train = subset(authors_train)
    X_dev, y_dev, a_dev = subset(authors_dev)
    X_test, y_test, a_test = subset(authors_test)

    return SplitData(X_train, y_train, a_train, X_dev, y_dev, a_dev, X_test, y_test, a_test)

# -----------------------------
# Text preprocessing utilities
# -----------------------------
_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")

def normalize_text(t: str) -> str:
    t = _URL_RE.sub(" URL ", t)
    t = t.replace("\u200b", "")  # zero-width space
    t = _WS_RE.sub(" ", t).strip()
    return t

# -----------------------------
# Obfuscation techniques
# -----------------------------
MBTI_TERMS = [
    # obvious MBTI mentions / codes
    r"\bmbti\b", r"\bmyers[- ]?briggs\b", r"\bpersonality\s*type\b",
    r"\bintp\b", r"\bintj\b", r"\binfp\b", r"\binfj\b",
    r"\bistp\b", r"\bistj\b", r"\bisfp\b", r"\bisfj\b",
    r"\bentp\b", r"\bentj\b", r"\benfp\b", r"\benfj\b",
    r"\bestp\b", r"\bestj\b", r"\besfp\b", r"\besfj\b",
    # S/N jargon + cognitive functions
    r"\bsensing\b", r"\bsensor(s)?\b", r"\bintuitive\b", r"\bintuition\b",
    r"\bsi\b", r"\bse\b", r"\bni\b", r"\bne\b", r"\bti\b", r"\bte\b", r"\bfi\b", r"\bfe\b",
    r"\bcognitive\s+function(s)?\b",
    # phrases that often indicate self-identification
    r"\b(as\s+an?|i\s+am)\s+(an?\s+)?(intuitive|sensor|sensing)\b",
]
MBTI_RE = re.compile("|".join(MBTI_TERMS), flags=re.IGNORECASE)

def obfuscate_lexical_cue_removal(text: str) -> str:
    """
    Removes / neutralizes explicit MBTI + S/N self-identification cues.
    This targets label leakage rather than deeper style.
    """
    t = text
    t = MBTI_RE.sub(" [trait] ", t)
    # Neutralize common “type mention” patterns like "I'm an INTP"
    t = re.sub(r"\b(i\s*['a]m|i\s+am|as\s+an?)\s+(__)?label__?\w+\b", " [selfref] ", t, flags=re.I)
    t = _WS_RE.sub(" ", t).strip()
    return t

# ---- Character/visual perturbations (Eger-style inspirations) ----
HOMOGLYPHS = {
    "a": ["\u0430", "\u03b1"],  # Cyrillic a, Greek alpha
    "e": ["\u0435", "\u03b5"],  # Cyrillic e, Greek epsilon
    "i": ["\u0456", "\u03b9"],  # Cyrillic i, Greek iota
    "o": ["\u043e", "\u03bf"],  # Cyrillic o, Greek omicron
    "c": ["\u0441"],            # Cyrillic es
    "p": ["\u0440"],            # Cyrillic er
    "x": ["\u0445"],            # Cyrillic ha
    "y": ["\u0443"],            # Cyrillic u
    "s": ["\u0455"],            # Cyrillic dze
}

DIACRITICS = {
    "a": ["\u00e1", "\u00e0", "\u00e2", "\u00e4"],
    "e": ["\u00e9", "\u00e8", "\u00ea", "\u00eb"],
    "i": ["\u00ed", "\u00ec", "\u00ee", "\u00ef"],
    "o": ["\u00f3", "\u00f2", "\u00f4", "\u00f6"],
    "u": ["\u00fa", "\u00f9", "\u00fb", "\u00fc"],
}

def obfuscate_char_visual(
    text: str,
    homoglyph_rate: float = 0.02,
    diacritic_rate: float = 0.01,
    seed: int = 42
) -> str:
    rng = random.Random(seed)
    chars = list(text)
    for idx, ch in enumerate(chars):
        low = ch.lower()
        if low in HOMOGLYPHS and rng.random() < homoglyph_rate:
            rep = rng.choice(HOMOGLYPHS[low])
            chars[idx] = rep if ch.islower() else rep  # keep simple
        elif low in DIACRITICS and rng.random() < diacritic_rate:
            rep = rng.choice(DIACRITICS[low])
            chars[idx] = rep if ch.islower() else rep
    return "".join(chars)

# ---- High-weight substitution (LR-driven) ----
def try_wordnet_synonym(token: str) -> Optional[str]:
    """
    Best-effort synonym via WordNet (optional). If not available, return None.
    """
    try:
        import nltk
        from nltk.corpus import wordnet as wn
    except Exception:
        return None

    # Token must be alphabetic and not too short
    if not token.isalpha() or len(token) < 4:
        return None

    try:
        synsets = wn.synsets(token)
    except LookupError:
        return None
    if not synsets:
        return None

    # Pick a lemma that's not identical, prefer single-word lemmas
    for ss in synsets[:3]:
        for lemma in ss.lemmas()[:5]:
            cand = lemma.name().replace("_", " ")
            if cand.lower() != token.lower() and " " not in cand:
                return cand
    return None

def build_lr_top_features(vectorizer: TfidfVectorizer, lr: LogisticRegression, top_k: int = 200) -> Tuple[List[str], List[str]]:
    """
    Returns top features for class 1 and class 0 (binary LR).
    """
    feat_names = np.array(vectorizer.get_feature_names_out())
    coef = lr.coef_.ravel()  # shape: (n_features,)
    top_pos = feat_names[np.argsort(coef)[-top_k:]][::-1].tolist()  # strongest for class 1
    top_neg = feat_names[np.argsort(coef)[:top_k]].tolist()         # strongest for class 0
    return top_pos, top_neg

def obfuscate_high_weight_substitution(
    text: str,
    sensitive_tokens: List[str],
    substitute_mode: str = "synonym_then_neutral",
    neutral_token: str = "neutral",
    max_replacements: int = 30
) -> str:
    """
    Replaces sensitive tokens found in text.
    For practicality, we treat sensitive_tokens as word-level items (unigrams).
    For n-grams with spaces, we do phrase replacement too.

    substitute_mode:
      - "synonym_then_neutral": try WordNet synonym, else replace with neutral_token
      - "neutral": always neutral_token
    """
    t = text
    replaced = 0

    # Replace longer phrases first (n-grams)
    phrases = sorted([s for s in sensitive_tokens if " " in s], key=len, reverse=True)
    words = [s for s in sensitive_tokens if " " not in s]

    for ph in phrases:
        if replaced >= max_replacements:
            break
        pattern = re.compile(r"\b" + re.escape(ph) + r"\b", flags=re.IGNORECASE)
        if pattern.search(t):
            t = pattern.sub(neutral_token, t)
            replaced += 1

    # Word replacement
    def repl_word(m: re.Match) -> str:
        nonlocal replaced
        if replaced >= max_replacements:
            return m.group(0)
        tok = m.group(0)
        if substitute_mode == "neutral":
            replaced += 1
            return neutral_token
        syn = try_wordnet_synonym(tok.lower())
        replaced += 1
        return syn if syn else neutral_token

    if words and replaced < max_replacements:
        # Compile one regex for all words (careful with length)
        safe_words = [w for w in words if re.match(r"^[\w'-]+$", w)]
        if safe_words:
            pattern = re.compile(r"\b(" + "|".join(map(re.escape, safe_words[:5000])) + r")\b", flags=re.IGNORECASE)
            t = pattern.sub(repl_word, t)

    t = _WS_RE.sub(" ", t).strip()
    return t

# ---- Optional: Paraphrasing + Translation (Transformers) ----
def paraphrase_transformers(text: str, model_name: str = "Vamsi/T5_Paraphrase_Paws", max_len: int = 256) -> str:
    """
    Optional paraphrase using Transformers (requires transformers + torch).
    This is best-effort and may be slow. We paraphrase sentence-by-sentence lightly.
    """
    try:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
    except Exception as e:
        raise RuntimeError("Paraphrasing requires `transformers` + `torch`. Install them or disable --run_paraphrase.") from e

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    # Keep it simple: paraphrase only the first chunk to control runtime
    chunk = text[:1500]
    prompt = f"paraphrase: {chunk} </s>"
    enc = tokenizer([prompt], return_tensors="pt", truncation=True, max_length=max_len)
    with torch.no_grad():
        out = model.generate(
            **enc,
            max_length=max_len,
            num_beams=5,
            num_return_sequences=1,
            temperature=1.0,
            early_stopping=True
        )
    dec = tokenizer.decode(out[0], skip_special_tokens=True)
    # If model outputs empty or too short, fall back
    return dec if len(dec.strip()) > 20 else text

def roundtrip_translate_transformers(text: str, src2tgt="Helsinki-NLP/opus-mt-en-de", tgt2src="Helsinki-NLP/opus-mt-de-en", max_len: int = 256) -> str:
    """
    Optional EN->DE->EN translation using MarianMT models (Transformers).
    """
    try:
        from transformers import MarianTokenizer, MarianMTModel
        import torch
    except Exception as e:
        raise RuntimeError("Translation requires `transformers` + `torch` + `sentencepiece`. Install them or disable --run_translation.") from e

    def translate(t: str, model_name: str) -> str:
        tok = MarianTokenizer.from_pretrained(model_name)
        mod = MarianMTModel.from_pretrained(model_name)
        # Translate only first chunk to keep runtime manageable
        chunk = t[:1500]
        enc = tok([chunk], return_tensors="pt", truncation=True, max_length=max_len)
        with torch.no_grad():
            out = mod.generate(**enc, max_length=max_len, num_beams=4, early_stopping=True)
        return tok.decode(out[0], skip_special_tokens=True)

    de = translate(text, src2tgt)
    en = translate(de, tgt2src)
    return en if len(en.strip()) > 20 else text

# -----------------------------
# Models: LR + fastText
# -----------------------------
@dataclass
class LRModel:
    vectorizer: TfidfVectorizer
    clf: LogisticRegression

def train_logreg(X_train: List[str], y_train: np.ndarray) -> LRModel:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),            # word n-grams
        analyzer="word"
    )
    Xv = vectorizer.fit_transform(X_train)
    clf = LogisticRegression(
        max_iter=2000,
        solver="liblinear",
        class_weight=None
    )
    clf.fit(Xv, y_train)
    return LRModel(vectorizer, clf)

def predict_logreg(model: LRModel, X: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    Xv = model.vectorizer.transform(X)
    proba = model.clf.predict_proba(Xv)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return pred, proba

# fastText wrapper
@dataclass
class FastTextModel:
    ft: Any
    label_prefix: str = "__label__"

def train_fasttext(X_train: List[str], y_train: np.ndarray, X_dev: List[str], y_dev: np.ndarray, out_dir: str, seed: int = 42) -> FastTextModel:
    """
    Trains fastText supervised model using temporary txt files.
    """
    try:
        import fasttext
    except Exception as e:
        raise RuntimeError("fastText not installed. Run `pip install fasttext` to continue.") from e

    os.makedirs(out_dir, exist_ok=True)
    train_path = os.path.join(out_dir, "fasttext_train.txt")
    dev_path = os.path.join(out_dir, "fasttext_dev.txt")

    def dump(path: str, X: List[str], y: np.ndarray) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for t, lab in zip(X, y):
                t = normalize_text(t)
                # fastText expects "__label__{label} text..."
                f.write(f"__label__{int(lab)} {t}\n")

    dump(train_path, X_train, y_train)
    dump(dev_path, X_dev, y_dev)

    # A small, reasonable hyperparam set for course project:
    # You can tune using dev, but keep simple here.
    model = fasttext.train_supervised(
        input=train_path,
        lr=0.5,
        epoch=10,
        wordNgrams=2,
        dim=100,
        minn=2,
        maxn=5,
        bucket=200000,
        loss="softmax",
        thread=4
    )
    return FastTextModel(model)

def predict_fasttext(model: FastTextModel, X: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    preds = []
    probs = []
    for t in X:
        t = normalize_text(t)
        labels, ps = model.ft.predict(t, k=2)
        # labels like ["__label__1", "__label__0"], ps like [0.7, 0.3]
        # Extract probability for class 1
        p1 = 0.0
        for lab, p in zip(labels, ps):
            if lab.endswith("1"):
                p1 = float(p)
                break
        pred = 1 if p1 >= 0.5 else 0
        preds.append(pred)
        probs.append(p1)
    return np.array(preds, dtype=int), np.array(probs, dtype=float)

# -----------------------------
# Evaluation + reporting helpers
# -----------------------------
def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "accuracy": float(accuracy_score(y_true, y_pred))
    }

def save_json(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def ensure_out_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)

# -----------------------------
# Qualitative analysis helpers
# -----------------------------
def categorize_feature(feat: str) -> str:
    f = feat.lower()

    # Strong leakage indicators
    if re.search(r"\bmbti\b|myers|briggs|intp|intj|infp|infj|entp|entj|enfp|enfj|istp|istj|isfp|isfj|estp|estj|esfp|esfj", f):
        return "MBTI/self-id"
    if re.search(r"\b(si|se|ni|ne|ti|te|fi|fe)\b|cognitive function", f):
        return "MBTI/jargon"
    if "trait" in f or "personality" in f:
        return "meta-trait talk"

    # Rough semantic buckets (very lightweight heuristics)
    concrete = ["smell", "taste", "touch", "sound", "see", "color", "shape", "texture", "weather", "food", "room", "car", "street", "size"]
    abstract = ["meaning", "concept", "theory", "idea", "symbol", "interpret", "abstract", "pattern", "possibility", "imagine", "intuition"]
    if any(w in f for w in concrete):
        return "concrete/sensory"
    if any(w in f for w in abstract):
        return "abstract/interpretive"

    return "other"

def build_feature_table(model: LRModel, top_k: int = 30) -> Dict[str, List[Dict[str, str]]]:
    top_pos, top_neg = build_lr_top_features(model.vectorizer, model.clf, top_k=top_k)
    pos_rows = [{"feature": f, "category": categorize_feature(f)} for f in top_pos]
    neg_rows = [{"feature": f, "category": categorize_feature(f)} for f in top_neg]
    return {"top_for_class_1": pos_rows, "top_for_class_0": neg_rows}

def pick_showcase_examples(
    texts: List[str],
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    n_per_class: int = 5
) -> List[int]:
    """
    Choose high-confidence correct examples for each class.
    """
    idxs = np.arange(len(texts))
    correct = idxs[y_pred == y_true]
    chosen = []
    for cls in [0, 1]:
        cls_idxs = correct[y_true[correct] == cls]
        if len(cls_idxs) == 0:
            continue
        # Sort by confidence margin
        conf = y_proba[cls_idxs]
        margin = np.abs(conf - 0.5)
        order = cls_idxs[np.argsort(margin)[::-1]]
        chosen.extend(order[:n_per_class].tolist())
    return chosen[:2 * n_per_class]

def error_theme_tags(text: str) -> List[str]:
    """
    Lightweight qualitative tags for error analysis. You can refine manually later.
    """
    t = text.lower()
    tags = []
    if MBTI_RE.search(t):
        tags.append("explicit_mbti")
    if any(w in t for w in ["subreddit", "r/", "upvote", "downvote"]):
        tags.append("reddit_meta")
    if any(w in t for w in ["meaning", "concept", "theory", "idea", "interpret", "possibility"]):
        tags.append("abstract_language")
    if any(w in t for w in ["smell", "taste", "touch", "sound", "see", "color", "texture", "weather", "food"]):
        tags.append("sensory_language")
    if len(tags) == 0:
        tags.append("misc")
    return tags

# -----------------------------
# Obfuscation pipeline runner
# -----------------------------
@dataclass
class ObfuscationConfig:
    do_lexical_remove: bool = True
    do_high_weight_sub: bool = True
    do_char_visual: bool = True
    do_paraphrase: bool = False
    do_translation: bool = False

    homoglyph_rate: float = 0.02
    diacritic_rate: float = 0.01
    seed: int = 42

def apply_obfuscations(
    texts: List[str],
    cfg: ObfuscationConfig,
    sensitive_feats: Optional[List[str]] = None
) -> List[str]:
    out = []
    for i, t in enumerate(texts):
        x = normalize_text(t)

        if cfg.do_lexical_remove:
            x = obfuscate_lexical_cue_removal(x)

        if cfg.do_high_weight_sub and sensitive_feats:
            x = obfuscate_high_weight_substitution(
                x,
                sensitive_tokens=sensitive_feats,
                substitute_mode="synonym_then_neutral",
                neutral_token="neutral",
                max_replacements=30
            )

        if cfg.do_char_visual:
            x = obfuscate_char_visual(
                x,
                homoglyph_rate=cfg.homoglyph_rate,
                diacritic_rate=cfg.diacritic_rate,
                seed=cfg.seed + i
            )

        if cfg.do_paraphrase:
            x = paraphrase_transformers(x)

        if cfg.do_translation:
            x = roundtrip_translate_transformers(x)

        out.append(x)
    return out

# -----------------------------
# Main experiment
# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="sensing_intuitive.xlsx")
    parser.add_argument("--out_dir", type=str, default="results_si")
    parser.add_argument("--seed", type=int, default=42)

    # Optional heavy transformations
    parser.add_argument("--run_paraphrase", action="store_true", help="Enable paraphrasing (requires transformers+torch).")
    parser.add_argument("--run_translation", action="store_true", help="Enable round-trip translation (requires transformers+torch).")

    # Obfuscation options
    parser.add_argument("--no_lexical_remove", action="store_true")
    parser.add_argument("--no_high_weight_sub", action="store_true")
    parser.add_argument("--no_char_visual", action="store_true")
    parser.add_argument("--homoglyph_rate", type=float, default=0.02)
    parser.add_argument("--diacritic_rate", type=float, default=0.01)

    # Qualitative
    parser.add_argument("--topk_features", type=int, default=30)
    parser.add_argument("--showcase_per_class", type=int, default=5)

    args = parser.parse_args()
    set_seed(args.seed)
    ensure_out_dir(args.out_dir)

    df = load_si_xlsx(args.data)
    df["text"] = df["text"].map(normalize_text)

    # Author-level split
    splits = author_level_split(df, seed=args.seed, train_size=0.8, dev_size=0.1, test_size=0.1)

    # Train LR baseline
    lr_model = train_logreg(splits.X_train, splits.y_train)

    # Build sensitive feature list for substitution:
    # We use top features from BOTH classes; focus on unigrams + short phrases.
    top_pos, top_neg = build_lr_top_features(lr_model.vectorizer, lr_model.clf, top_k=300)
    sensitive_feats = []
    for f in (top_pos + top_neg):
        # ignore very short and numeric-ish features
        if len(f) < 3:
            continue
        if re.search(r"^\d+$", f):
            continue
        sensitive_feats.append(f)

    # Evaluate LR on original test
    lr_pred, lr_prob = predict_logreg(lr_model, splits.X_test)
    base_lr_metrics = compute_metrics(splits.y_test, lr_pred)

    results = {
        "data": {
            "n_rows": int(len(df)),
            "n_authors": int(df["author_id"].nunique()),
            "split_sizes": {
                "train": int(len(splits.X_train)),
                "dev": int(len(splits.X_dev)),
                "test": int(len(splits.X_test)),
            }
        },
        "baseline": {
            "logreg": base_lr_metrics,
        },
        "obfuscations": {},
        "qualitative": {}
    }

    # Train fastText (required)
    ft_dir = os.path.join(args.out_dir, "fasttext")
    ensure_out_dir(ft_dir)
    ft_model = train_fasttext(splits.X_train, splits.y_train, splits.X_dev, splits.y_dev, ft_dir, seed=args.seed)
    ft_pred, ft_prob = predict_fasttext(ft_model, splits.X_test)
    results["baseline"]["fasttext"] = compute_metrics(splits.y_test, ft_pred)

    # Qualitative artifact 1: Top features table
    results["qualitative"]["lr_top_features"] = build_feature_table(lr_model, top_k=args.topk_features)

    # Define obfuscation configs to test separately + a combined pipeline
    configs = {
        "lexical_remove": ObfuscationConfig(
            do_lexical_remove=not args.no_lexical_remove,
            do_high_weight_sub=False,
            do_char_visual=False,
            do_paraphrase=False,
            do_translation=False,
            seed=args.seed
        ),
        "high_weight_sub": ObfuscationConfig(
            do_lexical_remove=False,
            do_high_weight_sub=not args.no_high_weight_sub,
            do_char_visual=False,
            do_paraphrase=False,
            do_translation=False,
            seed=args.seed
        ),
        "char_visual": ObfuscationConfig(
            do_lexical_remove=False,
            do_high_weight_sub=False,
            do_char_visual=not args.no_char_visual,
            do_paraphrase=False,
            do_translation=False,
            homoglyph_rate=args.homoglyph_rate,
            diacritic_rate=args.diacritic_rate,
            seed=args.seed
        ),
        "combined_core": ObfuscationConfig(
            do_lexical_remove=not args.no_lexical_remove,
            do_high_weight_sub=not args.no_high_weight_sub,
            do_char_visual=not args.no_char_visual,
            do_paraphrase=False,
            do_translation=False,
            homoglyph_rate=args.homoglyph_rate,
            diacritic_rate=args.diacritic_rate,
            seed=args.seed
        )
    }

    # Optional heavy configs
    if args.run_paraphrase:
        configs["paraphrase_only"] = ObfuscationConfig(
            do_lexical_remove=False,
            do_high_weight_sub=False,
            do_char_visual=False,
            do_paraphrase=True,
            do_translation=False,
            seed=args.seed
        )
        configs["combined_plus_paraphrase"] = ObfuscationConfig(
            do_lexical_remove=not args.no_lexical_remove,
            do_high_weight_sub=not args.no_high_weight_sub,
            do_char_visual=not args.no_char_visual,
            do_paraphrase=True,
            do_translation=False,
            homoglyph_rate=args.homoglyph_rate,
            diacritic_rate=args.diacritic_rate,
            seed=args.seed
        )

    if args.run_translation:
        configs["translation_only"] = ObfuscationConfig(
            do_lexical_remove=False,
            do_high_weight_sub=False,
            do_char_visual=False,
            do_paraphrase=False,
            do_translation=True,
            seed=args.seed
        )
        configs["combined_plus_translation"] = ObfuscationConfig(
            do_lexical_remove=not args.no_lexical_remove,
            do_high_weight_sub=not args.no_high_weight_sub,
            do_char_visual=not args.no_char_visual,
            do_paraphrase=False,
            do_translation=True,
            homoglyph_rate=args.homoglyph_rate,
            diacritic_rate=args.diacritic_rate,
            seed=args.seed
        )

    # Run obfuscations + evaluate
    for name, cfg in configs.items():
        X_obf = apply_obfuscations(splits.X_test, cfg, sensitive_feats=sensitive_feats)

        # LR evaluation
        pred_lr, prob_lr = predict_logreg(lr_model, X_obf)
        results["obfuscations"].setdefault(name, {})
        results["obfuscations"][name]["logreg"] = compute_metrics(splits.y_test, pred_lr)

        # fastText evaluation
        pred_ft, prob_ft = predict_fasttext(ft_model, X_obf)
        results["obfuscations"][name]["fasttext"] = compute_metrics(splits.y_test, pred_ft)

        # Qualitative artifact 2: Before/after showcase (only for combined_core to keep report manageable)
        if name == "combined_core":
            # choose examples based on LR baseline predictions (original)
            base_pred, base_prob = lr_pred, lr_prob
            chosen_idx = pick_showcase_examples(
                splits.X_test, splits.y_test, base_pred, base_prob, n_per_class=args.showcase_per_class
            )
            showcase = []
            for i in chosen_idx:
                orig = splits.X_test[i]
                obf = X_obf[i]
                # predictions
                p0, pr0 = int(base_pred[i]), float(base_prob[i])
                p1, pr1 = int(pred_lr[i]), float(prob_lr[i])

                showcase.append({
                    "index": int(i),
                    "true_label": int(splits.y_test[i]),
                    "orig_excerpt": orig[:500],
                    "obf_excerpt": obf[:500],
                    "logreg_before": {"pred": p0, "p(class=1)": pr0},
                    "logreg_after": {"pred": p1, "p(class=1)": pr1},
                })

            results["qualitative"]["showcase_examples_combined_core"] = showcase

            # Qualitative artifact 3: Error analysis themes
            # Compare flips vs stays-correct after obfuscation
            flips = []
            stays = []
            for i in range(len(splits.X_test)):
                before_ok = (lr_pred[i] == splits.y_test[i])
                after_ok = (pred_lr[i] == splits.y_test[i])
                if before_ok and not after_ok:
                    flips.append(i)
                if before_ok and after_ok:
                    stays.append(i)

            rng = random.Random(args.seed)
            rng.shuffle(flips)
            rng.shuffle(stays)
            flips = flips[:40]
            stays = stays[:40]

            def summarize_idxs(idxs: List[int], label: str) -> Dict:
                theme_counts: Dict[str, int] = {}
                ex = []
                for j in idxs[:20]:
                    tags = error_theme_tags(X_obf[j])
                    for tg in tags:
                        theme_counts[tg] = theme_counts.get(tg, 0) + 1
                    ex.append({
                        "index": int(j),
                        "true": int(splits.y_test[j]),
                        "pred_after": int(pred_lr[j]),
                        "tags": tags,
                        "excerpt": X_obf[j][:350]
                    })
                return {"group": label, "theme_counts": theme_counts, "examples": ex}

            results["qualitative"]["error_analysis_combined_core"] = {
                "flips_correct_to_wrong": summarize_idxs(flips, "flips"),
                "stays_correct": summarize_idxs(stays, "stays")
            }

    # Save results
    save_json(results, os.path.join(args.out_dir, "results.json"))

    # Print compact summary
    print("\n=== DATA ===")
    print(f"Rows: {results['data']['n_rows']}, Authors: {results['data']['n_authors']}")
    print(f"Split sizes: {results['data']['split_sizes']}")

    print("\n=== BASELINES ===")
    print("LogReg:", results["baseline"]["logreg"])
    print("fastText:", results["baseline"]["fasttext"])

    print("\n=== OBFUSCATIONS (LogReg macro-F1) ===")
    for name in results["obfuscations"]:
        print(f"{name:>22s} -> {results['obfuscations'][name]['logreg']}")

    print(f"\nSaved full output to: {os.path.join(args.out_dir, 'results.json')}")
    print("Qualitative artifacts are in results.json under ['qualitative'].")

if __name__ == "__main__":
    main()
