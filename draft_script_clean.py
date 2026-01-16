"""
S/I Stylometry + Adversarial Obfuscation (SOBR-style) — Runner Script

Goal:
- Train baseline models (Logistic Regression TF–IDF, fastText supervised) on the S/I slice dataset
- Evaluate on clean test data (no obfuscation)
- Evaluate on test data with exactly one obfuscation at a time (or none)
- Save all results in one results.json for easy comparison to SOBR

Project layout assumption:
- draft_script.py (this file) is in project root
- text_utils.py is in project root
- obfuscation modules are in: obfs/
    obfs/obfuscation_lexical.py
    obfs/obfuscation_high_weight.py
    obfs/obfuscation_char_visual.py

Tip: ensure `obfs/__init__.py` exists so imports work reliably.
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from typing import Callable, Counter, Dict, List, Optional, Tuple
import inspect
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split


# -----------------------------
# Imports: obfuscations (kept external, not implemented here)
# -----------------------------
from obfs.obfuscation_lexical import obfuscate_lexical_cue_removal
from obfs.obfuscation_char_visual import viper
from obfs.obfuscation_high_weight import obfuscate_high_weight_substitution, SubstitutionConfig


# -----------------------------
# fastText wrapper
# -----------------------------
class FastTextWrapper:
    def __init__(
        self,
        dim: int = 200,
        lr: float = 0.1,
        epoch: int = 25,          
        wordNgrams: int = 2,     
        minCount: int = 3,       
        minn: int = 0,           
        maxn: int = 0,           
        thread: int = 8,         
    ):
        import fasttext
        self.fasttext = fasttext
        self.params = dict(
            dim=dim,
            lr=lr,
            epoch=epoch,
            wordNgrams=wordNgrams,
            minCount=minCount,
            minn=minn,
            maxn=maxn,
            thread=thread,
        )
        self.model = None


    @staticmethod
    def _to_ft_label(y: int) -> str:
        return f"__label__{int(y)}"

    def _write_train_file(self, path: str, texts: List[str], labels: List[int]) -> None:
        with open(path, "w", encoding="utf-8") as f:
            for t, y in zip(texts, labels):
                t = (t or "").replace("\n", " ").strip()
                f.write(f"{self._to_ft_label(y)} {t}\n")

    def fit(self, X_train: List[str], y_train: List[int], tmp_dir: str) -> None:
        os.makedirs(tmp_dir, exist_ok=True)
        train_path = os.path.join(tmp_dir, "ft_train.txt")
        self._write_train_file(train_path, X_train, y_train)
        self.model = self.fasttext.train_supervised(train_path, **self.params)

    def predict(self, X: List[str]) -> Tuple[np.ndarray, np.ndarray]:
        preds = []
        confs = []
        for t in X:
            t = (t or "").replace("\n", " ").strip()
            labels, probs = self.model.predict(t, k=1)  # IMPORTANT: pass a string, not [string]
            lab = labels[0] if labels else "__label__0"
            prob = float(probs[0]) if probs else 0.0
            preds.append(int(lab.replace("__label__", "")))
            confs.append(prob)
        return np.array(preds), np.array(confs)


# -----------------------------
# Data loading
# -----------------------------
def load_dataset(path: str) -> pd.DataFrame:
    """
    Loads either .xlsx or .csv with at least:
      - author_id column (or author_ID / auhtor_ID)
      - text column (or post)
      - label column (sensing / label / y)
    """
    if path.lower().endswith(".xlsx"):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    cols = {c.strip().lower(): c for c in df.columns}

    author_col = None
    for cand in ["author_id", "authorid", "author_id ", "author", "author_id\t", "author_id\n", "auhtor_id", "author_id", "author_id "]:
        if cand in cols:
            author_col = cols[cand]
            break
    if author_col is None:
        # common in your screenshot: "author_ID"
        for k in cols:
            if "author" in k:
                author_col = cols[k]
                break
    if author_col is None:
        raise ValueError(f"Could not find author column. Columns: {list(df.columns)}")

    text_col = None
    for cand in ["text", "post", "body"]:
        if cand in cols:
            text_col = cols[cand]
            break
    if text_col is None:
        raise ValueError(f"Could not find text/post column. Columns: {list(df.columns)}")

    label_col = None
    for cand in ["female", "label", "y"]:
        if cand in cols:
            label_col = cols[cand]
            break
    if label_col is None:
        raise ValueError(f"Could not find label column (female/label/y). Columns: {list(df.columns)}")

    out = df[[author_col, text_col, label_col]].copy()
    out.columns = ["author_id", "text", "label"]
    out["author_id"] = out["author_id"].astype(str)
    out["text"] = out["text"].astype(str).fillna("")
    # Convert female label safely
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    out = out.dropna(subset=["label"]).copy()
    out["label"] = out["label"].astype(int)

    # Optional but recommended: ensure binary
    out = out[out["label"].isin([0, 1])].copy()
    return out


@dataclass
class Split:
    X_train: List[str]
    y_train: List[int]
    X_dev: List[str]
    y_dev: List[int]
    X_test: List[str]
    y_test: List[int]


def author_level_split(
    df: pd.DataFrame,
    seed: int = 42,
    test_size: float = 0.10,
    dev_size: float = 0.10,
) -> Split:
    """
    Author-level *stratified* split:
    - Authors are split into train/dev/test
    - All slices from an author go to the same split (prevents leakage)
    - Stratification preserves label distribution across splits

    test_size and dev_size are fractions of the full dataset (by authors).
    Default: 80/10/10.
    """
    # One label per author (assumption holds for gender / MBTI attributes)
    author_df = (
        df.groupby("author_id")["label"]
        .agg(lambda s: s.mode().iloc[0] if not s.mode().empty else s.iloc[0])
        .reset_index()
    )

    authors = author_df["author_id"].tolist()
    labels = author_df["label"].tolist()

    # 1) Split off test authors (10%)
    train_dev_auth, test_auth = train_test_split(
        authors,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    # 2) Split train+dev into train and dev
    # dev_size is fraction of total; relative to remaining (1 - test_size)
    dev_frac_of_train_dev = dev_size / (1.0 - test_size)

    # Get labels for train_dev authors for stratification
    train_dev_author_df = author_df[author_df["author_id"].isin(train_dev_auth)]
    train_dev_labels = train_dev_author_df.set_index("author_id").loc[train_dev_auth, "label"].tolist()

    train_auth, dev_auth = train_test_split(
        train_dev_auth,
        test_size=dev_frac_of_train_dev,
        random_state=seed,
        stratify=train_dev_labels,
    )

    def pick(auth_list):
        part = df[df["author_id"].isin(auth_list)]
        return part["text"].tolist(), part["label"].tolist()

    X_train, y_train = pick(train_auth)
    X_dev, y_dev = pick(dev_auth)
    X_test, y_test = pick(test_auth)

    return Split(X_train, y_train, X_dev, y_dev, X_test, y_test)



# -----------------------------
# Models + evaluation
# -----------------------------
def train_logreg(X_train: List[str], y_train: List[int], seed: int = 42):
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9)
    Xv = vec.fit_transform(X_train)
    clf = LogisticRegression(max_iter=2000, random_state=seed, n_jobs=1)
    clf.fit(Xv, y_train)
    return vec, clf


def eval_model(y_true: List[int], y_pred: np.ndarray) -> Dict:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "report": classification_report(y_true, y_pred, digits=4, output_dict=True),
        "confusion": confusion_matrix(y_true, y_pred).tolist(),
    }

def _top_features_for_class(vec, lr, class_idx: int, top_k: int = 50) -> List[str]:
    """
    Returns a list of feature strings (unigrams/bigrams) that LR weights most for class_idx.
    Works for binary LR with coef_.shape == (1, n_features).
    """
    feat_names = np.array(vec.get_feature_names_out())
    coef = lr.coef_
    if coef.shape[0] == 1:
        # binary: coef_[0] corresponds to class 1; class 0 is -coef_[0]
        w = coef[0] if class_idx == 1 else -coef[0]
    else:
        w = coef[class_idx]

    top_idx = np.argsort(w)[::-1][:top_k]
    return feat_names[top_idx].tolist()


# -----------------------------
# Obfuscation registry
# -----------------------------
def identity(texts: List[str], **kwargs) -> List[str]:
    return texts


def build_obfuscation_registry(args, vec=None, lr=None) -> Dict[str, Callable[[List[str]], List[str]]]:
    """
    Returns name -> function(list[str]) -> list[str]
    The runner never implements obfuscations itself; it only calls modules.
    """
    registry: Dict[str, Callable[[List[str]], List[str]]] = {
        "none": lambda X: identity(X),
        "lexical_remove": lambda X: [obfuscate_lexical_cue_removal(t) for t in X],
    }

    # --- VIPER (char/visual) ---
    def _char_visual(X: List[str]) -> List[str]:
        out = []
        for i, t in enumerate(X):
            kw = dict(
                p=args.viper_p,
                ces=args.viper_ces,
                k=args.viper_k,
                seed=args.seed + i,  # deterministic but different per text
            )
            # Only pass font_path when ICES is selected AND a path is provided
            if args.viper_ces == "ICES" and getattr(args, "viper_font_path", ""):
                kw["font_path"] = args.viper_font_path
            out.append(viper(t, **kw))
        return out

    registry["char_visual"] = _char_visual

    # --- High-weight substitution needs LR + vectorizer (white-box) ---
    if vec is not None and lr is not None:
        def _hw(X: List[str]) -> List[str]:
            Xv = vec.transform(X)
            preds = lr.predict(Xv).tolist()

            cfg = SubstitutionConfig(
                synonym_only=(args.substitute_mode == "synonym_only"),
            )

            out = []
            for t, target in zip(X, preds):
                out.append(obfuscate_high_weight_substitution(t, lr, vec, target, cfg))
            return out

        registry["high_weight_sub"] = _hw


        def _light_chain(X: List[str]) -> List[str]:
            # Chain the three light obfuscations: lexical -> char/visual -> high-weight
            X1 = [obfuscate_lexical_cue_removal(t) for t in X]
            X2 = _char_visual(X1)
            X3 = _hw(X2)
            return X3

        registry["chained_light"] = _light_chain


    return registry



# -----------------------------
# Runner
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="Path to gender.xlsx (or .csv)")
    p.add_argument("--out_dir", default="results_si", help="Output directory (results.json + caches)")
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--test_size", type=float, default=0.10,
               help="Author-level test fraction (default 0.10)")
    p.add_argument("--dev_size", type=float, default=0.10,
               help="Author-level dev fraction (default 0.10)")


    # Which obfuscations to run:
    #   all  -> run everything available
    #   none -> only baseline / identity
    #   comma list -> e.g. none,lexical_remove,char_visual,chained_light
    p.add_argument("--obfuscations", default="all")

    # VIPER params
    p.add_argument("--viper_p", type=float, default=0.2, help="VIPER: probability of perturbing a character")
    p.add_argument("--viper_ces", default="ECES", choices=["ECES", "DCES", "ICES"], help="VIPER: character edit strategy")
    p.add_argument("--viper_k", type=int, default=20, help="VIPER: top-k confusable candidates per character")
    p.add_argument("--viper_font_path", default="", help="VIPER ICES: path to .ttf font (only needed for ICES)")

    # high-weight substitution mode
    p.add_argument("--substitute_mode", default="synonym_only", choices=["synonym_only", "synonym_then_neutral", "neutral"])

    return p.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    os.makedirs(args.out_dir, exist_ok=True)

    df = load_dataset(args.data)
    split = author_level_split(
    df,
    seed=args.seed,
    test_size=args.test_size,
    dev_size=args.dev_size
)
    from collections import Counter

    print("Train label distribution:", Counter(split.y_train))
    print("Dev label distribution:", Counter(split.y_dev))
    print("Test label distribution:", Counter(split.y_test))


    # Train baselines
    vec, lr = train_logreg(split.X_train, split.y_train, seed=args.seed)
    ft = FastTextWrapper()
    ft.fit(split.X_train, split.y_train, tmp_dir=args.out_dir)

    # Evaluate baselines on clean test set
    X_test_vec = vec.transform(split.X_test)
    lr_pred = lr.predict(X_test_vec)
    ft_pred, ft_conf = ft.predict(split.X_test)

    results = {
        "meta": {
            "data": args.data,
            "seed": args.seed,
            "n_train": len(split.X_train),
            "n_dev": len(split.X_dev),
            "n_test": len(split.X_test),
            "label_counts_test": {str(i): int(np.sum(np.array(split.y_test) == i)) for i in sorted(set(split.y_test))},
        },
        "baseline": {
            "logreg": eval_model(split.y_test, lr_pred),
            "fasttext": eval_model(split.y_test, ft_pred),
        },
        "obfuscations": {},
    }

    # Build registry AFTER training (so high_weight_sub can use LR)
    registry = build_obfuscation_registry(args, vec=vec, lr=lr)

    # Decide which experiments to run
    requested = [s.strip() for s in args.obfuscations.split(",")]
    if len(requested) == 1 and requested[0].lower() == "all":
        run_names = list(registry.keys())
    else:
        run_names = requested

    # Validate
    missing = [n for n in run_names if n not in registry]
    if missing:
        raise ValueError(
            f"Unknown/disabled obfuscations: {missing}\n"
            f"Available now: {sorted(registry.keys())}\n"
        )

    # Run each obfuscation and evaluate
    for name in run_names:
        print(f"\n=== Running obfuscation: {name} ===")
        X_obf = registry[name](split.X_test)

        lr_pred_obf = lr.predict(vec.transform(X_obf))
        ft_pred_obf, _ = ft.predict(X_obf)

        results["obfuscations"][name] = {
            "logreg": eval_model(split.y_test, lr_pred_obf),
            "fasttext": eval_model(split.y_test, ft_pred_obf),
        }

    # Save
    out_path = os.path.join(args.out_dir, "results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved:", out_path)
    print("Obfuscations evaluated:", list(results["obfuscations"].keys()))


if __name__ == "__main__":
    main()
