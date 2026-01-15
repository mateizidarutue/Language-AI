# obfuscation_char_visual.py
"""
VIPER-style visual obfuscation based on:
Eger et al. (NAACL 2019): "Text Processing Like Humans Do: Visually Attacking and Shielding NLP Systems"

Implements three character embedding spaces (CES):
- ECES: simple handpicked diacritic variants (minimal perturbation)
- DCES: Unicode-name-based variants (same base letter/case)
- ICES: image-based vector space (24x24 glyph bitmap -> 576-dim vector), then nearest neighbors

Main entry:
    viper(text, p=0.2, ces="ECES", k=20, seed=42)
"""

from __future__ import annotations

import random
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:
    Image = None
    ImageDraw = None
    ImageFont = None


# -----------------------------
# ECES: minimal perturbations (one or few neighbors)
# -----------------------------
# Keep this small and readable: “minimal perturbance with maximal impact” (paper’s ECES idea)
_ECES: Dict[str, List[str]] = {
    # lowercase
    "a": ["á", "à", "â", "ä", "ã", "å"],
    "c": ["ç", "ć", "č"],
    "e": ["é", "è", "ê", "ë"],
    "i": ["í", "ì", "î", "ï"],
    "n": ["ñ"],
    "o": ["ó", "ò", "ô", "ö", "õ"],
    "u": ["ú", "ù", "û", "ü"],
    "y": ["ý", "ÿ"],
    "s": ["ś", "š"],
    "z": ["ź", "ž"],
    # uppercase (mirror the same idea)
    "A": ["Á", "À", "Â", "Ä", "Ã", "Å"],
    "C": ["Ç", "Ć", "Č"],
    "E": ["É", "È", "Ê", "Ë"],
    "I": ["Í", "Ì", "Î", "Ï"],
    "N": ["Ñ"],
    "O": ["Ó", "Ò", "Ô", "Ö", "Õ"],
    "U": ["Ú", "Ù", "Û", "Ü"],
    "Y": ["Ý", "Ÿ"],
    "S": ["Ś", "Š"],
    "Z": ["Ź", "Ž"],
}


# -----------------------------
# DCES: Unicode-name-based neighbors
# -----------------------------
def _unicode_name(ch: str) -> str:
    try:
        return unicodedata.name(ch)
    except ValueError:
        return ""


def _dces_neighbors_for_letter(letter: str) -> List[str]:
    """
    Approximate DCES:
    For a given Latin letter (same case), return Latin letters with diacritics that share the same base,
    using Unicode names like "LATIN SMALL LETTER A WITH GRAVE".
    """
    name = _unicode_name(letter)
    if not name:
        return []
    # Example: "LATIN SMALL LETTER A"
    # We match all "LATIN SMALL LETTER A WITH ..."
    base = name
    if " WITH " in base:
        base = base.split(" WITH ")[0]

    # Search in a limited Unicode range for performance.
    # Latin Extended blocks mostly live in these ranges.
    ranges = [
        (0x00C0, 0x017F),  # Latin-1 Supplement + Latin Extended-A
        (0x0180, 0x024F),  # Latin Extended-B
        (0x1E00, 0x1EFF),  # Latin Extended Additional
    ]
    out = []
    for lo, hi in ranges:
        for cp in range(lo, hi + 1):
            ch = chr(cp)
            n = _unicode_name(ch)
            if not n:
                continue
            if n.startswith(base + " WITH "):
                out.append(ch)

    # Remove the original letter if present
    out = [c for c in out if c != letter]
    return out


@lru_cache(maxsize=128)
def _dces_table() -> Dict[str, List[str]]:
    tbl: Dict[str, List[str]] = {}
    for cp in range(ord("a"), ord("z") + 1):
        c = chr(cp)
        tbl[c] = _dces_neighbors_for_letter(c)
    for cp in range(ord("A"), ord("Z") + 1):
        c = chr(cp)
        tbl[c] = _dces_neighbors_for_letter(c)
    return tbl


# -----------------------------
# ICES: image-based vector space neighbors (24x24 bitmap -> 576-dim)
# -----------------------------
@dataclass(frozen=True)
class ICESConfig:
    size: int = 24
    font_size: int = 20  # fits 24x24 reasonably
    # If you want to use a specific TTF, set font_path; otherwise PIL default font is used.
    font_path: Optional[str] = None


def _require_pillow():
    if Image is None:
        raise RuntimeError(
            "Pillow is required for ICES (image-based character embeddings). "
            "Install it with: pip install pillow"
        )


@lru_cache(maxsize=1024)
def _render_char_to_vec(ch: str, cfg: ICESConfig) -> np.ndarray:
    """
    Render character into a cfg.size x cfg.size grayscale image and return flattened vector in [0,1].
    """
    _require_pillow()

    img = Image.new("L", (cfg.size, cfg.size), color=255)  # white background
    draw = ImageDraw.Draw(img)

    if cfg.font_path:
        font = ImageFont.truetype(cfg.font_path, cfg.font_size)
    else:
        font = ImageFont.load_default()

    # Center the glyph
    bbox = draw.textbbox((0, 0), ch, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = (cfg.size - w) // 2
    y = (cfg.size - h) // 2
    draw.text((x, y), ch, fill=0, font=font)  # black glyph

    arr = np.asarray(img, dtype=np.float32) / 255.0
    return arr.reshape(-1)  # 576-dim for 24x24


def _candidate_chars_for_ices() -> List[str]:
    """
    Keep this candidate set realistic and not huge.
    Use Latin + Latin Extended + a small homoglyph-ish set.
    """
    chars = []

    # Basic ASCII letters/digits/punct
    for cp in range(32, 127):
        chars.append(chr(cp))

    # Latin-1 Supplement + Latin Extended-A
    for cp in range(0x00C0, 0x017F + 1):
        chars.append(chr(cp))

    # Latin Extended Additional (often diacritics)
    for cp in range(0x1E00, 0x1EFF + 1):
        chars.append(chr(cp))

    # Remove control chars and blanks
    chars = [c for c in chars if c.strip() != ""]
    return chars


@lru_cache(maxsize=8)
def _ices_neighbors_table(cfg: ICESConfig, k: int = 20) -> Dict[str, List[Tuple[str, float]]]:
    """
    Build nearest-neighbor table for a candidate charset using cosine distance.
    Returns mapping: char -> list[(neighbor, distance)] sorted ascending by distance.
    """
    _require_pillow()

    cand = _candidate_chars_for_ices()
    vecs = np.stack([_render_char_to_vec(c, cfg) for c in cand], axis=0)

    # Normalize for cosine distance
    norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8
    vecs_n = vecs / norms

    table: Dict[str, List[Tuple[str, float]]] = {}

    # Precompute dot products row-by-row (OK for moderate cand size)
    # If this ever gets slow, we can restrict cand further or approximate.
    for i, c in enumerate(cand):
        sims = vecs_n @ vecs_n[i]  # cosine similarity
        # exclude itself
        sims[i] = -1.0
        # take top-k most similar
        idx = np.argpartition(-sims, kth=min(k, len(cand)-1)-1)[:k]
        # sort those
        idx = idx[np.argsort(-sims[idx])]
        neighbors = []
        for j in idx:
            # cosine distance = 1 - cosine similarity
            dist = float(1.0 - sims[j])
            neighbors.append((cand[j], dist))
        table[c] = neighbors

    return table


def _sample_neighbor_weighted(rng: random.Random, neighbors: List[Tuple[str, float]]) -> str:
    """
    VIPER samples among neighbors, optionally proportional to distance (paper: proportional to distance).
    Distance-proportional makes farther neighbors more likely (stronger).
    We'll do inverse-distance weighting by default for "nearest emphasis" unless you want the stronger version.
    """
    if not neighbors:
        return ""

    dists = np.array([max(d, 1e-6) for _, d in neighbors], dtype=np.float64)

    # Choose ONE:
    # 1) nearest-emphasis (more human-readable):
    weights = 1.0 / dists
    # 2) distance-proportional (stronger, as mentioned in the paper):
    # weights = dists

    weights = weights / weights.sum()
    pick = rng.choices(range(len(neighbors)), weights=weights.tolist(), k=1)[0]
    return neighbors[pick][0]


# -----------------------------
# Public API
# -----------------------------
def viper(
    text: str,
    p: float = 0.2,
    ces: str = "ECES",
    k: int = 20,
    seed: int = 42,
    ices_cfg: Optional[ICESConfig] = None,
) -> str:
    """
    VIPER(p, CES): for each character, with probability p replace by a visual neighbor from CES.
    CES in {"ECES","DCES","ICES"}.
    """
    if p <= 0.0:
        return text
    if p > 1.0:
        p = 1.0

    rng = random.Random(seed)
    chars = list(text)

    ces_upper = ces.upper()
    dces_tbl = _dces_table() if ces_upper == "DCES" else None
    ices_tbl = None
    if ces_upper == "ICES":
        ices_cfg = ices_cfg or ICESConfig()
        ices_tbl = _ices_neighbors_table(ices_cfg, k=k)

    for i, ch in enumerate(chars):
        if rng.random() >= p:
            continue

        # Skip whitespace to keep readability
        if ch.isspace():
            continue

        repl = None

        if ces_upper == "ECES":
            neigh = _ECES.get(ch)
            if neigh:
                repl = rng.choice(neigh)

        elif ces_upper == "DCES":
            neigh = dces_tbl.get(ch) if dces_tbl else []
            if neigh:
                repl = rng.choice(neigh)

        elif ces_upper == "ICES":
            neigh = ices_tbl.get(ch, []) if ices_tbl else []
            if neigh:
                repl = _sample_neighbor_weighted(rng, neigh)

        if repl:
            chars[i] = repl

    return "".join(chars)
