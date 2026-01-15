"""
obfs package: obfuscation methods used by the experiment runner.

This file intentionally keeps imports lightweight and robust:
- If optional dependencies for some obfuscations are missing (e.g., transformers, torch, pillow),
  importing `obfs` still works, and only the specific method will be unavailable.
"""

# Always-available obfuscations (no heavy deps)
from .obfuscation_char_visual import viper
from .obfuscation_lexical import obfuscate_lexical_cue_removal

# High-weight substitution (may require transformers/torch/nltk/spacy depending on config)
try:
    from .obfuscation_high_weight import obfuscate_high_weight_substitution, SubstitutionConfig
except Exception:
    obfuscate_high_weight_substitution = None
    SubstitutionConfig = None

# Paraphrasing (requires transformers + torch; optional sentence-transformers)
try:
    from .obfuscation_paraphrase import paraphrase_obfuscate, paraphrase_transformers_batch
except Exception:
    paraphrase_obfuscate = None
    paraphrase_transformers_batch = None

# Translation (requires transformers + torch + sentencepiece)
try:
    from .obfuscation_translation import roundtrip_translate_transformers_batch
except Exception:
    roundtrip_translate_transformers_batch = None


__all__ = [
    "viper",
    "obfuscate_lexical_cue_removal",
    "obfuscate_high_weight_substitution",
    "SubstitutionConfig",
    "paraphrase_obfuscate",
    "paraphrase_transformers_batch",
    "roundtrip_translate_transformers_batch",
]
