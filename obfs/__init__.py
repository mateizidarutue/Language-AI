"""
obfs package: obfuscation methods used by the experiment runner.


"""


from .obfuscation_char_visual import viper
from .obfuscation_lexical import obfuscate_lexical_cue_removal

# High-weight substitution (may require transformers/torch/nltk/spacy depending on config)
try:
    from .obfuscation_high_weight import obfuscate_high_weight_substitution, SubstitutionConfig
except Exception:
    obfuscate_high_weight_substitution = None
    SubstitutionConfig = None

__all__ = [
    "viper",
    "obfuscate_lexical_cue_removal",
    "obfuscate_high_weight_substitution",
    "SubstitutionConfig",
]
