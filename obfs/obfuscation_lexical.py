

import re
from text_utils import WS_RE


# -----------------------------
# Gender leakage patterns


# Explicit gender words
GENDER_WORDS = r"\b(woman|women|man|men|female|male|girl|boy|guy|gal)\b"

# Relationships / family roles (strong explicit cues)
RELATIONSHIP_ROLES = r"\b(husband|wife|boyfriend|girlfriend|fianc[eé]|spouse)\b"
FAMILY_ROLES = r"\b(mother|father|mom|mum|dad|sister|brother|daughter|son)\b"

# Honorifics / titles
HONORIFICS = r"\b(mr|mrs|ms|miss|sir|ma'am|madam)\b"

# Explicit self-identification phrases
SELF_ID_1 = r"\b(i\s*(?:am|'m)\s*(?:a|an)\s*(?:woman|man|girl|boy|guy|female|male))\b"
SELF_ID_2 = r"\b(as\s+(?:a|an)\s+(?:woman|man|girl|boy|guy|female|male))\b"

GENDER_RE = re.compile(
    "|".join([
        GENDER_WORDS,
        RELATIONSHIP_ROLES,
        FAMILY_ROLES,
        HONORIFICS,
        SELF_ID_1,
        SELF_ID_2,
    ]),
    flags=re.IGNORECASE,
)

# Optional pronoun neutralization (off by default)
_PRONOUN_SUBS = [
    (re.compile(r"\bhe\b", flags=re.IGNORECASE), "they"),
    (re.compile(r"\bshe\b", flags=re.IGNORECASE), "they"),
    (re.compile(r"\bhim\b", flags=re.IGNORECASE), "them"),
    (re.compile(r"\bher\b", flags=re.IGNORECASE), "them"),
    (re.compile(r"\bhis\b", flags=re.IGNORECASE), "their"),
    (re.compile(r"\bhers\b", flags=re.IGNORECASE), "theirs"),
]


def obfuscate_lexical_cue_removal(
    text: str,
    neutralize_pronouns: bool = False,
    replacement: str = "[trait]",
) -> str:
    """
    Removes explicit gender-identifying lexical cues.

    Parameters
    ----------
    text : str
        Input text.
    neutralize_pronouns : bool
        If True, replace he/she/him/her/his/hers with they/them/their.
        OFF by default because it is more invasive.
    replacement : str
        Token used to replace removed cues.

    Returns
    -------
    str
        Obfuscated text.
    """
    t = text or ""

    # Remove explicit gender leakage
    t = GENDER_RE.sub(f" {replacement} ", t)

    # Optional pronoun neutralization
    if neutralize_pronouns:
        for rx, repl in _PRONOUN_SUBS:
            t = rx.sub(repl, t)

    # Normalize whitespace
    t = WS_RE.sub(" ", t).strip()
    return t
