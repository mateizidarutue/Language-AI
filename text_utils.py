import re

_URL_RE = re.compile(r"https?://\S+|www\.\S+")
_WS_RE = re.compile(r"\s+")
WS_RE = _WS_RE


def normalize_text(t: str) -> str:
    t = _URL_RE.sub(" URL ", t)
    t = t.replace("\u200b", "")  # zero-width space
    t = _WS_RE.sub(" ", t).strip()
    return t
