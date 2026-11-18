"""The output contract: how a free-text answer becomes a label, and how strictly it follows the instruction."""
import re

from .data import LABELS

WORD = re.compile(r"[a-z]+")


def first_label(text):
    """Label word that appears first anywhere in the output ('unknown' if none). Lenient: the usual parser."""
    t = str(text).lower()
    hits = [(t.find(l), l) for l in LABELS if l in t]
    return min(hits)[1] if hits else "unknown"


def strict_label(text):
    """Label only if the output's first word is a label (what the instruction asked for), else 'unknown'."""
    w = WORD.findall(str(text).lower())
    return w[0] if w and w[0] in LABELS else "unknown"


def mentions(text):
    """How many distinct label words the output contains (more than one = the parse is ambiguous)."""
    t = str(text).lower()
    return sum(1 for l in LABELS if re.search(rf"\b{l}\b", t))


NEGATED = re.compile(r"\b(not|no|neither|nor|isn't|is not|doesn't|without)\b[^.]{0,30}\b(hawkish|dovish|neutral)\b")


def negated_label(text):
    """True when the output says 'not hawkish' or similar: a lenient parser can read the opposite."""
    return bool(NEGATED.search(str(text).lower()))
