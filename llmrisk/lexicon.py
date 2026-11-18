"""A transparent keyword benchmark: counts hawkish and dovish cue words. What any model should beat."""
import re

HAWKISH = ["tighten", "tightening", "firming", "firm", "raise", "raising", "hike", "higher rates", "restrictive",
           "inflation pressures", "elevated inflation", "upward pressure", "overheating", "remove accommodation",
           "reduce accommodation", "increase in the target", "upside risks to inflation"]
DOVISH = ["ease", "easing", "accommodation", "accommodative", "lower", "lowering", "cut", "reduce the target",
          "weak", "weakness", "slack", "downside risks", "unemployment remains elevated", "subdued", "deteriorat",
          "below the committee's", "support the economy"]


def score(sentence):
    s = sentence.lower()
    h = sum(len(re.findall(r"\b" + re.escape(w), s)) for w in HAWKISH)
    d = sum(len(re.findall(r"\b" + re.escape(w), s)) for w in DOVISH)
    return "hawkish" if h > d else "dovish" if d > h else "neutral"
