"""Section 6: behavioural tests that call a model live (CheckList-style: Ribeiro et al., 2020).

- B1 minimum functionality: unambiguous template sentences with a known stance
- B2 directional: swap raise/lower (or similar) in a template; the stance must flip
- B3 invariance: edits that should not change the stance (prefix, entity name, a typo, a number)
- B4 prompt sensitivity: the same sentences under three wordings of the instruction
- B5 determinism: the same calls twice

When the live model is not the model under review (e.g. the demo runs the TF-IDF benchmark), results are shown
as information only and do not affect the rating.
"""
import itertools
import random
import re

import pandas as pd

from . import Check
from .. import charts
from ..parse import first_label

ACTORS = ["The Committee", "Participants", "Policymakers", "FOMC members"]
TEMPLATES = [   # (sentence, stance, (word, replacement) whose swap must flip the stance, or None)
    ("{a} decided to raise the target range for the federal funds rate by {n} basis points.", "hawkish",
     ("raise", "lower")),
    ("{a} judged that further policy firming would likely be appropriate.", "hawkish", ("firming", "easing")),
    ("{a} expects to tighten monetary policy at coming meetings.", "hawkish", ("tighten", "ease")),
    ("{a} noted that inflation remained well above 2 percent and that further tightening may be needed.",
     "hawkish", None),
    ("{a} decided to lower the target range for the federal funds rate by {n} basis points.", "dovish",
     ("lower", "raise")),
    ("{a} judged that further policy easing would likely be appropriate.", "dovish", ("easing", "firming")),
    ("{a} expects to ease monetary policy at coming meetings.", "dovish", ("ease", "tighten")),
    ("{a} agreed that the weakening labor market called for additional monetary accommodation.", "dovish", None),
    ("{a} reviewed the staff's report on developments in foreign exchange markets.", "neutral", None),
    ("{a} will next meet on Tuesday and Wednesday of the following month.", "neutral", None),
    ("{a} discussed the schedule for publishing the minutes of its meetings.", "neutral", None),
]
FLIP = {"hawkish": "dovish", "dovish": "hawkish"}


def mft_cases():
    rows = []
    for (t, lab, sw), a, n in itertools.product(TEMPLATES, ACTORS, (25, 50)):
        if "{n}" not in t and n == 50:
            continue
        s = t.format(a=a, n=n)
        rows.append({"sentence": s, "expected": lab,
                     "flipped": s.replace(sw[0], sw[1], 1) if sw else None})
    return pd.DataFrame(rows)


def _typo(s, rng):
    words = s.split()
    idx = [i for i, w in enumerate(words) if len(w) > 6 and w.isalpha()]
    if not idx:
        return s
    i = rng.choice(idx)
    w = words[i]
    j = rng.randrange(1, len(w) - 2)
    words[i] = w[:j] + w[j + 1] + w[j] + w[j + 2:]
    return " ".join(words)


PERTURB = {
    "prefix": lambda s, rng: "According to the record, " + s[0].lower() + s[1:],
    "entity": lambda s, rng: re.sub(r"\bthe Committee\b", "policymakers",
                                    re.sub(r"\bThe Committee\b", "Policymakers", s)),
    "typo": _typo,
    "number": lambda s, rng: re.sub(r"\b(\d)(\d)?\b", lambda m: str(int(m.group(0)) + 1), s, count=1),
}


def run(ev, out_dir):
    cfg, th = ev.cfg, ev.cfg["thresholds"]
    from ..backends import make
    seed0 = sorted(ev.splits)[0]
    model = make(cfg, ev.splits[seed0][0])
    review = bool(cfg.get("live", {}).get("same_model_as_reviewed"))
    tag = "" if review else f"[Live model: {model.name}; not the model under review, shown for information.] "

    def check(cid, title, ok, finding, metrics, threshold, sev, table=None, chart=None):
        status = ("pass" if ok else "fail" if sev == "high" else "warn") if review else "info"
        return Check(cid, "behaviour", title, status, tag + finding, metrics, threshold,
                     None if (ok or not review) else sev, table=table, chart=chart)

    checks = []
    m = mft_cases()
    m["pred"] = [first_label(x) for x in model.classify(m.sentence)]
    m["ok"] = m.pred == m.expected
    by = m.groupby("expected").ok.mean()
    rate = float(m.ok.mean())
    checks.append(check("B1", "Minimum functionality: unambiguous sentences", rate >= th["min_mft_pass"],
                        f"{rate:.0%} of {len(m)} unambiguous template sentences are labelled correctly "
                        f"(hawkish {by.get('hawkish', 0):.0%}, dovish {by.get('dovish', 0):.0%}, "
                        f"neutral {by.get('neutral', 0):.0%}).",
                        {"cases": len(m), "pass_rate": rate}, f">= {th['min_mft_pass']:.0%}", "high",
                        table=m[~m.ok][["sentence", "expected", "pred"]].head(20).reset_index(drop=True)))

    d = m[m.flipped.notna()].copy()
    d["pred_flipped"] = [first_label(x) for x in model.classify(d.flipped)]
    d["ok"] = d.pred_flipped == d.expected.map(FLIP)
    dr = float(d.ok.mean())
    checks.append(check("B2", "Directional: swapping 'raise' and 'lower' must flip the stance",
                        dr >= th["min_directional_pass"],
                        f"After swapping only the policy word ('raise' for 'lower', 'firming' for 'easing', 'tighten' for 'ease'), the "
                        f"label flips the right way in {dr:.0%} of {len(d)} pairs.",
                        {"pairs": len(d), "pass_rate": dr}, f">= {th['min_directional_pass']:.0%}", "high",
                        table=d[~d.ok][["flipped", "expected", "pred_flipped"]].head(20).reset_index(drop=True)))

    rng = random.Random(0)
    te = ev.splits[seed0][1]
    sample = te.sample(min(cfg["live"].get("sample", 150), len(te)), random_state=0).sentence.tolist()
    base = [first_label(x) for x in model.classify(sample)]
    rows = []
    for name, fn in PERTURB.items():
        pert = [fn(s, rng) for s in sample]
        changed = [i for i, (a, b) in enumerate(zip(sample, pert)) if a != b]
        if not changed:
            continue
        out = [first_label(x) for x in model.classify([pert[i] for i in changed])]
        flips = sum(o != base[i] for o, i in zip(out, changed))
        rows.append({"edit": name, "sentences_changed": len(changed), "label_flips": flips,
                     "flip_rate": flips / len(changed)})
    inv = pd.DataFrame(rows)
    worst = inv.sort_values("flip_rate").iloc[-1]
    checks.append(check("B3", "Invariance: edits that must not change the stance",
                        inv.flip_rate.max() <= th["max_invariance_flip"],
                        "Label changes after meaning-preserving edits: " + "; ".join(
                            f"{r.edit} {r.flip_rate:.0%}" for _, r in inv.iterrows()) +
                        f". The worst is '{worst.edit}'.",
                        {"max_flip_rate": float(inv.flip_rate.max()), "worst_edit": worst.edit},
                        f"flip rate <= {th['max_invariance_flip']:.0%}", "medium", table=inv))
    fig, ax = charts.plt.subplots(figsize=(6.2, 2.4))
    ax.barh(inv.edit, inv.flip_rate * 100, color=charts.SERIES[0], height=0.5)
    ax.axvline(th["max_invariance_flip"] * 100, color=charts.INK2, lw=0.8, ls="--")
    ax.invert_yaxis()
    ax.set_xlabel("% of labels that change (dashed: tolerance)")
    ax.set_title("Label flips after meaning-preserving edits")
    checks[-1].chart = charts.save(fig, out_dir, "b3_invariance.png")

    if model.uses_prompt:
        from ..backends import PROMPTS
        outs = {p: [first_label(x) for x in model.classify(sample, prompt=p)] for p in PROMPTS}
        agree = float(pd.DataFrame(outs).nunique(axis=1).eq(1).mean())
        checks.append(check("B4", "Prompt sensitivity: three wordings of the same instruction",
                            1 - agree <= th["max_prompt_flip"],
                            f"The three instruction wordings give the same label for {agree:.0%} of {len(sample)} "
                            "sentences.", {"agreement": agree}, f"disagreement <= {th['max_prompt_flip']:.0%}",
                            "medium"))
    else:
        checks.append(Check("B4", "behaviour", "Prompt sensitivity: three wordings of the same instruction", "info",
                            tag + "Not applicable: the live model does not take a prompt."))

    again = [first_label(x) for x in model.classify(sample)]
    same = float(sum(a == b for a, b in zip(base, again)) / len(base))
    checks.append(check("B5", "Determinism: identical calls give identical answers", same >= th["min_determinism"],
                        f"Repeating {len(sample)} identical calls gives the same label {same:.0%} of the time.",
                        {"agreement": same}, f">= {th['min_determinism']:.0%}", "medium"))
    return checks
