"""Data: the labelled FOMC sentences of Shah, Paturi & Chava (2023) and the recorded outputs of a model.

The authors' repo (github.com/gtfintechlab/fomc-hawkish-dovish, CC BY-NC 4.0) holds three random train/test
splits (seeds 5768, 78516, 944601) of the combined dataset, and ChatGPT's zero-shot answers on each test split.
`tools/get_data.py` clones it into data/fomc-hawkish-dovish.
"""
import json
import os
import re

import pandas as pd

LABELS = ["dovish", "hawkish", "neutral"]
INT2LABEL = {0: "dovish", 1: "hawkish", 2: "neutral"}
SEEDS = [5768, 78516, 944601]
CONTRAST = re.compile(r"\b(but|however|although|though|while|whereas|nevertheless|nonetheless|yet)\b", re.I)


def _norm(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def load_splits(root):
    """{seed: (train_df, test_df)} with columns sentence, year, label (string)."""
    tt = os.path.join(root, "training_data", "test-and-training")
    out = {}
    for s in SEEDS:
        tr = pd.read_excel(os.path.join(tt, "training_data", f"lab-manual-combine-train-{s}.xlsx"))
        te = pd.read_excel(os.path.join(tt, "test_data", f"lab-manual-combine-test-{s}.xlsx"))
        for d in (tr, te):
            d["label"] = d["label"].map(INT2LABEL)
        out[s] = (tr[["sentence", "year", "label"]].reset_index(drop=True),
                  te[["sentence", "year", "label"]].reset_index(drop=True))
    return out


def source_map(root):
    """sentence -> document type (meeting minutes, speech, press conference)."""
    ad = os.path.join(root, "data", "annotated_data")
    m = {}
    for code, name in [("mm", "meeting minutes"), ("sp", "speech"), ("pc", "press conference")]:
        d = pd.read_excel(os.path.join(ad, f"manual-{code}.xlsx"))
        for s in d["sentence"]:
            m.setdefault(_norm(s), name)
    return m


def load_recorded(cfg, splits):
    """The model under review's recorded outputs: one row per (split, sentence) with the raw text output.

    kind 'shah_chatgpt': the authors' CSVs, aligned row by row with the test splits.
    kind 'jsonl': files with {"sentence", "gold", "raw"} per line (e.g. fed-tone-finetune results).
    """
    rec = cfg["model"]["recorded"]
    rows = []
    if rec["kind"] == "shah_chatgpt":
        for s in SEEDS:
            c = pd.read_csv(rec["path"].format(seed=s))
            te = splits[s][1]
            if not (c["original_sent"].map(_norm).values == te["sentence"].map(_norm).values).all():
                raise ValueError(f"recorded outputs for seed {s} are not aligned with the test split")
            rows.append(pd.DataFrame({"split": s, "sentence": te.sentence, "year": te.year, "label": te.label,
                                      "raw": c["text_output"].astype(str)}))
    elif rec["kind"] == "jsonl":
        with open(rec["path"], encoding="utf-8") as f:
            recs = [json.loads(line) for line in f if line.strip()]
        d = pd.DataFrame(recs)
        rows.append(pd.DataFrame({"split": "own", "sentence": d["sentence"], "year": d.get("year"),
                                  "label": d["gold"], "raw": d["raw"].astype(str)}))
    else:
        raise ValueError(rec["kind"])
    return pd.concat(rows, ignore_index=True)


def enrich(df, smap):
    df = df.copy()
    df["source"] = df.sentence.map(lambda s: smap.get(_norm(s), "unknown"))
    df["contrast"] = df.sentence.str.contains(CONTRAST)
    df["words"] = df.sentence.str.split().str.len()
    y = pd.to_numeric(df.year, errors="coerce")
    df["era"] = pd.cut(y, [0, 2007, 2015, 3000], labels=["1996-2007", "2008-2015 (crisis, zero bound)", "2016-2022"])
    return df
