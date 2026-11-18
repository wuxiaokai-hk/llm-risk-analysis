"""Gather everything the checks need: data, the model's recorded outputs, parsed labels and benchmarks."""
import os

import pandas as pd
import yaml

from . import lexicon
from .backends import Baseline
from .data import load_splits, load_recorded, source_map, enrich, SEEDS
from .parse import first_label, strict_label, mentions, negated_label


def load_config(path):
    path = os.path.abspath(path)
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    base = os.path.dirname(path)
    cfg["_base"] = base
    cfg["data_root"] = os.path.normpath(os.path.join(base, cfg["data_root"]))
    rec = cfg["model"]["recorded"]
    rec["path"] = os.path.normpath(os.path.join(cfg["data_root"] if rec["kind"] == "shah_chatgpt" else base,
                                                rec["path"]))
    return cfg


class Evidence:
    def __init__(self, cfg):
        self.cfg = cfg
        self.splits = load_splits(cfg["data_root"])
        self.smap = source_map(cfg["data_root"])
        rec = load_recorded(cfg, self.splits)
        rec["pred"] = rec.raw.map(first_label)
        rec["strict"] = rec.raw.map(strict_label)
        rec["n_labels"] = rec.raw.map(mentions)
        rec["negated"] = rec.raw.map(negated_label)
        rec["chars"] = rec.raw.str.len()
        self.rec = enrich(rec, self.smap)
        self.model = cfg["model"]["name"]

        # benchmarks, scored on exactly the same test sentences
        rows = []
        self.baselines = {}
        for s in SEEDS:
            tr, te = self.splits[s]
            b = Baseline(tr)
            self.baselines[s] = b
            maj = tr.label.mode()[0]
            rows.append(pd.DataFrame({"split": s, "sentence": te.sentence, "label": te.label,
                                      "majority": maj, "lexicon": te.sentence.map(lexicon.score),
                                      "tfidf": b.classify(te.sentence)}))
        self.bench = pd.concat(rows, ignore_index=True)
        if (self.rec.split != "own").all():
            self.rec = self.rec.merge(self.bench[["split", "sentence", "majority", "lexicon", "tfidf"]]
                                      .drop_duplicates(["split", "sentence"]), on=["split", "sentence"], how="left")
