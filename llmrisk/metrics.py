"""Scores with uncertainty: macro-F1, bootstrap intervals and paired bootstrap differences."""
import numpy as np

from .data import LABELS


def macro_f1(y, p):
    y, p = np.asarray(y), np.asarray(p)
    f = []
    for l in LABELS:
        tp = np.sum((y == l) & (p == l))
        fp = np.sum((y != l) & (p == l))
        fn = np.sum((y == l) & (p != l))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def per_class(y, p):
    y, p = np.asarray(y), np.asarray(p)
    out = {}
    for l in LABELS:
        tp = np.sum((y == l) & (p == l))
        out[l] = {"precision": tp / max(np.sum(p == l), 1), "recall": tp / max(np.sum(y == l), 1),
                  "predicted_share": float(np.mean(p == l)), "true_share": float(np.mean(y == l))}
    return out


def bootstrap_ci(y, p, fn=macro_f1, n=1000, seed=0):
    rng = np.random.default_rng(seed)
    y, p = np.asarray(y), np.asarray(p)
    idx = rng.integers(0, len(y), (n, len(y)))
    vals = np.array([fn(y[i], p[i]) for i in idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_diff(y, p1, p2, fn=macro_f1, n=1000, seed=0):
    """Bootstrap the difference fn(p1) - fn(p2) on the same sentences. Returns (diff, lo, hi, p_two_sided)."""
    rng = np.random.default_rng(seed)
    y, p1, p2 = np.asarray(y), np.asarray(p1), np.asarray(p2)
    idx = rng.integers(0, len(y), (n, len(y)))
    d = np.array([fn(y[i], p1[i]) - fn(y[i], p2[i]) for i in idx])
    obs = fn(y, p1) - fn(y, p2)
    p = 2 * min(np.mean(d <= 0), np.mean(d >= 0))
    return float(obs), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5)), float(min(p, 1.0))
