"""Sections 2-5: tests on the model's recorded outputs (no model calls needed)."""
import numpy as np
import pandas as pd

from . import Check
from .. import charts
from ..data import LABELS
from ..metrics import macro_f1, per_class, bootstrap_ci, paired_diff

BENCH = {"majority": "Majority class", "lexicon": "Keyword lexicon", "tfidf": "TF-IDF + logistic regression"}


def contract(ev, out_dir):
    r = ev.rec
    th = ev.cfg["thresholds"]
    checks = []
    unk = float((r.pred == "unknown").mean())
    checks.append(Check(
        "O1", "contract", "Every answer can be turned into a label", "pass" if unk <= th["max_unparseable"] else "fail",
        f"{(1 - unk):.1%} of {len(r)} answers contain a label word; {int((r.pred == 'unknown').sum())} do not.",
        {"answers": len(r), "unparseable_share": unk}, f"unparseable <= {th['max_unparseable']:.0%}",
        None if unk <= th["max_unparseable"] else "high"))

    strict_ok = float((r.strict != "unknown").mean())
    multi = float((r.n_labels > 1).mean())
    neg = float(r.negated.mean())
    words = r.raw.str.split().str.len()
    disagree = float(((r.strict != "unknown") & (r.strict != r.pred)).mean())
    ok = strict_ok >= th["min_instruction_compliance"] and multi <= th["max_multi_label"]
    checks.append(Check(
        "O2", "contract", "Answers follow the instruction (one word)", "pass" if ok else "warn",
        f"The prompt asked for one word. {strict_ok:.1%} of answers start with a label, but the median answer is "
        f"{int(words.median())} words long (an unrequested explanation); {multi:.1%} name more than one label and "
        f"{neg:.1%} contain a negated label ('not hawkish'). " +
        ("Because the label always comes first, the usual 'first label word' parser and a strict 'first word' "
         "parser agree here, but a lenient parser would misread answers if the model ever put the explanation "
         "first. The extra text also multiplies cost and latency." if disagree == 0 else
         f"The usual 'first label word' parser and a strict 'first word' parser disagree on {disagree:.1%} of "
         "answers, so the labels depend on an arbitrary parsing choice."),
        {"starts_with_label": strict_ok, "median_words": float(words.median()), "multi_label_share": multi,
         "negated_share": neg, "parser_disagreement": disagree},
        f"starts with label >= {th['min_instruction_compliance']:.0%}; multi-label <= {th['max_multi_label']:.0%}",
        None if ok else "medium",
        "" if ok else "Constrain the output (one token, a fixed label set or structured output) and log raw answers.",
        table=r[r.n_labels > 1][["split", "label", "pred", "raw"]].head(15).reset_index(drop=True)))
    return checks


def performance(ev, out_dir):
    th = ev.cfg["thresholds"]
    r = ev.rec
    checks = []
    rows = []
    for s, g in r.groupby("split"):
        row = {"split": s, "n": len(g), "model_f1": macro_f1(g.label, g.pred),
               "model_f1_ci": "%.3f-%.3f" % bootstrap_ci(g.label, g.pred, seed=int(s) % 997)}
        for k in BENCH:
            row[f"{k}_f1"] = macro_f1(g.label, g[k])
        d, lo, hi, p = paired_diff(g.label, g.pred, g.tfidf, seed=int(s) % 991)
        row.update(diff_vs_tfidf=d, diff_lo=lo, diff_hi=hi, p_vs_tfidf=p, model_acc=float((g.label == g.pred).mean()))
        rows.append(row)
    tab = pd.DataFrame(rows)
    better = int(((tab.diff_lo > 0)).sum())
    worse = int(((tab.diff_hi < 0)).sum())
    mean_f1, mean_tf = tab.model_f1.mean(), tab.tfidf_f1.mean()
    status = "pass" if better == len(tab) else "fail"
    sev = None if status == "pass" else "high"
    checks.append(Check(
        "A1", "performance", "Accuracy against simple benchmarks (macro-F1, 3 test splits)", status,
        f"Mean macro-F1 is {mean_f1:.3f} for the model against {mean_tf:.3f} for a TF-IDF logistic regression "
        f"trained in seconds on the same training data, {tab.lexicon_f1.mean():.3f} for a keyword lexicon and "
        f"{tab.majority_f1.mean():.3f} for always answering the majority class. The model is significantly better "
        f"than TF-IDF on {better} of {len(tab)} splits and significantly worse on {worse} (paired bootstrap, 95%)."
        + ("" if status == "pass" else " The added cost, opacity and vendor dependence of the LLM buy no accuracy."),
        {"model_f1_mean": mean_f1, "tfidf_f1_mean": mean_tf, "lexicon_f1_mean": tab.lexicon_f1.mean(),
         "majority_f1_mean": tab.majority_f1.mean(), "splits_better": better, "splits_worse": worse},
        "significantly better than TF-IDF on every split", sev,
        "" if status == "pass" else "Do not adopt the zero-shot LLM over the benchmark; fine-tune (or use a "
                                    "fine-tuned encoder) and re-assess against the same benchmarks.",
        table=tab))
    fig, ax = charts.plt.subplots(figsize=(6.2, 2.8))
    names = [("model_f1", ev.cfg["model"]["short"])] + [(f"{k}_f1", v) for k, v in BENCH.items()]
    vals = [tab[c].mean() for c, _ in names]
    ax.barh([n for _, n in names], vals, color=[charts.SERIES[0]] + [charts.GRID] * 3, height=0.5)
    for i, v in enumerate(vals):
        ax.annotate(f"{v:.3f}", (v, i), xytext=(4, 0), textcoords="offset points", va="center", fontsize=8,
                    color=charts.INK2)
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("macro-F1, mean of 3 test splits")
    ax.set_title("The LLM against simple benchmarks")
    checks[-1].chart = charts.save(fig, out_dir, "a1_benchmarks.png")

    rng = tab.model_f1.max() - tab.model_f1.min()
    ok = rng <= th["max_split_range"]
    checks.append(Check(
        "A2", "performance", "Stability of accuracy across test splits", "pass" if ok else "warn",
        f"Macro-F1 ranges from {tab.model_f1.min():.3f} to {tab.model_f1.max():.3f} across the three splits.",
        {"f1_min": tab.model_f1.min(), "f1_max": tab.model_f1.max(), "range": rng},
        f"range <= {th['max_split_range']}", None if ok else "low"))

    dup = r[r.duplicated("sentence", keep=False)]
    if len(dup):
        grp = dup.groupby("sentence").pred.agg(lambda x: x.nunique() == 1)
        agree = float(grp.mean())
        ok = agree >= th["min_repeat_agreement"]
        checks.append(Check(
            "A3", "performance", "Repeatability: the same sentence asked in separate calls", "pass" if ok else "warn",
            f"{len(grp)} sentences appear in more than one test split and were classified in separate calls; the "
            f"model gave the same label every time for {agree:.1%} of them." +
            ("" if ok else " Identical inputs get different answers, so results cannot be reproduced exactly."),
            {"sentences": len(grp), "agreement": agree}, f"agreement >= {th['min_repeat_agreement']:.0%}",
            None if ok else "medium",
            "" if ok else "Fix temperature at 0, pin the model version and cache answers."))
    return checks


def segments(ev, out_dir):
    th = ev.cfg["thresholds"]
    r = ev.rec
    checks = []
    pc = per_class(r.label, r.pred)
    tab = pd.DataFrame([{"class": k, **v} for k, v in pc.items()])
    gaps = (tab.predicted_share - tab.true_share)
    worst = tab.iloc[int(gaps.abs().idxmax())]
    ok = gaps.abs().max() <= th["max_class_share_gap"]
    hk = pc["hawkish"]
    checks.append(Check(
        "C1", "segments", "No systematic bias toward one class", "pass" if ok else "fail",
        f"The model answers '{worst['class']}' for {worst.predicted_share:.0%} of sentences, against a true share of "
        f"{worst.true_share:.0%}. It finds only {hk['recall']:.0%} of hawkish sentences (recall), against "
        f"{pc['dovish']['recall']:.0%} of dovish ones." +
        ("" if ok else " Errors are not random: they lean one way, so they will not average out in an index."),
        {"max_share_gap": float(gaps.abs().max()), "hawkish_recall": hk["recall"],
         "dovish_recall": pc["dovish"]["recall"], "neutral_predicted_share": pc["neutral"]["predicted_share"],
         "neutral_true_share": pc["neutral"]["true_share"]},
        f"|predicted - true share| <= {th['max_class_share_gap']:.0%}", None if ok else "high",
        "" if ok else "Recalibrate (class thresholds on label probabilities) or fine-tune; monitor the "
                      "predicted class mix in production.", table=tab))
    fig, ax = charts.plt.subplots(figsize=(6.2, 2.6))
    x = np.arange(len(LABELS))
    ax.bar(x - 0.18, tab.true_share * 100, width=0.34, color=charts.SERIES[1], label="True share")
    ax.bar(x + 0.18, tab.predicted_share * 100, width=0.34, color=charts.SERIES[0], label="Predicted share")
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS)
    ax.set_ylabel("% of sentences")
    ax.set_title("What the model answers vs what is true")
    ax.legend(loc="upper left")
    checks[-1].chart = charts.save(fig, out_dir, "c1_class_mix.png")

    flips = r[r.label.isin(["hawkish", "dovish"]) & r.pred.isin(["hawkish", "dovish"]) & (r.label != r.pred)]
    sided = r[r.label.isin(["hawkish", "dovish"])]
    rate = len(flips) / len(sided)
    tf_flips = r[r.label.isin(["hawkish", "dovish"]) & r.tfidf.isin(["hawkish", "dovish"]) & (r.label != r.tfidf)]
    tf_rate = len(tf_flips) / len(sided)
    ok = rate <= th["max_direction_error"]
    conf = pd.crosstab(r.label, r.pred).reindex(index=LABELS, columns=LABELS + ["unknown"], fill_value=0)
    checks.append(Check(
        "C2", "segments", "Wrong-direction errors (hawkish read as dovish, or the reverse)", "pass" if ok else "warn",
        f"{rate:.1%} of hawkish or dovish sentences are given the opposite stance (TF-IDF: {tf_rate:.1%}). "
        "Most errors are therefore misses (a stance read as neutral) rather than reversals.",
        {"direction_error_rate": rate, "tfidf_direction_error_rate": tf_rate}, f"<= {th['max_direction_error']:.0%}",
        None if ok else "medium", table=conf.reset_index().rename(columns={"label": "true \\ predicted"})))

    rows = []
    overall = macro_f1(r.label, r.pred)
    r = r.assign(length=pd.qcut(r.words, 4, labels=["shortest 25%", "short", "long", "longest 25%"]))
    for col, name in [("source", "Document type"), ("era", "Period"), ("contrast", "Has a contrast word"),
                      ("length", "Sentence length")]:
        for v, g in r.groupby(col, observed=True):
            if len(g) < 30 or v == "unknown":
                continue
            rows.append({"segment": name, "value": str(v), "n": len(g), "model_f1": macro_f1(g.label, g.pred),
                         "tfidf_f1": macro_f1(g.label, g.tfidf)})
    seg = pd.DataFrame(rows)
    weak = seg[(seg.model_f1 < overall - th["max_segment_drop"]) & (seg.n >= 50)]
    checks.append(Check(
        "C3", "segments", "Errors by document type, period, sentence structure and length",
        "pass" if weak.empty else "warn",
        f"Overall macro-F1 is {overall:.3f}. " + ("No segment with 50 or more sentences is more than "
                                                  f"{th['max_segment_drop']:.2f} below it." if weak.empty else
                                                  "Weak segments: " + "; ".join(
                                                      f"{w.segment.lower()} = {w.value}: {w.model_f1:.3f} (n = {w.n})"
                                                      for _, w in weak.iterrows()) + "."),
        {"overall_f1": overall, "weak_segments": len(weak)}, f"no segment below overall - {th['max_segment_drop']}",
        None if weak.empty else "medium", table=seg))
    return checks


def use(ev, out_dir):
    """The intended use is a tone index: (hawkish - dovish) / sentences, per year."""
    th = ev.cfg["thresholds"]
    r = ev.rec.drop_duplicates("sentence").copy()
    r["year"] = pd.to_numeric(r.year, errors="coerce")
    sc = {"hawkish": 1, "dovish": -1, "neutral": 0, "unknown": 0}
    g = r.groupby("year").agg(n=("label", "size"), true=("label", lambda x: x.map(sc).mean()),
                              model=("pred", lambda x: x.map(sc).mean()),
                              tfidf=("tfidf", lambda x: x.map(sc).mean())).reset_index()
    g = g[g.n >= th["min_sentences_per_year"]]
    slope = float(np.polyfit(g.true, g.model, 1)[0])
    corr = float(np.corrcoef(g.true, g.model)[0, 1])
    tf_corr = float(np.corrcoef(g.true, g.tfidf)[0, 1])
    bias = float((g.model - g.true).mean())
    sign_wrong = int((np.sign(g.model.round(3)) != np.sign(g.true.round(3))).sum())
    ok = slope >= th["min_index_slope"] and abs(bias) <= th["max_index_bias"]
    fig, ax = charts.plt.subplots(figsize=(6.2, 3.0))
    ax.plot(g.year, g.true, color=charts.SERIES[1], marker="o", ms=4, label="True labels")
    ax.plot(g.year, g.model, color=charts.SERIES[0], marker="o", ms=4, label=ev.cfg["model"]["short"])
    ax.axhline(0, color=charts.INK2, lw=0.8)
    ax.set_ylabel("tone = (hawkish - dovish) / n")
    ax.set_title("Tone index by year: model vs true labels")
    ax.legend(loc="lower left")
    chart = charts.save(fig, out_dir, "u1_tone_index.png")
    checks = [Check(
        "U1", "use", "The tone index built from the model tracks the true index", "pass" if ok else "fail",
        f"Across {len(g)} years with at least {th['min_sentences_per_year']} labelled sentences, the model's yearly "
        f"index correlates {corr:.2f} with the index from the true labels (TF-IDF: {tf_corr:.2f}). The slope of "
        f"model on truth is {slope:.2f}, the average offset is {bias:+.3f} (negative = too dovish) and the sign is "
        f"wrong in {sign_wrong} years. " +
        ("The model's index carries little of the true year-to-year variation. " if corr < 0.3 else "") +
        "With 10-85 sentences per year the true index is itself noisy, but the benchmark tracks it far better."
        if not ok else
        f"Across {len(g)} years the model's index correlates {corr:.2f} with the true one, with slope {slope:.2f} "
        f"and average offset {bias:+.3f}.",
        {"years": len(g), "corr": corr, "tfidf_corr": tf_corr, "slope": slope, "mean_bias": bias,
         "years_wrong_sign": sign_wrong},
        f"slope >= {th['min_index_slope']}, |bias| <= {th['max_index_bias']}", None if ok else "high",
        "" if ok else "Do not feed the index into decisions without rescaling; validate rescaling out of sample, "
                      "or replace the classifier.", table=g, chart=chart)]
    return checks
