# Methodology

## Scope

The model is the whole pipeline: the prompt, the vendor model, the parser that turns free text into a label,
and the aggregation into a tone index. The use is the index, not single labels. So the assessment judges
labels against the human annotations, and also judges the index against the index built from those
annotations.

## Data

There are three random train/test splits (seeds 5768, 78516 and 944601) of the authors' combined dataset of
meeting minutes, speeches and press conferences, with 476 test sentences each. The ChatGPT answers are aligned
row by row with each test split, and the code checks that alignment. Benchmarks are trained on the matching
training split. The document type comes from matching each sentence to the authors' per-source annotation files.

## Tests

**O2 Output contract.** The strict parser accepts a label only as the first word. The lenient parser takes the
first label word found anywhere. If the two disagree, or an answer names several labels or a negated label,
then the label depends on an arbitrary parsing rule.

**A1 Benchmarks.** These are the majority class, a 34-word keyword lexicon, and TF-IDF (word 1–2-grams,
sublinear term frequency) with class-balanced logistic regression. Macro-F1 has 1,000-draw bootstrap 95%
intervals. The model is compared with TF-IDF using a paired bootstrap on the same sentences. The bar to pass
is significantly better than TF-IDF on every split: an LLM that only matches a cheap, transparent benchmark does
not justify its cost and opacity.

**A3 Repeatability.** The splits overlap, so 257 sentences were sent to the model in more than one separate call.
A3 is the agreement across those calls.

**C1–C3 Errors.** C1 compares the predicted class mix with the true mix, plus recall by class. C2 is the share of
hawkish or dovish sentences given the opposite stance. C3 gives macro-F1 by segment (only segments with 30 or
more sentences are shown; only those with 50 or more can be flagged).

**U1 Fitness for use.** The tone for a year is (hawkish − dovish) / sentences. It is computed from the model's
labels and from the true labels, over unique test sentences, in years with at least 10 sentences. U1 reports
the correlation, the slope of the model index on the true index (1 = no attenuation), the mean offset, and the
number of years with the wrong sign. The yearly samples are small (10–85 sentences), so the true index is noisy.
The TF-IDF index is shown on the same basis for comparison.

**B1–B5 Behaviour** (CheckList, Ribeiro et al. 2020).
- B1 uses 52 template sentences (4 subjects × 11 templates, plus rate-move sizes) whose stance is unambiguous.
- B2 swaps only the policy word ('raise' and 'lower', 'firming' and 'easing', 'tighten' and 'ease') in 32
  pairs. The label must flip.
- B3 applies 4 meaning-preserving edits to 150 test sentences: a neutral prefix, "the Committee" →
  "policymakers", one swapped-letter typo, and a small number change. The label must not change.
- B4 asks the same 150 sentences with 3 instruction wordings, including a different label order.
- B5 repeats the same calls.

If the live model is not the model under review, section 6 is information only and does not affect the rating.

## Rating rule (a proposal only)

| Proposed rating | When |
|---|---|
| Unsatisfactory | a high-severity failure of the output contract or of fitness for use, or 3 or more high findings |
| Needs improvement | at least one high finding, or 3 or more medium findings |
| Satisfactory | otherwise |

The reviewer sets the final rating with a rationale. A rating that differs from the proposal is logged as an
override.
