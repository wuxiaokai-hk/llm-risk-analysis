# llm-risk-analysis: a model risk assessment of an LLM used for a finance task

A research team wants to use an LLM to label every sentence of FOMC statements, minutes, speeches and press
conferences as **hawkish, dovish or neutral**, and to turn those labels into a policy-tone index for its
rate-path views. This repo assesses that model the way a model risk function would. It checks the output
contract, compares accuracy with cheap benchmarks, looks for systematic bias, tests fitness for the *actual
use* (the index), and runs behavioural robustness tests. It ends with a rated report that a human signs off.

The model under review is **ChatGPT, zero-shot**. Its actual answers on three test splits (1,428 answers) were
released with Shah, Paturi & Chava (2023), *Trillion Dollar Words* (ACL), together with the human labels, so
every number here comes from a real LLM's real output.

**[Open the sample report](sample_report/assessment_report.html)** (download it and open it in a browser; it is
one self-contained HTML file). It is a draft: no reviewer has signed it.

## What the assessment found

| | Finding | Severity |
|---|---|---|
| **A1** | Macro-F1 is **0.552**, no better than a TF-IDF logistic regression trained in seconds on the same data (**0.570**). It is not significantly better on any of the 3 splits (paired bootstrap). The LLM adds cost, opacity and vendor dependence without adding accuracy. | High |
| **C1** | It answers *neutral* for **70%** of sentences (true share 48%) and finds only **28%** of hawkish sentences. The errors lean one way, so they do not average out. | High |
| **U1** | The yearly tone index built from its labels correlates **−0.10** with the index from the true labels (TF-IDF: 0.50). On average it sits 0.16 too dovish, and its sign is wrong in 14 of 27 years. For the intended use, it is not fit. | High |
| **O2** | Asked for one word, it writes a 37-word explanation (median), and 5% of answers name more than one label. Parsing works today only because the label happens to come first. | Medium |

What holds up: every answer can be parsed (O1). Accuracy is stable across splits (A2). The same sentence sent in
separate calls gets the same label 99% of the time (A3). Wrong-direction errors are rare, at 8% against 17% for
TF-IDF (C2). The model mostly *misses* a stance rather than reversing it. There is no weak segment by document
type, period, sentence structure or length (C3).

Proposed rating (rule-based): **Unsatisfactory**. Only a reviewer sets the final rating.

## How it works

| Section | Tests | Needs a live model? |
|---|---|---|
| 2 Output contract | O1 every answer parses · O2 follows "one word", multiple or negated labels, parser dependence | no |
| 3 Accuracy | A1 macro-F1 with bootstrap intervals vs majority, keyword lexicon and TF-IDF, with paired tests · A2 stability across splits · A3 repeatability across separate calls | no |
| 4 Errors | C1 class bias and recall · C2 wrong-direction errors · C3 errors by document type, period, contrast words and length | no |
| 5 Fitness for use | U1 the tone index: correlation, slope, offset and sign against the true index, by year | no |
| 6 Behaviour | B1 must-pass sentences · B2 swapping *raise* and *lower* must flip the label · B3 invariance to prefix, name, typo and number edits · B4 three prompt wordings · B5 determinism | **yes** |

Sections 2–5 run on recorded answers. Section 6 calls a model live: Ollama (local, no API key), any
OpenAI-compatible endpoint, or an MLX model with a LoRA adapter. The sample report runs section 6 on the TF-IDF
benchmark to show the tests working (it fails 31% of the must-pass sentences), and marks those results as
information only, because that is not the model under review. To test a real LLM, set `live:` in
[`assess.yaml`](assess.yaml):

```yaml
live:
  backend: ollama
  model: qwen2.5:1.5b
  same_model_as_reviewed: true   # only when the live model IS the model under review
```

To assess **your own fine-tuned model** (for example
[fed-tone-finetune](https://github.com/wuxiaokai-hk/fed-tone-finetune)), point the recorded answers at its
predictions file and the live tests at its adapter:

```yaml
model:
  name: Qwen2.5-1.5B + LoRA (fed-tone-finetune)
  short: Fine-tuned Qwen
  recorded: {kind: jsonl, path: ../fed-tone-finetune/results/finetuned_predictions.jsonl}
live: {backend: mlx, model: Qwen/Qwen2.5-1.5B-Instruct, adapter: ../fed-tone-finetune/adapters,
       same_model_as_reviewed: true}
```

The report is drafted and signed off the same way as in
[macroforecast-validation](https://github.com/wuxiaokai-hk/macroforecast-validation). A local LLM drafts each
section's text. A grounding guard rejects any draft that invents a number, leaves out a warning or uses
approval language. A named reviewer approves, rates and finalizes, and every step is written to an audit log.

## Run it

```bash
pip install -r requirements.txt
python tools/get_data.py                 # clones the authors' data repo into data/
python -m llmrisk run --out report       # 15 tests -> report/assessment_report.html
python -m pytest -q                      # 18 tests
```

The review loop:

```bash
python -m llmrisk status   --out report
python -m llmrisk approve  --out report --section all --reviewer "Your Name"
python -m llmrisk rate     --out report --rating Unsatisfactory --reviewer "Your Name" \
                           --rationale "No accuracy gain over the benchmark; index not fit for use"
python -m llmrisk finalize --out report --reviewer "Your Name"
```

## Layout

```
llmrisk/
  data.py        splits, recorded answers (authors' CSVs or any JSONL), document type, period, sentence features
  parse.py       lenient and strict label parsing, multi-label and negated-label detection
  lexicon.py     transparent keyword benchmark
  backends.py    live models: TF-IDF baseline, Ollama, OpenAI-compatible, MLX (+ LoRA adapter)
  metrics.py     macro-F1, bootstrap intervals, paired bootstrap differences
  checks/        static.py (sections 2-5), behaviour.py (section 6)
  agent.py       drafting agent and grounding guard
  signoff.py     approvals, rating, finalize, audit log
  report.py      self-contained HTML report
tools/get_data.py
sample_report/
```

## Data and licence

The data and ChatGPT answers come from Shah, Paturi & Chava (2023), "Trillion Dollar Words: A New Financial
Dataset, Task & Market Analysis", ACL ([repo](https://github.com/gtfintechlab/fomc-hawkish-dovish)), under
CC BY-NC 4.0. They are downloaded at run time and are not stored here. The sample report quotes a few sentences
and answers for illustration, with this attribution.

## License

MIT, see [LICENSE](LICENSE). Free to use, change and share. Data downloaded from public sources stay under their providers' terms. Independent open-source project, not affiliated with or endorsed by any employer, regulator or data vendor.
