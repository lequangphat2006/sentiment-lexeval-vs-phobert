# LexEval vs PhoBERT — Vietnamese Product-Review Sentiment Analysis

A lexicon-enhanced stacking classifier (**LexEval**, internally **V40**) compared with a fine-tuned
**PhoBERT-base-v2** for three-class sentiment analysis of Vietnamese product reviews, under a
leakage-safe, ten-seed protocol. The repository also contains the error analysis, the semi-supervised
lexicon-expansion study, the log-odds lexicon audit, and the behaviour under distribution shift
(label shift and source shift).

Final term paper for **Natural Language Processing**, Faculty of Information Technology,
Quy Nhơn University.
Student: **Lê Quang Phát** (4754100004, class TTNT 47). Supervisor: **Dr. Lê Quang Hùng**.
Consolidates the weekly reports of weeks 1–6 — November 2026.

## Overview

Product reviews are a rich source of opinion, and labelling them negative / neutral / positive is a
routine task for e-commerce platforms. For Vietnamese, two families of approaches coexist:
pretrained transformers such as PhoBERT reach high accuracy but are heavy, need a GPU and give
poorly calibrated confidence; classical pipelines built from TF-IDF, linear models and gradient
boosting are light and fast, and are often supplemented with a hand-weighted sentiment lexicon.

This repository follows one such classical system, **LexEval**, from construction to critical
evaluation, and asks four practical questions:

- **RQ1 — Error analysis.** On which linguistic phenomena does the system fail, and why?
- **RQ2 — Lexicon expansion.** Does enlarging the lexicon from labelled data repair those failures,
  and under what conditions?
- **RQ3 — Comparison.** Under a leakage-safe protocol and ten matched seeds, how does LexEval
  compare with a fine-tuned PhoBERT in accuracy, calibration and cost?
- **RQ4 — Lexicon audit.** Do the hand-assigned polarities and intensities agree with corpus
  statistics, and did development refine them?

## Key results

### In-domain comparison (10 matched seeds 42–51, test n = 7,982 per seed)

| Metric | LexEval (V40) | PhoBERT-base-v2 |
|---|---|---|
| Macro-F1 | **0.9586 ± 0.0018** | 0.9482 ± 0.0028 |
| Accuracy | 0.9585 ± 0.0019 | 0.9480 ± 0.0028 |
| ECE (10 bins) | **0.0071 ± 0.0015** | 0.0271 ± 0.0044 |
| Brier score | **0.0647 ± 0.0024** | 0.0850 ± 0.0047 |
| Model size | **159.4 MB** | 517.0 MB |

Paired difference in macro-F1 **+0.0105**; LexEval wins **10/10 seeds** (Wilcoxon **p = 0.0020**).
Probabilities are about **4× better calibrated** and the model is **~3× smaller**.

### Gold set (label shift, same source)

| Metric | LexEval | PhoBERT |
|---|---|---|
| Macro-F1 | **0.8636 ± 0.0019** | 0.8482 ± 0.0035 |
| ECE | **0.0468 ± 0.0024** | 0.0837 ± 0.0030 |

### Error analysis (Study 1, baseline TF-IDF + linear SVM, seed 42)

| Group | Reviews (share) | Macro-F1 | Recall neg / neu / pos |
|---|---|---|---|
| All data | 39,890 (100%) | 0.9644 | 0.956 / 0.969 / 0.968 |
| Short (≤ 6 words) | 2,025 (5.1%) | 0.9292 | 0.933 / 0.886 / 0.968 |
| Negation | 17,581 (44.1%) | 0.9392 | 0.962 / 0.967 / **0.873** |
| **Hedging** | 4,161 (10.4%) | **0.7558** | **0.602 / 0.997 / 0.444** |
| Contrast | 11,836 (29.7%) | 0.9155 | 0.929 / 0.986 / 0.764 |
| No phenomenon | 14,610 (36.6%) | 0.9571 | 0.958 / 0.921 / 0.989 |

- **Hedging is the largest weakness** (macro-F1 0.7558, ≈0.21 below the overall level); 95.8% of
  hedging reviews are neutral, so the model learns “hedge word → neutral” (neutral recall 0.997,
  but only 60.2% of negatives and 44.4% of positives recognised).
- **Negation mainly hurts the positive class**: positive recall drops to 0.873 (0.968 overall).
- Short text (−3.5 pts) and contrast (−4.9 pts) follow.

### Lexicon expansion (Study 1, 15 fits; lexicon-only logistic regression on 8 lexicon features)

| Configuration | Entries | Test coverage | Neg. coverage | Lexicon-only macro-F1 | Δ vs. original (wins/15) |
|---|---|---|---|---|---|
| Original lexicon | 70 | 72.4% | 52.5% | 0.7254 | — |
| A. Raw (\|z\| ≥ 3) | 1,064 | **99.8%** | 99.7% | 0.7155 | **−0.0098 (0/15)** |
| B. A + drop fragments | 984 | 99.8% | 99.7% | 0.7168 | −0.0086 (0/15) |
| C. B + \|δ\| ≥ 1.0 | 772 | 98.1% | 96.6% | 0.7486 | +0.0233 (15/15) |
| **D. B + \|δ\| ≥ 1.5** | **546** | **93.7%** | **90.1%** | **0.7683** | **+0.0430 (15/15)** |
| E. B + \|δ\| ≥ 2.0 | 329 | 86.6% | 79.6% | 0.7652 | +0.0399 (15/15) |

- **Coverage is not quality.** The raw expansion lifts coverage to 99.8% but lowers macro-F1 by
  0.0098 (0/15 wins): it adds very frequent words that are barely polarised (“hàng”, “giao”,
  “sản phẩm”, “được”). With a large corpus, statistical significance is almost guaranteed — **effect
  size is what must be controlled**.
- **Effect-size filtering is the decisive step.** \|δ\| ≥ 1.5 raises lexicon-only macro-F1 from
  0.7254 to **0.7683** (+0.0430, 15/15) while keeping 93.7% coverage.
- **No visible gain when added to TF-IDF**: Δ = −0.0008…−0.0015 in every configuration. TF-IDF
  bigrams already learn the polarised phrases from the same labels; the lexicon's value must be
  assessed elsewhere (interpretability, robustness under source/label change).

### Lexicon audit (Study 4)

A log-odds audit of the 70-entry core lexicon (34 positive, 36 negative) finds that it chooses the
right polarity for **97.7%** of its words, but that its **intensity weights are systematically off
and were never revised in eleven development rounds**.

## Repository structure

```
LEXEVAL/            train_v40.py, lexicon_v40_4.py (70-entry core + semi-supervised expansion),
                    seeds_config.py, split_fingerprint.py, input_data/train_3class.csv,
                    logs/, reports/ (per-seed per_class.json + report.png, per-sample probabilities)
PHOBERT/            train_phobert.py, measure_phobert_length.py, seeds_config.py,
                    split_fingerprint.py, input_data/train_3class.csv, logs/,
                    reports/ (per-seed confusion matrix, per-sample logits/probs)
results/            results_*_multiseed.json, split_fingerprints.json (both systems merged),
                    comparison_table.md/.json
GOLD_SET/           probs_gold/ (per-sample probabilities on the gold set), eval_gold.py,
                    metrics_gold.json
compare_systems.py  merge results, paired tests (Wilcoxon, t, Holm–Bonferroni)
```

> Model checkpoints (`models/`, ~517 MB for PhoBERT) are **not** included in this repository —
> they are regenerated by `train_phobert.py`.

## Setup and running

```bash
pip install -r requirements.txt

# 1) LexEval — 10 seeds, one session (~5–6 h on 4 vCPU, no GPU)
cd LEXEVAL && python train_v40.py

# 2) PhoBERT — needs a GPU (original log: 2 × Tesla T4). Seeds can be split across sessions
cd PHOBERT && python train_phobert.py                 # all missing seeds
cd PHOBERT && python train_phobert.py --seeds 42,43   # selected seeds only
cd PHOBERT && python train_phobert.py --quick_test    # smoke test, 1 seed, 1 epoch

# 3) Merge results + paired tests (reads results/*.json, no retraining)
python compare_systems.py --dir results

# 4) Recompute gold-set metrics from stored probabilities
python GOLD_SET/eval_gold.py
```

**Splitting protocol.** `StratifiedGroupKFold` on `near_dup_group_id` (≈ 60/20/20), same function and
same seed for both systems; `split_fingerprint.py` automatically checks that both systems receive the
same test set (**10/10 seeds match**).

## Data

- `input_data/train_3class.csv`: **39,890 Vietnamese reviews** (columns `comment`, `sentiment`,
  `near_dup_group_id`) collected from public review endpoints of the **Tiki** marketplace. Only text
  and labels were kept — **no personal identifying information is stored**. Labels are manual: four
  annotators labelled independently following a guideline that places factual, balanced or
  non-committal reviews in the neutral class (**Fleiss' κ = 0.831**). A language model was applied
  only afterwards to flag possible duplicates and inconsistent labels for the annotators to re-check;
  **no label was produced by the model**.
  Corpus class mix: 13,000 negative / 13,890 neutral / 13,000 positive; 19.5 words on average
  (median 15); near-duplicate analysis finds 39,769 groups, 99.8% singletons.
  The same file is kept in both `LEXEVAL/` and `PHOBERT/` (identical md5: `737F53148BD40C4CA67B4922221A0EC9`)
  so each script runs independently.
- **Gold set**: only per-sample predicted probabilities are stored (`GOLD_SET/probs_gold/`); the raw
  text is **not** included.
- **AIVIVN 2019** (source shift, 15,922 usable two-class reviews): results are reported in the paper;
  the dataset is distributed by the organisers and is **not** included in this repository.

## Reproducibility notes (not yet reconciled)

- **PhoBERT length.** Files in this repository are from the `max_length = 192` run (see
  `PHOBERT/logs/train_phobert_20260831_045154.log`). The related manuscript reports
  `max_length = 128` with slightly different numbers (in-domain: macro-F1 0.9483 ± 0.0017,
  ECE 0.0286; gold set: macro-F1 0.8419, ECE 0.0849). The logs of that run are not in the
  repository.
- **Gold set size.** Per-seed probability files contain **3,563** rows; the manuscript states
  **3,653**.
- Logs were edited only for label wording (one line describing an analysis step); **no numeric
  value was changed**.

## Licence

No licence has been chosen yet; the code and data are shared for academic learning and course
assessment. If you plan to reuse them, please contact the author.