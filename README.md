# Comparing Basic Fairness Metrics

**TrustAI Internship — First Task** · Osman Eren Doğan · Product area: TrustAI-X

A simple binary classification model is trained on synthetic credit data and audited with three
standard fairness metrics — **demographic parity**, **equal opportunity** and **disparate
impact** — across three sensitive attributes. Each metric is explained in plain language, with a
verdict on whether the model looks fair according to it.

## Structure

```
src/
  data_prep.py         features and sensitive group definitions
  train_model.py       logistic regression baseline
  fairness_metrics.py  the three metrics, computed from scratch
results/               metric tables and figure
fairness_analysis.ipynb  the deliverable notebook
```

## Setup and running

```bash
pip install -r requirements.txt
python src/data_prep.py
python src/train_model.py
python src/fairness_metrics.py
```

Then open `fairness_analysis.ipynb`. It only reads the CSVs in `results/`, so its tables cannot
drift from the scripts. The dataset (~15 MB) is downloaded automatically on first run and is not
committed.

## Data and model

100,000 synthetic credit applicants; target `delinquency_FL` (1 = fell behind on payments).
Model: standard scaling + logistic regression, 80/20 stratified split, seed 42, held-out ROC AUC
**0.914**. `class_weight="balanced"` compensates for only 26 % of applicants defaulting.

**Framing.** In lending the favourable outcome is being **approved**, so every metric is computed
on the approval rate rather than the raw model output.

## The three metrics in plain language

| Metric | The question it asks | Formula | Fair when |
|---|---|---|---|
| **Demographic parity** | Does the model approve the same *share* of people in each group? Ignores whether they repay. | `P(approved｜protected) − P(approved｜reference)` | \|gap\| ≤ 0.10 |
| **Equal opportunity** | Among people who *genuinely repay*, is each group approved equally often? The merit-aware view. | same difference, restricted to those who repay | \|gap\| ≤ 0.10 |
| **Disparate impact** | The legal test — the ratio of the two approval rates. | `P(approved｜protected) ÷ P(approved｜reference)` | ratio ≥ 0.80 (four-fifths rule) |

## Results

True delinquency rates differ substantially between groups, which matters for interpretation:

| Attribute | Protected group | Reference group | Delinquency (prot. / ref.) |
|---|---|---|---|
| Education | below university (72.6 %) | university+ | 32.8 % / 8.7 % |
| Age | young < 30 (18.2 %) | 30+ | 39.9 % / 23.2 % |
| Home ownership | non-homeowner (32.7 %) | homeowner | 52.4 % / 13.5 % |

Approval rates and metric verdicts:

| Attribute | Approval (prot. / ref.) | Demographic parity | Equal opportunity | Disparate impact |
|---|---|---|---|---|
| **Education** | 0.605 / 0.814 | **−0.208** ❌ | **−0.028** ✅ | **0.744** ❌ |
| **Age** | 0.455 / 0.708 | **−0.253** ❌ | **−0.165** ❌ | **0.642** ❌ |
| **Home ownership** | 0.245 / 0.865 | **−0.620** ❌ | **−0.468** ❌ | **0.283** ❌ |

## Is the model fair?

**It depends on which metric you ask — and that is the point of the comparison.**

**Education.** The three metrics disagree. Demographic parity and disparate impact both fail:
below-university applicants are approved 60.5 % of the time versus 81.4 %, a ratio of 0.744,
under the 0.80 legal line. But **equal opportunity passes** (−0.028) — among applicants who
genuinely repay, both groups are approved at almost the same rate. The approval gap therefore
reflects the real difference in default rates (32.8 % vs 8.7 %) rather than the model
mistreating creditworthy people. Whether that gap is acceptable is a policy question.

**Age.** All three fail. Base rates explain part of it, but equal opportunity fails at −0.165:
young applicants who **do** repay are approved about 16 points less often than older ones who
repay. That is harder to justify.

**Home ownership.** All three fail by the widest margins — a 62-point approval gap, a disparate
impact ratio of **0.283** (barely a third of the legal threshold), and a 47-point equal
opportunity gap. Creditworthy non-homeowners are rejected far more often than creditworthy
homeowners.

**Overall:** the model would not pass a fairness review. It fails the four-fifths rule on all
three attributes.

## Why the metrics disagree

Demographic parity looks only at outcomes; equal opportunity conditions on who actually repays.
When two groups genuinely default at different rates, these definitions **cannot both hold** — a
known impossibility result (Kleinberg et al.; Chouldechova). Education demonstrates it directly:
conditioning on repayment removes almost the entire gap. A single fairness number is never
enough; you have to state which definition you are using.

## Limitations

1. The data is **synthetic** — results describe this model and dataset, not real people.
2. All metrics use one decision threshold (0.5); a different threshold moves them.
3. Group definitions (age 30 cut-off, education split) are analytic choices.
4. No mitigation is attempted — the task is to measure and compare.

## Data source

Synthetic dataset from
[atalaydenknalbant/underbanked_risk_estimation](https://github.com/atalaydenknalbant/underbanked_risk_estimation),
accompanying *Credit Risk Estimation with Non-Financial Features* ([arXiv:2512.12783](https://arxiv.org/abs/2512.12783)).
