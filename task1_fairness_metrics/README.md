# Task 1 — Compare Basic Fairness Metrics

Train a simple binary classifier and audit it with **demographic parity**, **equal opportunity**
and **disparate impact**, explaining each in plain language.

## Run

```bash
python ../src/data_prep.py
python ../src/train_model.py
python ../src/fairness_metrics.py
```

Then open `analysis.ipynb` — it reads the CSVs in `results/`, so its tables cannot drift from
the scripts.

## Model

Standard scaling + logistic regression, 11 features, 80/20 stratified split, seed 42.
Held-out ROC AUC **0.914**. Favourable outcome = **approval**.

## Results

| Attribute | Approval prot./ref. | Demographic parity | Equal opportunity | Disparate impact |
|---|---|---|---|---|
| Education | 0.605 / 0.814 | −0.208 ❌ | **−0.028 ✅** | 0.744 ❌ |
| Age | 0.455 / 0.708 | −0.253 ❌ | −0.165 ❌ | 0.642 ❌ |
| Home ownership | 0.245 / 0.865 | −0.620 ❌ | −0.468 ❌ | 0.283 ❌ |

**The model fails the four-fifths rule on all three attributes.** The interesting case is
education, where the three metrics disagree: demographic parity and disparate impact fail, but
equal opportunity passes — among applicants who genuinely repay, both groups are approved at
almost the same rate. The approval gap there reflects the real difference in default rates
(32.8 % vs 8.7 %) rather than the model mistreating creditworthy people.

Full explanation, formulas and limitations: [`analysis.ipynb`](analysis.ipynb) and the
[repository README](../README.md).
