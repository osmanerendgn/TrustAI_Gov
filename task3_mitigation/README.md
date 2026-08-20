# Task 3 — Fairness Mitigation and Before/After Comparison

Apply simple fairness mitigations to the Task 1 model and compare fairness **and** performance
before and after.

## Run

```bash
python ../src/mitigation.py
```

Then open [`analysis.ipynb`](analysis.ipynb) — it reads the CSVs in `results/`, so its tables
cannot drift from the experiment.

## Target: age

Age is a **legally protected class** in credit lending (e.g. under the US Equal Credit
Opportunity Act), and the Task 1 baseline fails all three metrics on it. Education and home
ownership are still measured, so spillover stays visible.

## The three methods

| | Method | Idea |
|---|---|---|
| **A** | Remove sensitive + proxy features | Drop `age` and its three strongest proxies (`subscription_count`, `online_shopping_frequency`, `owns_car`) — "fairness through unawareness" |
| **B** | Group-specific threshold | One model, but a young applicant must look riskier before being rejected. Protected threshold **0.63**, reference stays 0.50 |
| **C** | Sample reweighting | Reweight training rows so group and outcome look independent (Kamiran & Calders), weight range 0.657–1.228 |

**Method notes.** Every variant uses the *identical* train/test rows, so the comparison isolates
the mitigation rather than the split. The threshold in B is tuned on **out-of-fold predictions
from the training set** — never the test set — and targets disparate impact ≥ **0.82** rather
than 0.80, leaving a margin for sampling noise.

## Fairness — before and after (age)

| Variant | Approval young / 30+ | Demographic parity | Equal opportunity | Disparate impact |
|---|---|---|---|---|
| baseline | 0.455 / 0.708 | −0.253 ❌ | −0.165 ❌ | 0.642 ❌ |
| A: remove age+proxies | 0.502 / 0.707 | −0.206 ❌ | **−0.096 ✅** | 0.709 ❌ |
| **B: group threshold** | 0.582 / 0.708 | −0.126 ❌ | **−0.031 ✅** | **0.822 ✅** |
| C: reweighting | 0.550 / 0.698 | −0.147 ❌ | **−0.035 ✅** | 0.789 ❌ |

**Only method B clears the 0.80 legal threshold.** C misses it by one point. Equal opportunity
improves under all three — young applicants who *genuinely repay* are now approved at nearly the
same rate as older ones. Demographic parity never passes, which is expected: young applicants
really do default more often (39.9 % vs 23.2 %), and that metric does not allow for it.

## Performance cost

| Variant | ROC AUC | Accuracy | Recall (delinq.) | Defaulters approved (FN) | Good payers rejected (FP) |
|---|---|---|---|---|---|
| baseline | 0.9136 | 0.859 | 0.875 | 654 | 2,173 |
| A: remove age+proxies | **0.9146** | 0.865 | 0.872 | 670 | 2,027 |
| B: group threshold | 0.9136 | 0.865 | 0.843 | **824** | 1,880 |
| C: reweighting | 0.9123 | 0.860 | 0.860 | 733 | 2,071 |

* **A is nearly free** — ROC AUC actually *rises* slightly and only 16 more defaulters are
  approved. Dropping the age proxies removed noise as well as signal.
* **B costs the most**: recall 0.875 → 0.843 and **170 more approved defaulters** per 20,000
  applications. ROC AUC is unchanged because the model is identical — only the decision rule moved.
* **C sits in between**: 79 extra approved defaulters.

**Accuracy rises for every variant — do not read this as a fairness bonus.** It is an artifact of
`class_weight="balanced"` over-flagging risk at threshold 0.5 (2,173 good payers rejected vs 654
defaulters approved). The decision-relevant costs are recall and the approved-defaulter count.

## Spillover — does fixing age fix anything else?

Disparate impact ratio per attribute:

| Attribute | baseline | A | B | C |
|---|---|---|---|---|
| age | 0.642 | 0.709 | **0.822** | 0.789 |
| education | 0.744 | 0.744 | 0.763 | 0.759 |
| home ownership | 0.283 | 0.285 | 0.294 | 0.259 |

**No.** Education and home ownership barely move under any variant, and **home ownership stays
near a third of the legal threshold throughout**. A model cannot be declared fixed because one
protected attribute was remediated — each attribute needs its own assessment.

## Recommendation

**Adopt A permanently** — it improves fairness *and* accuracy slightly, so there is no reason not
to. **Layer B on top only if the business accepts 170 additional approved defaulters per 20,000
applications**, because it is the only route to a compliant age axis in this experiment.

### Caveat: group-specific thresholds are disparate treatment

Method B uses the protected attribute **directly in the decision rule** — two applicants with the
same risk score get different decisions based on age. Even though the outcome is fairer, this is
**disparate treatment**, which ECOA prohibits in credit. A production system should prefer
methods that do not condition the decision on the protected attribute: reject-option
classification around the decision boundary, or calibrated equalized-odds post-processing.

Method C has no such problem — it only changes how the model is *trained*, not how decisions are
made — which is why it is worth pursuing further despite falling one point short here.

## Limitations

1. The data is **synthetic**; the numbers describe this model and dataset, not real applicants.
2. Only the age axis was mitigated; home ownership remains severely unfair under all variants.
3. Demographic parity is not reachable here without a much larger recall sacrifice.
4. All variants use a single reference-group threshold (0.5).
