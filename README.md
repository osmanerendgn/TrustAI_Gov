# TrustAI Internship — Fairness Assignments

Osman Eren Doğan · Product area: TrustAI-X

Fairness work on a simple credit-risk classifier trained on a synthetic İstanbul dataset. Three
linked tasks: measure fairness, act on it, and report it for governance.

| Task | Title | Status |
|---|---|---|
| [1](task1_fairness_metrics/) | Compare Basic Fairness Metrics | ✅ done |
| [2](task2_governance_report/) | Governance-Based Fairness Assessment Report | ✅ done |
| [3](task3_mitigation/) | Fairness Mitigation and Before/After Comparison | ✅ done |

## Structure

```
src/                      shared code
  paths.py                where each task writes its output
  data_prep.py            features and sensitive group definitions
  train_model.py          logistic regression baseline (reusable helpers)
  fairness_metrics.py     the three metrics, computed from scratch
  mitigation.py           the three mitigation experiments
task1_fairness_metrics/   metric comparison + results + notebook
task2_governance_report/  governance assessment
task3_mitigation/         mitigation experiments + before/after tables
```

## Setup

```bash
pip install -r requirements.txt
```

The dataset (~15 MB) is downloaded automatically on first run and is not committed.

```bash
python src/data_prep.py        # features and group base rates
python src/train_model.py      # baseline classifier
python src/fairness_metrics.py # Task 1: the three metrics
python src/mitigation.py       # Task 3: the three mitigations
```

## Model

Standard scaling + logistic regression, 11 features, 80/20 stratified split, seed 42, held-out
ROC AUC **0.914**. Deliberately simple — the assignments are about fairness, not accuracy.
`class_weight="balanced"` compensates for only 26 % of applicants defaulting.

**Framing.** In lending the favourable outcome is being **approved**, so every metric is computed
on approval rates rather than the raw model output.

## Task 1 — the three metrics

| Attribute | Approval prot./ref. | Demographic parity | Equal opportunity | Disparate impact |
|---|---|---|---|---|
| Education | 0.605 / 0.814 | −0.208 ❌ | **−0.028 ✅** | 0.744 ❌ |
| Age | 0.455 / 0.708 | −0.253 ❌ | −0.165 ❌ | 0.642 ❌ |
| Home ownership | 0.245 / 0.865 | −0.620 ❌ | −0.468 ❌ | 0.283 ❌ |

The model **fails the four-fifths rule on all three attributes**. Education is the instructive
case: the three metrics *disagree*. Demographic parity and disparate impact fail, but equal
opportunity passes — among applicants who genuinely repay, both groups are approved at almost the
same rate. A single fairness number is never enough; you have to say which definition you mean.

→ [Task 1 details](task1_fairness_metrics/README.md)

## Task 2 — governance assessment

| ID | Risk | Affected group | Severity | DI before → after |
|---|---|---|---|---|
| FR-01 | Education level | below university (72.6 %) | Medium | 0.744 → 0.763 ❌ |
| FR-02 | Age | young < 30 (18.2 %) | High | 0.642 → **0.822 ✅** |
| FR-03 | Home ownership | non-homeowner (32.7 %) | **Critical** | 0.283 → 0.294 ❌ |

**Deployment decision: 🔴 NO-GO.** Two of three attributes stay below the four-fifths threshold
under every mitigation tested. Home ownership sits at roughly a third of the legal threshold and
is essentially unmoved. Age can be fixed, but only by a method that is itself disparate treatment
under ECOA.

→ [Full report (HTML)](task2_governance_report/report.html) · [Task 2 details](task2_governance_report/README.md)

## Task 3 — mitigation on the age axis

| Variant | Disparate impact (age) | Equal opportunity | Defaulters approved |
|---|---|---|---|
| baseline | 0.642 ❌ | −0.165 ❌ | 654 |
| A: remove age + proxies | 0.709 ❌ | −0.096 ✅ | 670 |
| **B: group threshold** | **0.822 ✅** | −0.031 ✅ | **824** |
| C: reweighting | 0.789 ❌ | −0.035 ✅ | 733 |

**Only the group-specific threshold reaches legal compliance**, at a cost of 170 additional
approved defaulters per 20,000 applications. Removing age and its proxies is nearly free — ROC
AUC even rises slightly — but is not sufficient alone. Mitigating age does **not** improve
education or home ownership; home ownership stays near a third of the legal threshold under every
variant.

→ [Task 3 details](task3_mitigation/README.md)

## Data source

Synthetic dataset from
[atalaydenknalbant/underbanked_risk_estimation](https://github.com/atalaydenknalbant/underbanked_risk_estimation),
accompanying *Credit Risk Estimation with Non-Financial Features*
([arXiv:2512.12783](https://arxiv.org/abs/2512.12783)).
