# Task 2 — Governance-Based Fairness Assessment Report

Turn the Task 1 fairness results into a governance artefact: identified risks, affected groups,
severity, possible impact, recommended mitigations, monitoring requirements, and a deployment
decision.

## The report

**→ [`report.html`](report.html)** — self-contained, figures embedded, prints cleanly to PDF
(browser → Print → Save as PDF).
**→ [`report.md`](report.md)** — same content as Markdown.

## Regenerate

```bash
python ../src/governance_report.py   # risk register + deployment decision
python ../src/render_report.py       # report.md and report.html
```

Both read the Task 1 and Task 3 result files, so the report cannot drift from the experiments it
describes. No figure in it is transcribed by hand.

## Risk register

| ID | Risk | Affected group | Share | Protected class? | Severity | DI ratio | EO gap |
|---|---|---|---|---|---|---|---|
| FR-01 | Adverse impact on education level | below university | 72.6 % | No — SES proxy | **Medium** | 0.744 | −0.028 |
| FR-02 | Adverse impact on age | young (< 30) | 18.2 % | **Yes — ECOA** | **High** | 0.642 | −0.165 |
| FR-03 | Adverse impact on home ownership | non-homeowner | 32.7 % | No — wealth proxy | **Critical** | 0.283 | −0.468 |

**Severity method.** Disparate impact drives the rating because it is the legally operative test
(< 0.50 Critical, < 0.70 High, < 0.80 Medium). The equal-opportunity gap can raise a rating but
never lower it (≥ 0.30 Critical, ≥ 0.15 High, ≥ 0.10 Medium). Likelihood is *certain* for every
row — these are measurements on held-out data, not forecasts.

## Deployment decision

> ### 🔴 NO-GO
> Two of three attributes remain below the four-fifths threshold even under the best mitigation
> tested. Blocking risks: **FR-01 (education)** and **FR-03 (home ownership)**.

Age (FR-02) *can* be brought into compliance — 0.642 → 0.822 with a group-specific threshold —
but at a cost of **170 additional approved defaulters per 20,000 applications**, and by a method
that is itself **disparate treatment** under ECOA. Home ownership moves 0.283 → 0.294 under the
best remedy: essentially unchanged, and still about a third of the legal threshold.

### What would change the decision

1. A remedy that lifts education and home ownership above 0.80 **without** disparate treatment —
   reject-option classification or calibrated equalized odds, neither tested here.
2. Evidence from **real** applicant data; every finding rests on a synthetic dataset.
3. A documented business justification for the residual gap, reviewed by legal.

## Monitoring requirements

Required before any go-live:

| # | Metric | Threshold | Frequency | Action on breach |
|---|---|---|---|---|
| M1 | Disparate impact, per protected attribute | ≥ 0.80 | Monthly | Halt approvals for the affected group; escalate |
| M2 | Equal opportunity gap | ≤ 0.10 | Monthly | Investigate within 5 business days |
| M3 | Approval rate per group | ±5 pp vs prior quarter | Monthly | Investigate drift |
| M4 | Delinquency recall | ≥ 0.80 | Monthly | Review threshold settings |
| M5 | Group population shares | ±10 % vs training | Quarterly | Trigger revalidation |
| M6 | Full fairness re-audit | — | Quarterly + every retrain | Re-run this assessment |

Plus: log score, threshold and decision for every application so any decision can be
reconstructed; keep an adverse-action reason per rejection; route model changes through a
documented approval step.

## Files

| File | Purpose |
|---|---|
| `report.html` | the deliverable — self-contained, print-to-PDF ready |
| `report.md` | same content, Markdown |
| `results/risk_register.csv` | the register as data |
| `results/deployment_decision.json` | verdict, rationale, blocking risks |
