"""Render the governance report as Markdown and as a self-contained HTML file.

The report body is assembled from the result files, so the prose and the tables
always agree with the experiments. The HTML version inlines its figures as data
URIs, which makes it a single file that can be emailed or printed to PDF without
losing anything.

Usage:
    python src/render_report.py
"""

import json
import re
import sys
from pathlib import Path

import markdown
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from governance_report import (  # noqa: E402
    TASK2_DIR, TASK2_RESULTS, build_risk_register, deployment_decision, embed_image,
)
from paths import ROOT, TASK1_RESULTS, TASK3_RESULTS  # noqa: E402

STYLE = """
:root { --ink:#1a1a1a; --muted:#5b6470; --rule:#d8dee6; --accent:#1f4e79;
        --crit:#b3261e; --high:#c26a00; --med:#7a6a00; --low:#1b6b3a; }
* { box-sizing:border-box; }
body { font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
       line-height:1.6; color:var(--ink); max-width:900px; margin:0 auto;
       padding:48px 28px; background:#fff; }
h1 { font-size:1.9rem; border-bottom:3px solid var(--accent); padding-bottom:.4rem; }
h2 { font-size:1.35rem; margin-top:2.4rem; color:var(--accent);
     border-bottom:1px solid var(--rule); padding-bottom:.3rem; }
h3 { font-size:1.08rem; margin-top:1.6rem; }
table { border-collapse:collapse; width:100%; margin:1.1rem 0; font-size:.9rem;
        display:block; overflow-x:auto; }
th,td { border:1px solid var(--rule); padding:7px 10px; text-align:left; vertical-align:top; }
th { background:#f2f5f8; font-weight:600; }
tr:nth-child(even) td { background:#fafbfc; }
img { max-width:100%; height:auto; border:1px solid var(--rule); border-radius:4px; margin:1rem 0; }
code { background:#f2f5f8; padding:1px 5px; border-radius:3px; font-size:.87em; }
blockquote { border-left:4px solid var(--accent); margin:1.2rem 0; padding:.6rem 1rem;
             background:#f7f9fb; color:var(--muted); }
.meta { color:var(--muted); font-size:.9rem; margin-bottom:2rem; }
.verdict { border:2px solid var(--crit); border-radius:6px; padding:16px 20px;
           margin:1.5rem 0; background:#fdf3f2; }
.verdict.go { border-color:var(--low); background:#f1f8f3; }
.verdict h3 { margin:0 0 .4rem 0; font-size:1.2rem; }
.sev-Critical { color:var(--crit); font-weight:700; }
.sev-High { color:var(--high); font-weight:700; }
.sev-Medium { color:var(--med); font-weight:600; }
.sev-Low { color:var(--low); font-weight:600; }
@media print { body { padding:0; max-width:none; } h2 { page-break-after:avoid; }
               table,img { page-break-inside:avoid; } }
"""


def md_table(frame: pd.DataFrame, columns, headers) -> str:
    """Render selected columns as a Markdown table."""
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def build_markdown() -> str:
    """Assemble the whole report from the result files."""
    audit = pd.read_csv(TASK1_RESULTS / "fairness_metrics.csv")
    groups = pd.read_csv(TASK1_RESULTS / "group_summary.csv")
    performance = json.loads((TASK1_RESULTS / "model_performance.json").read_text())
    fairness = pd.read_csv(TASK3_RESULTS / "before_after_fairness.csv")
    costs = pd.read_csv(TASK3_RESULTS / "before_after_performance.csv")

    register = build_risk_register(
        audit, groups, fairness[fairness["variant"] != "baseline"])
    decision = deployment_decision(register)

    baseline_cost = costs[costs["variant"] == "baseline"].iloc[0]
    threshold_cost = costs[costs["variant"] == "B: group threshold"].iloc[0]
    extra_defaulters = int(threshold_cost["defaulters_approved_FN"]
                           - baseline_cost["defaulters_approved_FN"])
    n_test = performance["n_test"]

    critical = register[register["severity"] == "Critical"]
    blocking = ", ".join(decision["blocking_attributes"])

    audit_view = audit.assign(
        Attribute=lambda d: d["attribute"],
        Approval=lambda d: d["approval_rate_protected"].map("{:.3f}".format) + " / "
                           + d["approval_rate_reference"].map("{:.3f}".format),
        DP=lambda d: d["demographic_parity_difference"].map("{:+.3f}".format) + " "
                     + d["demographic_parity_verdict"].map({"FAIR": "PASS", "UNFAIR": "**FAIL**"}),
        EO=lambda d: d["equal_opportunity_difference"].map("{:+.3f}".format) + " "
                     + d["equal_opportunity_verdict"].map({"FAIR": "PASS", "UNFAIR": "**FAIL**"}),
        DI=lambda d: d["disparate_impact_ratio"].map("{:.3f}".format) + " "
                     + d["disparate_impact_verdict"].map({"FAIR": "PASS", "UNFAIR": "**FAIL**"}),
    )

    spillover = fairness.pivot(index="attribute", columns="variant",
                               values="disparate_impact_ratio").reset_index()
    spillover_cols = ["attribute", "baseline", "A: remove age+proxies",
                      "B: group threshold", "C: reweighting"]

    return f"""# Governance-Based Fairness Assessment

**System:** credit-risk approval classifier (TrustAI-X)
**Assessed by:** Osman Eren Doğan · TrustAI Internship, Task 2
**Evidence base:** Task 1 fairness audit and Task 3 mitigation experiments
**Data:** 100,000 synthetic applicants; {n_test:,} held out for evaluation

---

## 1. Management summary

A machine-learning model decides which credit applicants are approved. We tested whether it
treats different groups of people equally, and whether the unequal treatment we found can be
fixed.

**What we found.** The model does not pass the standard legal fairness test — the *four-fifths
rule*, which says a disadvantaged group should be approved at least 80 % as often as the
reference group. It fails this test for **all three** groups we examined. The worst case is
**home ownership**: applicants who do not own a home are approved at **{register.loc[register['attribute'] == 'home_ownership', 'disparate_impact'].iloc[0]:.0%}**
of the rate of homeowners — roughly a third of the legal threshold.

**What we tried.** Three standard remedies were tested. One of them — adjusting the decision
threshold for the disadvantaged group — brings the **age** gap into compliance. It costs
approximately **{extra_defaulters} additional approved defaulters per {n_test:,} applications**,
which is a direct credit loss.

**What we could not fix.** Neither education nor home ownership reaches compliance under any
remedy tested. Home ownership barely moves at all.

**Recommendation: {decision['verdict']}.** {decision['rationale']} The blocking issues are
**{blocking}**. Deploying the model as it stands would expose the organisation to adverse-impact
liability on groups that make up a large share of applicants.

---

## 2. Technical fairness summary

### Model under assessment

| Property | Value |
|---|---|
| Algorithm | {performance['model']} |
| Features | {performance['n_features']} |
| Training / test rows | {performance['n_train']:,} / {performance['n_test']:,} |
| Decision threshold | {performance['threshold']} |
| ROC AUC (held out) | **{performance['roc_auc']}** |
| Overall approval rate | {performance['overall_approval_rate']:.1%} |

The favourable outcome is **approval**, so all metrics are computed on approval rates. The model
is a competent risk predictor — this assessment is about *distribution* of its decisions, not
accuracy.

### Group base rates

Some of the approval gap reflects genuine differences in default behaviour. These have to be on
the table before judging the model.

{md_table(groups, ["attribute", "group", "role", "n", "share", "true_delinquency_rate"],
          ["Attribute", "Group", "Role", "n", "Share", "True delinquency rate"])}

### Fairness metrics (baseline)

{md_table(audit_view, ["Attribute", "Approval", "DP", "EO", "DI"],
          ["Attribute", "Approval prot./ref.", "Demographic parity", "Equal opportunity",
           "Disparate impact"])}

![Approval rates by group](../task1_fairness_metrics/results/fig_approval_rates.png)

**Education is the instructive case:** the three metrics disagree. Demographic parity and
disparate impact fail, but equal opportunity passes — among applicants who genuinely repay, both
groups are approved at nearly the same rate. The gap there tracks the real difference in default
rates (32.8 % vs 8.7 %) rather than mistreatment of creditworthy people. **Age and home ownership
have no such defence:** they fail the merit-aware metric too.

---

## 3. Risk register

{md_table(register,
          ["risk_id", "risk", "affected_group", "population_share", "protected_class",
           "severity", "disparate_impact", "equal_opportunity_gap"],
          ["ID", "Risk", "Affected group", "Share", "Protected class?", "Severity",
           "DI ratio", "EO gap"])}

**Severity method.** Disparate impact drives the rating because it is the legally operative test
(< 0.50 Critical, < 0.70 High, < 0.80 Medium). The equal-opportunity gap can raise a rating but
never lower it (≥ 0.30 Critical, ≥ 0.15 High, ≥ 0.10 Medium). Likelihood is *certain* for every
row — these are measurements on held-out data, not forecasts.

### Possible impact

{md_table(register, ["risk_id", "impact"], ["ID", "Impact if deployed unchanged"])}

{"**" + str(len(critical)) + " risk(s) rated Critical.**" if len(critical) else ""}
Beyond the direct harm to applicants, an adverse-impact finding on a protected class carries
regulatory exposure (ECOA in the US, and comparable provisions elsewhere), remediation cost, and
reputational damage.

---

## 4. Recommended mitigation actions

Three remedies were tested on the **age** axis, each on the identical test rows.

{md_table(fairness[fairness["attribute"] == "age"],
          ["variant", "approval_rate_protected", "demographic_parity_difference",
           "equal_opportunity_difference", "disparate_impact_ratio"],
          ["Variant", "Approval (young)", "Demographic parity", "Equal opportunity",
           "Disparate impact"])}

### What each costs

{md_table(costs, ["variant", "roc_auc", "recall_delinquency", "defaulters_approved_FN",
                  "good_payers_rejected_FP"],
          ["Variant", "ROC AUC", "Recall (delinquency)", "Defaulters approved",
           "Good payers rejected"])}

![Before and after](../task3_mitigation/results/fig_before_after.png)

### Recommendations

1. **Adopt "remove sensitive + proxy features" permanently.** It improves fairness *and* accuracy
   slightly (ROC AUC {baseline_cost['roc_auc']} → {costs[costs['variant'] == 'A: remove age+proxies'].iloc[0]['roc_auc']}).
   There is no reason not to. It is not sufficient on its own.
2. **Apply the group-specific threshold for age only if the business accepts
   {extra_defaulters} additional approved defaulters per {n_test:,} applications.** It is the only
   remedy that reaches compliance. ⚠️ It also constitutes **disparate treatment** — the protected
   attribute enters the decision rule directly, which ECOA prohibits. Prefer
   reject-option classification or calibrated equalized odds in production.
3. **Do not deploy for home ownership or education under any tested remedy.** Investigate the
   feature set: home ownership behaves like a wealth proxy that the model leans on heavily.
4. **Re-run this assessment on real data before any production decision.** All findings here rest
   on synthetic data.

### Spillover — mitigating one attribute does not fix the others

{md_table(spillover, spillover_cols,
          ["Attribute", "Baseline", "A: remove", "B: threshold", "C: reweight"])}

Education and home ownership barely move under any variant. **A model cannot be declared fixed
because one attribute was remediated.**

---

## 5. Monitoring requirements

If any version of this model is deployed, the following must be in place before go-live.

| # | Metric | Threshold | Frequency | Action on breach |
|---|---|---|---|---|
| M1 | Disparate impact, every protected attribute | ≥ 0.80 | Monthly | Halt approvals for the affected group; escalate to model risk |
| M2 | Equal opportunity gap, every protected attribute | ≤ 0.10 | Monthly | Investigate within 5 business days |
| M3 | Approval rate per group | ±5 pp vs the previous quarter | Monthly | Investigate drift |
| M4 | Delinquency recall | ≥ 0.80 | Monthly | Review threshold settings |
| M5 | Group population shares | ±10 % vs training distribution | Quarterly | Trigger revalidation |
| M6 | Full fairness re-audit | — | Quarterly, and after every retrain | Re-run this assessment |

**Additional controls.** Log the decision, score and threshold for every application so any
decision can be reconstructed. Keep an adverse-action reason for each rejection. Route model
changes through a documented approval step, and give an accountable owner to each risk in
Section 3.

---

## 6. Deployment decision

<div class="verdict">
<h3>Decision: {decision['verdict']}</h3>
{decision['rationale']} Blocking risks: <strong>{', '.join(decision['blocking_risks'])}</strong>
({blocking}).
</div>

**Rationale.** The four-fifths rule is the operative legal test and the model fails it on
{len(register)} of {len(register)} attributes before mitigation, and on
{len(decision['blocking_risks'])} of {len(register)} after the best remedy tested. Home ownership
in particular sits at roughly a third of the threshold and is essentially unmoved by every
remedy. Age can be brought into compliance, but only by a method that is itself legally
problematic.

**What would change this decision:**

1. A remedy that brings home ownership and education above 0.80 without disparate treatment —
   reject-option classification and calibrated equalized odds are the obvious candidates and were
   not tested here.
2. Evidence from **real** applicant data, since these conclusions rest on a synthetic dataset.
3. A documented business justification for the residual gap, if one exists, reviewed by legal.

---

## 7. Limitations

1. The dataset is **synthetic**; every figure describes this model and this data, not real people.
2. Only three mitigation methods were tested, all of them simple. In-processing methods (fairness
   constraints during training) were not attempted.
3. Metrics are computed at a single decision threshold (0.5 for the reference group).
4. Demographic parity, equal opportunity and calibration cannot all hold simultaneously when
   groups have different base rates (Kleinberg et al.; Chouldechova). "Fair" is always relative to
   a stated definition — this assessment uses disparate impact as the operative test because it
   is the legal one.
5. Intersectional effects (e.g. young non-homeowners) were not assessed.

---

*Generated by `src/render_report.py` from the Task 1 and Task 3 result files. Every figure in
this report is read from those files rather than transcribed.*
"""


def main() -> None:
    TASK2_DIR.mkdir(parents=True, exist_ok=True)
    TASK2_RESULTS.mkdir(parents=True, exist_ok=True)

    body = build_markdown()
    (TASK2_DIR / "report.md").write_text(body, encoding="utf-8")

    html_body = markdown.markdown(
        body, extensions=["tables", "attr_list", "md_in_html"])

    # Inline every figure so the HTML is a single portable file.
    def inline(match: re.Match) -> str:
        src = match.group(1)
        path = (TASK2_DIR / src).resolve()
        return f'src="{embed_image(path)}"' if path.exists() else match.group(0)

    html_body = re.sub(r'src="([^"]+\.png)"', inline, html_body)

    # Colour the severity words in the risk table.
    for level in ("Critical", "High", "Medium", "Low"):
        html_body = html_body.replace(
            f"<td>{level}</td>", f'<td class="sev-{level}">{level}</td>')

    html = (
        "<!doctype html>\n<html lang=\"en\">\n<head>\n"
        "<meta charset=\"utf-8\">\n"
        "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
        "<title>Governance-Based Fairness Assessment</title>\n"
        f"<style>{STYLE}</style>\n</head>\n<body>\n{html_body}\n</body>\n</html>\n"
    )
    (TASK2_DIR / "report.html").write_text(html, encoding="utf-8")

    size_kb = (TASK2_DIR / "report.html").stat().st_size / 1024
    print(f"wrote {TASK2_DIR / 'report.md'}")
    print(f"wrote {TASK2_DIR / 'report.html'}  ({size_kb:,.0f} KB, self-contained)")
    print(f"embedded figures: {html.count('data:image/png;base64')}")


if __name__ == "__main__":
    main()
