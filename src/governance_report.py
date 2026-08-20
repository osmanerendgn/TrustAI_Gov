"""Governance-based fairness assessment report.

Turns the Task 1 audit and the Task 3 mitigation experiments into a governance
artefact: a risk register with severity ratings, monitoring requirements and a
deployment decision.

Every number is read from the result files rather than typed in, so the report
cannot drift away from the experiments it describes.

Usage:
    python src/governance_report.py
"""

import base64
import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fairness_metrics import FOUR_FIFTHS, PARITY_TOLERANCE  # noqa: E402
from paths import ROOT, TASK1_RESULTS, TASK3_RESULTS  # noqa: E402

TASK2_DIR = ROOT / "task2_governance_report"
TASK2_RESULTS = TASK2_DIR / "results"

# Severity bands. Disparate impact is the legally operative test, so it drives
# the rating; the equal-opportunity gap can only raise it, never lower it.
#
# The two metrics run in opposite directions: a LOW disparate impact ratio is
# bad, while a LARGE equal-opportunity gap is bad. They therefore need separate
# banding functions - treating them the same rates every attribute Critical.
DI_BANDS = [(0.50, "Critical"), (0.70, "High"), (FOUR_FIFTHS, "Medium")]
EO_BANDS = [(0.30, "Critical"), (0.15, "High"), (PARITY_TOLERANCE, "Medium")]
SEVERITY_ORDER = ["Low", "Medium", "High", "Critical"]

ATTRIBUTE_LABELS = {
    "education": "Education level",
    "age": "Age",
    "home_ownership": "Home ownership",
}
# Whether the attribute is a protected class in credit lending, which changes
# how much regulatory weight a failure carries.
PROTECTED_CLASS = {
    "education": "No - proxy for socioeconomic status",
    "age": "Yes - protected under ECOA",
    "home_ownership": "No - proxy for wealth",
}


def band_lower_is_worse(value: float, bands: List) -> str:
    """Severity for a metric where a small value is bad, e.g. disparate impact."""
    for limit, label in bands:
        if value < limit:
            return label
    return "Low"


def band_higher_is_worse(value: float, bands: List) -> str:
    """Severity for a metric where a large value is bad, e.g. an equality gap."""
    for limit, label in bands:
        if value >= limit:
            return label
    return "Low"


def worst(*labels: str) -> str:
    """Highest severity among the labels."""
    return max(labels, key=SEVERITY_ORDER.index)


def build_risk_register(
    audit: pd.DataFrame, groups: pd.DataFrame, mitigated: pd.DataFrame
) -> pd.DataFrame:
    """One row per sensitive attribute, rated and paired with its best mitigation."""
    rows = []
    for i, (_, row) in enumerate(audit.iterrows(), start=1):
        attribute = row["attribute"]
        di = row["disparate_impact_ratio"]
        eo = abs(row["equal_opportunity_difference"])

        severity = worst(band_lower_is_worse(di, DI_BANDS), band_higher_is_worse(eo, EO_BANDS))

        protected_row = groups[
            (groups["attribute"] == attribute) & (groups["role"] == "protected")
        ].iloc[0]

        # Best achievable disparate impact across the tested mitigations.
        after = mitigated[mitigated["attribute"] == attribute]
        best = after.loc[after["disparate_impact_ratio"].idxmax()]
        residual = "Low" if best["disparate_impact_ratio"] >= FOUR_FIFTHS else severity

        rows.append({
            "risk_id": f"FR-{i:02d}",
            "attribute": attribute,
            "risk": f"Adverse impact on {ATTRIBUTE_LABELS[attribute].lower()}",
            "affected_group": protected_row["group"],
            "population_share": f"{protected_row['share']:.1%}",
            "protected_class": PROTECTED_CLASS[attribute],
            "disparate_impact": round(di, 4),
            "equal_opportunity_gap": round(row["equal_opportunity_difference"], 4),
            "severity": severity,
            "likelihood": "Certain - measured on held-out data",
            "impact": (
                f"{protected_row['group']} applicants are approved at "
                f"{di:.0%} of the reference group's rate"
            ),
            "best_mitigation": best["variant"],
            "disparate_impact_after": round(best["disparate_impact_ratio"], 4),
            "residual_severity": residual,
            "compliant_after_mitigation": bool(best["disparate_impact_ratio"] >= FOUR_FIFTHS),
        })
    return pd.DataFrame(rows)


def deployment_decision(register: pd.DataFrame) -> Dict[str, object]:
    """Go / Conditional Go / No-Go, derived from the register rather than asserted."""
    blocking = register[~register["compliant_after_mitigation"]]
    remediable = register[register["compliant_after_mitigation"]]

    if blocking.empty:
        verdict = "CONDITIONAL GO"
        rationale = ("Every attribute reaches the four-fifths threshold once the "
                     "recommended mitigation is applied.")
    else:
        verdict = "NO-GO"
        rationale = (
            f"{len(blocking)} of {len(register)} attributes remain below the "
            f"four-fifths threshold even under the best mitigation tested."
        )

    return {
        "verdict": verdict,
        "rationale": rationale,
        "blocking_risks": list(blocking["risk_id"]),
        "blocking_attributes": [ATTRIBUTE_LABELS[a] for a in blocking["attribute"]],
        "remediable_risks": list(remediable["risk_id"]),
        "remediable_attributes": [ATTRIBUTE_LABELS[a] for a in remediable["attribute"]],
    }


def embed_image(path: Path) -> str:
    """Inline a PNG as a data URI so the HTML report is a single self-contained file."""
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def markdown_table(frame: pd.DataFrame, columns: List[str], headers: List[str]) -> str:
    """Render selected columns as a GitHub-flavoured Markdown table."""
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in columns) + " |")
    return "\n".join(lines)


def main() -> None:
    TASK2_RESULTS.mkdir(parents=True, exist_ok=True)

    audit = pd.read_csv(TASK1_RESULTS / "fairness_metrics.csv")
    groups = pd.read_csv(TASK1_RESULTS / "group_summary.csv")
    performance = json.loads((TASK1_RESULTS / "model_performance.json").read_text())
    mitigation_fairness = pd.read_csv(TASK3_RESULTS / "before_after_fairness.csv")
    mitigation_performance = pd.read_csv(TASK3_RESULTS / "before_after_performance.csv")

    mitigated = mitigation_fairness[mitigation_fairness["variant"] != "baseline"]
    register = build_risk_register(audit, groups, mitigated)
    register.to_csv(TASK2_RESULTS / "risk_register.csv", index=False)

    decision = deployment_decision(register)
    (TASK2_RESULTS / "deployment_decision.json").write_text(
        json.dumps(decision, indent=2), encoding="utf-8"
    )

    print(register[[
        "risk_id", "affected_group", "disparate_impact", "severity",
        "best_mitigation", "disparate_impact_after", "compliant_after_mitigation",
    ]].to_string(index=False))
    print(f"\nDeployment decision: {decision['verdict']}")
    print(f"Blocking risks: {', '.join(decision['blocking_risks']) or 'none'}")
    print(f"\nwrote {TASK2_RESULTS / 'risk_register.csv'}")
    print(f"wrote {TASK2_RESULTS / 'deployment_decision.json'}")

    # The report itself is assembled by render_report.py, which imports these.
    return None


if __name__ == "__main__":
    main()
