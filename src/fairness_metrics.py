"""The three fairness metrics.

Each is computed from scratch so the arithmetic is visible rather than hidden
behind a library call.

Notation: A is the sensitive attribute, Y-hat the model decision, Y the truth.
The favourable outcome is approval.

  Demographic parity difference
      P(approved | protected) - P(approved | reference)
      "Does the model approve the same share of people in each group?"
      Ignores whether people actually repay. Ideal 0; |gap| <= 0.10 accepted.

  Equal opportunity difference
      P(approved | protected, repays) - P(approved | reference, repays)
      "Among people who genuinely repay, is each group approved equally often?"
      Merit-aware: it only looks at applicants who deserve approval.
      Ideal 0; |gap| <= 0.10 accepted.

  Disparate impact ratio
      P(approved | protected) / P(approved | reference)
      The legal four-fifths rule: below 0.80 counts as adverse impact.
      Ideal 1.0.

Usage:
    python src/fairness_metrics.py
"""

import sys
from pathlib import Path
from typing import Dict

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_prep import SENSITIVE_GROUPS  # noqa: E402

from paths import PREDICTIONS, TASK1_RESULTS

PARITY_TOLERANCE = 0.10   # accepted gap for demographic parity / equal opportunity
FOUR_FIFTHS = 0.80        # legal threshold for disparate impact


def compute_metrics(
    predictions: pd.DataFrame, attribute: str, protected: str, reference: str
) -> Dict[str, float]:
    """Return per-group rates and the three fairness metrics for one attribute."""
    column = f"{attribute}_group"
    in_protected = predictions[column] == protected
    in_reference = predictions[column] == reference
    repays = predictions["repays"] == 1

    approval_protected = predictions.loc[in_protected, "approved"].mean()
    approval_reference = predictions.loc[in_reference, "approved"].mean()
    # Approval rate among applicants who genuinely repay = true positive rate.
    tpr_protected = predictions.loc[in_protected & repays, "approved"].mean()
    tpr_reference = predictions.loc[in_reference & repays, "approved"].mean()

    return {
        "attribute": attribute,
        "protected_group": protected,
        "reference_group": reference,
        "approval_rate_protected": round(approval_protected, 4),
        "approval_rate_reference": round(approval_reference, 4),
        "repayer_approval_protected": round(tpr_protected, 4),
        "repayer_approval_reference": round(tpr_reference, 4),
        "demographic_parity_difference": round(approval_protected - approval_reference, 4),
        "equal_opportunity_difference": round(tpr_protected - tpr_reference, 4),
        "disparate_impact_ratio": round(approval_protected / approval_reference, 4),
    }


def add_verdicts(table: pd.DataFrame) -> pd.DataFrame:
    """Mark each metric FAIR or UNFAIR against its conventional threshold."""
    table = table.copy()
    table["demographic_parity_verdict"] = [
        "FAIR" if abs(v) <= PARITY_TOLERANCE else "UNFAIR"
        for v in table["demographic_parity_difference"]
    ]
    table["equal_opportunity_verdict"] = [
        "FAIR" if abs(v) <= PARITY_TOLERANCE else "UNFAIR"
        for v in table["equal_opportunity_difference"]
    ]
    table["disparate_impact_verdict"] = [
        "FAIR" if v >= FOUR_FIFTHS else "UNFAIR"
        for v in table["disparate_impact_ratio"]
    ]
    return table


def plot_group_rates(table: pd.DataFrame) -> Path:
    """Approval rate side by side for the protected and reference group."""
    fig, axes = plt.subplots(1, len(table), figsize=(4 * len(table), 4), sharey=True)
    for ax, (_, row) in zip(axes, table.iterrows()):
        ax.bar([0, 1],
               [row["approval_rate_protected"], row["approval_rate_reference"]],
               color=["#C44E52", "#4878CF"], width=0.6)
        ax.axhline(row["approval_rate_reference"] * FOUR_FIFTHS,
                   color="black", ls="--", lw=1)
        ax.set_xticks([0, 1])
        ax.set_xticklabels([row["protected_group"], row["reference_group"]], fontsize=8)
        ax.set_title(f"{row['attribute']}\nDI ratio = {row['disparate_impact_ratio']:.3f}",
                     fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("approval rate")
    fig.suptitle("Approval rate by group (dashed line = four-fifths of the reference group)",
                 fontsize=11)
    fig.tight_layout()
    path = TASK1_RESULTS / "fig_approval_rates.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    predictions = pd.read_csv(PREDICTIONS)

    rows = [
        compute_metrics(predictions, attribute, spec["protected"], spec["reference"])
        for attribute, spec in SENSITIVE_GROUPS.items()
    ]
    table = add_verdicts(pd.DataFrame(rows))
    table.to_csv(TASK1_RESULTS / "fairness_metrics.csv", index=False)

    figure = plot_group_rates(table)

    display_columns = [
        "attribute", "approval_rate_protected", "approval_rate_reference",
        "demographic_parity_difference", "demographic_parity_verdict",
        "equal_opportunity_difference", "equal_opportunity_verdict",
        "disparate_impact_ratio", "disparate_impact_verdict",
    ]
    print(table[display_columns].to_string(index=False))
    print(f"\nwrote {TASK1_RESULTS / 'fairness_metrics.csv'}")
    print(f"wrote {figure}")


if __name__ == "__main__":
    main()
