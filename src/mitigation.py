"""Fairness mitigation experiments and the before/after comparison.

Target attribute: **age**. Age is a legally protected class in credit lending
(e.g. under the US Equal Credit Opportunity Act), and the baseline fails all
three fairness metrics on it, so there is a real gap to close.

Three simple mitigations are applied to the same baseline and evaluated on the
identical test rows:

  A  Remove the sensitive feature and its strongest proxies
     "Fairness through unawareness" - stop the model from seeing age at all.

  B  Group-specific decision threshold
     Keep one model, but require a higher risk score before rejecting a young
     applicant. The threshold is tuned on out-of-fold training predictions,
     never on the test set.

  C  Sample reweighting (Kamiran and Calders)
     Reweight training rows so that every (group, outcome) cell carries the
     influence it would have if group and outcome were independent.

Usage:
    python src/mitigation.py
"""

import sys
from pathlib import Path
from typing import Dict, List

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from sklearn.metrics import confusion_matrix, roc_auc_score  # noqa: E402
from sklearn.model_selection import cross_val_predict  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_prep import FEATURES, SENSITIVE_GROUPS, TARGET, prepare  # noqa: E402
from fairness_metrics import FOUR_FIFTHS, compute_metrics  # noqa: E402
from paths import TASK3_RESULTS, ensure_dirs  # noqa: E402
from train_model import (  # noqa: E402
    THRESHOLD, build_model, build_predictions, fit_and_score, split_indices,
)

TARGET_ATTRIBUTE = "age"
PROTECTED = SENSITIVE_GROUPS[TARGET_ATTRIBUTE]["protected"]
REFERENCE = SENSITIVE_GROUPS[TARGET_ATTRIBUTE]["reference"]

# Aim above the legal 0.80 line rather than at it: a rule tuned to land exactly
# on the boundary has no room for sampling noise and fails out of sample.
DI_TARGET = 0.82
THRESHOLD_GRID = np.arange(0.30, 0.951, 0.01)
PROXY_COUNT = 3


def find_age_proxies(df: pd.DataFrame, k: int = PROXY_COUNT) -> List[str]:
    """The k features most correlated with age - the channels that leak it."""
    others = [f for f in FEATURES if f != "age"]
    correlations = df[others].corrwith(df["age"]).abs().sort_values(ascending=False)
    return list(correlations.head(k).index)


def reweighting_weights(df: pd.DataFrame, train_idx: pd.Index) -> np.ndarray:
    """Kamiran and Calders weights: P(group) * P(outcome) / P(group, outcome).

    Cells under-represented relative to independence are weighted up, so the
    model can no longer lower its loss by exploiting the association between
    group membership and outcome.
    """
    train = df.loc[train_idx]
    groups = train[f"{TARGET_ATTRIBUTE}_group"]
    outcomes = train[TARGET]

    weights = np.ones(len(train), dtype=float)
    for group in groups.unique():
        for outcome in outcomes.unique():
            cell = ((groups == group) & (outcomes == outcome)).to_numpy()
            observed = cell.mean()
            if observed == 0:
                continue
            expected = (groups == group).mean() * (outcomes == outcome).mean()
            weights[cell] = expected / observed
    return weights


def tune_group_threshold(df: pd.DataFrame, train_idx: pd.Index) -> float:
    """Smallest protected-group threshold reaching DI_TARGET out of fold.

    Cross-validated predictions on the training rows are used, so the threshold
    never sees the test set. Falls back to the shared threshold if the grid
    cannot reach the target.
    """
    oof_probability = cross_val_predict(
        build_model(), df.loc[train_idx, FEATURES], df.loc[train_idx, TARGET],
        cv=5, method="predict_proba", n_jobs=-1,
    )[:, 1]

    groups = df.loc[train_idx, f"{TARGET_ATTRIBUTE}_group"].to_numpy()
    repays = 1 - df.loc[train_idx, TARGET].to_numpy()
    is_protected = groups == PROTECTED

    sweep = []
    for threshold in THRESHOLD_GRID:
        flagged = np.where(
            is_protected, oof_probability >= threshold, oof_probability >= THRESHOLD
        )
        frame = pd.DataFrame({
            "approved": 1 - flagged.astype(int),
            "repays": repays,
            f"{TARGET_ATTRIBUTE}_group": groups,
        })
        metrics = compute_metrics(frame, TARGET_ATTRIBUTE, PROTECTED, REFERENCE)
        sweep.append((round(float(threshold), 2), metrics["disparate_impact_ratio"]))

    sweep_df = pd.DataFrame(sweep, columns=["threshold", "di_ratio"])
    sweep_df.to_csv(TASK3_RESULTS / "threshold_sweep.csv", index=False)

    reaching = sweep_df[sweep_df["di_ratio"] >= DI_TARGET]
    return float(reaching["threshold"].min()) if len(reaching) else THRESHOLD


def performance_row(name: str, df: pd.DataFrame, test_idx: pd.Index,
                    predictions: pd.DataFrame, probability: np.ndarray) -> Dict[str, float]:
    """Accuracy, recall and the two error counts that actually cost money."""
    truth = df.loc[test_idx, TARGET].to_numpy()
    flagged_risky = 1 - predictions["approved"].to_numpy()
    tn, fp, fn, tp = confusion_matrix(truth, flagged_risky).ravel()
    return {
        "variant": name,
        "roc_auc": round(float(roc_auc_score(truth, probability)), 4),
        "accuracy": round(float((tn + tp) / len(truth)), 4),
        "recall_delinquency": round(float(tp / (tp + fn)), 4),
        "precision_delinquency": round(float(tp / (tp + fp)), 4),
        "defaulters_approved_FN": int(fn),
        "good_payers_rejected_FP": int(fp),
        "approval_rate": round(float(predictions["approved"].mean()), 4),
    }


def fairness_rows(name: str, predictions: pd.DataFrame) -> List[Dict[str, float]]:
    """Fairness metrics for every attribute, so spillover stays visible."""
    rows = []
    for attribute, spec in SENSITIVE_GROUPS.items():
        metrics = compute_metrics(predictions, attribute, spec["protected"], spec["reference"])
        metrics["variant"] = name
        rows.append(metrics)
    return rows


def plot_before_after(fairness: pd.DataFrame) -> Path:
    """Disparate impact per variant, for the target attribute and the other two."""
    subset = fairness[fairness["attribute"] == TARGET_ATTRIBUTE]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    axes[0].bar(subset["variant"], subset["disparate_impact_ratio"],
                color=["#8C8C8C", "#DD8452", "#4878CF", "#6ACC65"])
    axes[0].axhline(FOUR_FIFTHS, color="red", ls="--", lw=1, label="four-fifths rule")
    axes[0].set_ylabel("disparate impact ratio")
    axes[0].set_title(f"{TARGET_ATTRIBUTE}: disparate impact by variant")
    axes[0].tick_params(axis="x", rotation=18, labelsize=8)
    axes[0].grid(axis="y", alpha=0.3)
    axes[0].legend(fontsize=8)

    attributes = list(fairness["attribute"].unique())
    variants = list(fairness["variant"].unique())
    width = 0.8 / len(variants)
    positions = np.arange(len(attributes))
    for offset, variant in enumerate(variants):
        values = [
            fairness[(fairness["variant"] == variant) & (fairness["attribute"] == a)]
            ["disparate_impact_ratio"].iloc[0] for a in attributes
        ]
        axes[1].bar(positions + (offset - (len(variants) - 1) / 2) * width,
                    values, width=width, label=variant)
    axes[1].axhline(FOUR_FIFTHS, color="red", ls="--", lw=1)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(attributes, fontsize=9)
    axes[1].set_ylabel("disparate impact ratio")
    axes[1].set_title("Spillover: effect on the other attributes")
    axes[1].grid(axis="y", alpha=0.3)
    axes[1].legend(fontsize=7)

    fig.tight_layout()
    path = TASK3_RESULTS / "fig_before_after.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main() -> None:
    ensure_dirs()
    df = prepare()
    train_idx, test_idx = split_indices(df)

    fairness: List[Dict[str, float]] = []
    performance: List[Dict[str, float]] = []
    notes: List[str] = []

    # ---------------------------------------------------------- baseline
    _, probability = fit_and_score(df, FEATURES, train_idx, test_idx)
    baseline = build_predictions(df, test_idx, probability)
    fairness += fairness_rows("baseline", baseline)
    performance.append(performance_row("baseline", df, test_idx, baseline, probability))

    # ------------------------------- A: remove sensitive + proxy features
    proxies = find_age_proxies(df)
    blind_features = [f for f in FEATURES if f not in (["age"] + proxies)]
    notes.append(f"A: dropped age plus its {len(proxies)} strongest proxies {proxies}")
    _, probability_a = fit_and_score(df, blind_features, train_idx, test_idx)
    variant_a = build_predictions(df, test_idx, probability_a)
    fairness += fairness_rows("A: remove age+proxies", variant_a)
    performance.append(
        performance_row("A: remove age+proxies", df, test_idx, variant_a, probability_a))

    # ------------------------------------- B: group-specific threshold
    protected_threshold = tune_group_threshold(df, train_idx)
    notes.append(
        f"B: protected-group threshold {protected_threshold:.2f} "
        f"(tuned out of fold for DI >= {DI_TARGET}); reference stays {THRESHOLD}"
    )
    is_protected = (df.loc[test_idx, f"{TARGET_ATTRIBUTE}_group"] == PROTECTED).to_numpy()
    flagged = np.where(is_protected, probability >= protected_threshold, probability >= THRESHOLD)
    variant_b = baseline.copy()
    variant_b["approved"] = 1 - flagged.astype(int)
    fairness += fairness_rows("B: group threshold", variant_b)
    performance.append(
        performance_row("B: group threshold", df, test_idx, variant_b, probability))

    # --------------------------------------------- C: sample reweighting
    weights = reweighting_weights(df, train_idx)
    notes.append(f"C: reweighting, weight range {weights.min():.3f} to {weights.max():.3f}")
    _, probability_c = fit_and_score(df, FEATURES, train_idx, test_idx, sample_weight=weights)
    variant_c = build_predictions(df, test_idx, probability_c)
    fairness += fairness_rows("C: reweighting", variant_c)
    performance.append(
        performance_row("C: reweighting", df, test_idx, variant_c, probability_c))

    # ------------------------------------------------------------ output
    fairness_df = pd.DataFrame(fairness)
    performance_df = pd.DataFrame(performance)
    fairness_df.to_csv(TASK3_RESULTS / "before_after_fairness.csv", index=False)
    performance_df.to_csv(TASK3_RESULTS / "before_after_performance.csv", index=False)
    (TASK3_RESULTS / "mitigation_notes.txt").write_text("\n".join(notes), encoding="utf-8")
    figure = plot_before_after(fairness_df)

    target = fairness_df[fairness_df["attribute"] == TARGET_ATTRIBUTE][[
        "variant", "approval_rate_protected", "approval_rate_reference",
        "demographic_parity_difference", "equal_opportunity_difference",
        "disparate_impact_ratio",
    ]]
    print(f"=== {TARGET_ATTRIBUTE}: fairness before and after ===")
    print(target.to_string(index=False))
    print("\n=== performance cost ===")
    print(performance_df.to_string(index=False))
    print("\n=== spillover (disparate impact, all attributes) ===")
    print(fairness_df.pivot(index="attribute", columns="variant",
                            values="disparate_impact_ratio").round(4).to_string())
    print("\n" + "\n".join(notes))
    print(f"\nwrote before_after_fairness.csv, before_after_performance.csv, {figure.name}")


if __name__ == "__main__":
    main()
