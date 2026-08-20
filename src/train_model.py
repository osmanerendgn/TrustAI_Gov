"""Train the binary classifier whose fairness we audit.

Deliberately a simple model: logistic regression inside a scikit-learn pipeline.
The task is to compare fairness metrics, not to maximise accuracy, and a linear
model keeps the behaviour easy to reason about.

The helpers here are reused by the mitigation experiments, so that every
before/after comparison runs on the identical train/test rows.

Usage:
    python src/train_model.py
"""

import json
import sys
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_prep import FEATURES, SENSITIVE_GROUPS, TARGET, prepare
from paths import PREDICTIONS, TASK1_RESULTS, ensure_dirs

SEED = 42
TEST_SIZE = 0.20
THRESHOLD = 0.5


def split_indices(df: pd.DataFrame) -> Tuple[pd.Index, pd.Index]:
    """Train/test row indices.

    Splitting on the index rather than on a feature matrix means every variant -
    baseline, fewer features, reweighted - is evaluated on exactly the same test
    rows, which is what makes the before/after comparison meaningful.
    """
    train_idx, test_idx = train_test_split(
        df.index, test_size=TEST_SIZE, stratify=df[TARGET], random_state=SEED
    )
    return train_idx, test_idx


def build_model() -> Pipeline:
    """Standard scaling plus balanced logistic regression."""
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
    )


def fit_and_score(
    df: pd.DataFrame,
    features: Sequence[str],
    train_idx: pd.Index,
    test_idx: pd.Index,
    sample_weight: Optional[np.ndarray] = None,
) -> Tuple[Pipeline, np.ndarray]:
    """Fit on the training rows and return the model plus test-set probabilities."""
    model = build_model()
    fit_kwargs = {}
    if sample_weight is not None:
        fit_kwargs["logisticregression__sample_weight"] = sample_weight
    model.fit(df.loc[train_idx, list(features)], df.loc[train_idx, TARGET], **fit_kwargs)
    probability_default = model.predict_proba(df.loc[test_idx, list(features)])[:, 1]
    return model, probability_default


def build_predictions(
    df: pd.DataFrame,
    test_idx: pd.Index,
    probability_default: np.ndarray,
    threshold: float = THRESHOLD,
) -> pd.DataFrame:
    """Frame the model output as an approval decision, with the group labels attached.

    The favourable outcome in lending is being APPROVED, so every fairness metric
    is computed on approval rather than on the raw model output.
    """
    flagged_risky = (probability_default >= threshold).astype(int)
    predictions = pd.DataFrame({
        "probability_default": probability_default,
        "approved": 1 - flagged_risky,                       # 1 = model would approve
        "repays": 1 - df.loc[test_idx, TARGET].to_numpy(),   # 1 = truly repays
    })
    for attribute in SENSITIVE_GROUPS:
        column = f"{attribute}_group"
        predictions[column] = df.loc[test_idx, column].to_numpy()
    return predictions


def main() -> None:
    ensure_dirs()
    df = prepare()
    train_idx, test_idx = split_indices(df)

    _, probability_default = fit_and_score(df, FEATURES, train_idx, test_idx)
    predictions = build_predictions(df, test_idx, probability_default)
    predictions.to_csv(PREDICTIONS, index=False)

    performance = {
        "model": "LogisticRegression(class_weight='balanced')",
        "n_features": len(FEATURES),
        "n_train": len(train_idx),
        "n_test": len(test_idx),
        "threshold": THRESHOLD,
        "roc_auc": round(float(roc_auc_score(df.loc[test_idx, TARGET], probability_default)), 4),
        "overall_approval_rate": round(float(predictions["approved"].mean()), 4),
        "overall_repayment_rate": round(float(predictions["repays"].mean()), 4),
    }
    (TASK1_RESULTS / "model_performance.json").write_text(
        json.dumps(performance, indent=2), encoding="utf-8"
    )

    print(json.dumps(performance, indent=2))
    print(f"\nwrote {PREDICTIONS}")
    print(f"wrote {TASK1_RESULTS / 'model_performance.json'}")


if __name__ == "__main__":
    main()
