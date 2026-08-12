"""Train the binary classifier whose fairness we audit.

Deliberately a simple model: logistic regression inside a scikit-learn pipeline.
The task is to compare fairness metrics, not to maximise accuracy, and a linear
model keeps the behaviour easy to reason about.

Usage:
    python src/train_model.py
"""

import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_prep import FEATURES, SENSITIVE_GROUPS, TARGET, prepare  # noqa: E402

RESULTS = Path(__file__).resolve().parents[1] / "results"

SEED = 42
TEST_SIZE = 0.20
THRESHOLD = 0.5


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    df = prepare()

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED
    )

    # class_weight="balanced" because only 26% of applicants default; without it
    # the model would under-predict the minority class.
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=1000, random_state=SEED),
    )
    model.fit(X_train, y_train)

    probability_default = model.predict_proba(X_test)[:, 1]
    flagged_risky = (probability_default >= THRESHOLD).astype(int)

    # The favourable outcome in lending is being APPROVED, so the fairness
    # metrics are computed on approval rather than on the raw model output.
    predictions = pd.DataFrame({
        "approved": 1 - flagged_risky,               # 1 = model would approve
        "repays": 1 - y_test.to_numpy(),             # 1 = applicant truly repays
    })
    for attribute in SENSITIVE_GROUPS:
        column = f"{attribute}_group"
        predictions[column] = df.loc[X_test.index, column].to_numpy()

    predictions.to_csv(RESULTS / "predictions.csv", index=False)

    performance = {
        "model": "LogisticRegression(class_weight='balanced')",
        "n_features": len(FEATURES),
        "n_train": len(X_train),
        "n_test": len(X_test),
        "threshold": THRESHOLD,
        "roc_auc": round(float(roc_auc_score(y_test, probability_default)), 4),
        "overall_approval_rate": round(float(predictions["approved"].mean()), 4),
        "overall_repayment_rate": round(float(predictions["repays"].mean()), 4),
    }
    (RESULTS / "model_performance.json").write_text(
        json.dumps(performance, indent=2), encoding="utf-8"
    )

    print(json.dumps(performance, indent=2))
    print(f"\nwrote {RESULTS / 'predictions.csv'}")
    print(f"wrote {RESULTS / 'model_performance.json'}")


if __name__ == "__main__":
    main()
