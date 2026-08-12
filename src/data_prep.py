"""Data preparation for the fairness comparison.

Loads the synthetic Istanbul credit dataset, builds the features the classifier
uses, and defines the sensitive groups the fairness metrics are computed over.

Usage:
    python src/data_prep.py
"""

import ast
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "istanbul_synthetic_data_v22.csv"
RESULTS = ROOT / "results"
SOURCE_URL = (
    "https://raw.githubusercontent.com/atalaydenknalbant/"
    "underbanked_risk_estimation/main/istanbul_synthetic_data_v22.csv"
)

TARGET = "delinquency_FL"
REFERENCE_DATE = pd.Timestamp("2025-01-01")
UNIVERSITY = ["University", "Masters", "Doctorate"]

# Each sensitive attribute is a (protected group, reference group) pair.
# "Protected" means the group we are checking for disadvantage.
SENSITIVE_GROUPS = {
    "education": {"protected": "below university", "reference": "university+"},
    "age": {"protected": "young (<30)", "reference": "30+"},
    "home_ownership": {"protected": "non-homeowner", "reference": "homeowner"},
}

FEATURES = [
    "age", "owns_car", "owns_home", "owns_credit_card",
    "online_shopping_frequency", "social_media_active",
    "income_log", "rent_log", "phone_age_days", "subscription_count",
    "is_unemployed",
]


def download_if_missing(path: Path = DATA) -> Path:
    """Fetch the dataset from the source repository the first time it is needed."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading dataset -> {path}")
        urllib.request.urlretrieve(SOURCE_URL, path)
    return path


def count_subscriptions(cell: object) -> int:
    """Number of entries in a stringified subscription list, e.g. "['Gym', 'Netflix']"."""
    if pd.isna(cell):
        return 0
    try:
        return len(ast.literal_eval(str(cell)))
    except (ValueError, SyntaxError):
        return 0


def prepare(path: Path = DATA) -> pd.DataFrame:
    """Return the dataset with model features and sensitive group labels added."""
    df = pd.read_csv(download_if_missing(path))

    # Sensitive group labels — kept out of the feature list, used only for auditing.
    df["education_group"] = np.where(
        df["education"].isin(UNIVERSITY), "university+", "below university")
    df["age_group"] = np.where(df["age"] < 30, "young (<30)", "30+")
    df["home_ownership_group"] = np.where(
        df["owns_home"], "homeowner", "non-homeowner")

    # Features. Income and rent are heavily right-skewed, so they are log-scaled;
    # log1p rather than log because rent is 0 for most homeowners.
    for col in ["owns_car", "owns_home", "owns_credit_card", "social_media_active"]:
        df[col] = df[col].astype(int)
    df["income_log"] = np.log1p(df["monthly_income"])
    df["rent_log"] = np.log1p(df["monthly_rent"])
    df["phone_age_days"] = (
        REFERENCE_DATE - pd.to_datetime(df["phone_purchase_date"], errors="coerce")
    ).dt.days
    df["subscription_count"] = df["monthly_subscriptions"].apply(count_subscriptions)
    df["is_unemployed"] = (df["employment_status"] == "Unemployed").astype(int)

    return df


def group_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Size and true delinquency rate of every sensitive group.

    Worth reading before the fairness results: if two groups genuinely default at
    different rates, an accurate model will approve them at different rates too.
    """
    rows = []
    for attribute, spec in SENSITIVE_GROUPS.items():
        column = f"{attribute}_group"
        for group in (spec["protected"], spec["reference"]):
            subset = df[df[column] == group]
            rows.append({
                "attribute": attribute,
                "group": group,
                "role": "protected" if group == spec["protected"] else "reference",
                "n": len(subset),
                "share": round(len(subset) / len(df), 4),
                "true_delinquency_rate": round(subset[TARGET].mean(), 4),
            })
    return pd.DataFrame(rows)


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    df = prepare()
    summary = group_summary(df)
    summary.to_csv(RESULTS / "group_summary.csv", index=False)

    print(f"rows: {len(df):,}   features: {len(FEATURES)}")
    print(f"overall delinquency rate: {df[TARGET].mean():.4f}\n")
    print(summary.to_string(index=False))
    print(f"\nwrote {RESULTS / 'group_summary.csv'}")


if __name__ == "__main__":
    main()
