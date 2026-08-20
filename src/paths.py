"""Shared filesystem locations.

Each task writes its deliverables into its own folder; intermediates that only
exist to pass data between scripts go to .cache/ and are not committed.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

DATA = ROOT / "data" / "istanbul_synthetic_data_v22.csv"
CACHE = ROOT / ".cache"

TASK1_RESULTS = ROOT / "task1_fairness_metrics" / "results"
TASK3_RESULTS = ROOT / "task3_mitigation" / "results"

PREDICTIONS = CACHE / "predictions.csv"


def ensure_dirs() -> None:
    """Create the output directories if they do not exist yet."""
    for directory in (CACHE, TASK1_RESULTS, TASK3_RESULTS):
        directory.mkdir(parents=True, exist_ok=True)
