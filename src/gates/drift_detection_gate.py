import argparse
import logging

import numpy as np
from scipy import stats
from utils.logging_utils import setup_logging
from data.preprocessing import load_data, compute_baseline_stats, infer_schema

logger = logging.getLogger(__name__)


def _psi_for_column(
    baseline: np.ndarray, current: np.ndarray, n_bins: int = 10, eps: float = 1e-6
) -> float:
    """
    Compute Population Stability Index between baseline and current distributions.
    PSI < 0.10  → No significant change
    PSI 0.10–0.20 → Moderate change (warning)
    PSI > 0.20  → Significant change (drift)
    """
    global_min = min(baseline.min(), current.min())
    global_max = max(baseline.max(), current.max())
    bins = np.linspace(global_min, global_max, n_bins + 1)

    baseline_counts, _ = np.histogram(baseline, bins=bins)
    current_counts, _ = np.histogram(current, bins=bins)

    baseline_pct = (baseline_counts / len(baseline)) + eps
    current_pct = (current_counts / len(current)) + eps

    psi = float(
        np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    )
    return psi


def _ks_test(baseline: np.ndarray, current: np.ndarray) -> tuple[float, float]:
    """Return (statistic, p_value) from the KS two-sample test."""
    result = stats.ks_2samp(baseline, current)
    return float(result.statistic), float(result.pvalue)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Drift Detection Gate")
    p.add_argument(
        "--gold-data-version",
        type=str,
        required=True,
        help="Mounted input path for gold star data",
    )
    p.add_argument(
        "--new-data-version",
        type=str,
        required=True,
        help="Mounted input path for new data to check",
    )
    p.add_argument(
        "--drift-threshold",
        type=float,
        default=0.05,
        help="PSI threshold. Defaults to 0.05",
    )
    return p.parse_args()


def run(args: argparse.Namespace) -> bool:
    setup_logging()

    # Step 1: Load data asset ----------------
    baseline = load_data(args.gold_data_version)
    logger.info(f"Baseline Data shape: {baseline.shape}")

    new_data, _ = load_data(args.new_data_version)
    logger.info(f"New Data Shape: {new_data.shape}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
