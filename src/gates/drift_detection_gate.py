import argparse
import logging
import sys
from pathlib import Path
import numpy as np
from scipy import stats
from utils.logging_utils import setup_logging
from data.preprocessing import load_data, compute_baseline_stats, infer_schema, clean

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
    p.add_argument(
        "--drift-output-path",
        type=str,
        required=True,
        help="Path to write drift detection results",
    )
    return p.parse_args()


def run(args: argparse.Namespace) -> bool:
    setup_logging()

    # Step 1: Load data asset ----------------
    baseline_dirty = load_data(args.gold_data_version)
    logger.info(f"Baseline Data shape: {baseline_dirty.shape}")

    train, _ = load_data(args.new_data_version)
    logger.info(f"New Data Shape: {train.shape}")

    logger.info("Inferring schema from baseline")
    baseline = clean(baseline_dirty)
    num_cols, _ = infer_schema(baseline)

    # compute baseline stats and get data ready
    baseline_stats = compute_baseline_stats(baseline, num_cols)
    logger.info("Baseline statistics calculated")

    # ---- Compute PSI and KS per feature
    drift_results: dict[str, dict] = {}
    drifted_features: list[str] = []

    for col in num_cols:
        if col not in train.columns:
            logger.warning(f"Column {col} missing from current data - skipping")
            continue

        current_vals = train[col].dropna().values
        bstats = baseline_stats[col]
        rng = np.random.default_rng(42)
        baseline_vals = rng.normal(
            loc=bstats["mean"],
            scale=max(bstats["std"], 1e-6),
            size=bstats["n"],
        )

        psi = _psi_for_column(baseline_vals, current_vals)
        ks_stat, ks_pval = _ks_test(baseline_vals, current_vals)

        drift_results[col] = {
            "psi": psi,
            "ks_stat": ks_stat,
            "ks_pval": ks_pval,
            "drifted": psi > args.drift_threshold,
        }

        if psi > args.drift_threshold:
            drifted_features.append(col)
            logger.warning(
                f"DRIFT detected in {col}: PSI={psi:.4f} > threshold={args.drift_threshold:.4f}"
            )
        else:
            logger.info(f"Stable: {col} PSI={psi:.4f}")

    max_psi = max((r["psi"] for r in drift_results.values()), default=0.0)
    n_drifted = len(drifted_features)
    stable = n_drifted == 0

    if stable:
        logger.info(f"Drift gate PASSED - all features stable (max PSI={max_psi:.4f}")
    else:
        logger.error(
            f"Drift gate FAILED - {n_drifted} feature(s) drifted: {drifted_features}"
        )

    output_path = Path(args.drift_output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("stable" if stable else "drifted")
    logger.info(f"Wrote drift result {stable} to {output_path}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
