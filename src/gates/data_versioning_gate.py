import argparse
import json
import logging
from pathlib import Path

import mlflow

from data.preprocessing import (
    clean,
    compute_baseline_stats,
    infer_schema,
    load_data,
    split,
)
from utils.logging_utils import configure_mlflow, setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data versioning gate")
    parser.add_argument(
        "--raw-data",
        type=str,
        required=True,
        help="Mounted input path to raw dataset (Azure ML uri_folder)",
    )
    parser.add_argument(
        "--output-training-path",
        type=str,
        required=True,
        help="Path to write the registered asset version number to",
    )

    return parser.parse_args()


def run(args: argparse.Namespace) -> str:

    setup_logging()
    configure_mlflow()

    # Step 1: Load data asset ----------------
    df = load_data(args.raw_data)
    logger.info(f"Data shape: {df.shape}")

    mlflow.set_tag("gate", "data-versioning")

    # Step 2: Clean data ---------------------
    df_clean = clean(df)
    logger.info(f"Cleaned data shape: {df_clean.shape}")
    mlflow.log_metric("raw_rows", len(df))
    mlflow.log_metric("clean_rows", len(df_clean))

    # Step 3: Split Data ---------------------
    train_df, val_df, test_df = split(df_clean)
    mlflow.log_metric("train_rows", len(train_df))
    mlflow.log_metric("val_rows", len(val_df))
    mlflow.log_metric("test_rows", len(test_df))

    # Step 4: Compute and Log Baselines:
    num_cols, _ = infer_schema(df_clean)
    baseline = compute_baseline_stats(df_clean, num_cols)
    mlflow.log_dict(baseline, "baseline/baseline.json")
    logger.info(f"Baseline Stats: {baseline}")

    # Step 5: Upload Processed Splits to Path
    # Write version to output file for the orchestrator to read
    output_path = Path(args.output_training_path)
    output_path.mkdir(parents=True, exist_ok=True)

    train_df.to_parquet(output_path / "train.parquet", index=False)
    val_df.to_parquet(output_path / "validation.parquet", index=False)
    test_df.to_parquet(output_path / "test.parquet", index=False)

    # add baseline data to this as well
    with open(output_path / "baseline.json", "w") as f:
        json.dump(baseline, f)

    logger.info(f"Wrote splits and metadata to {output_path}")

    mlflow.set_tag("gate.status", "PASSED")
    logger.info("Data versioning gate complete")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
