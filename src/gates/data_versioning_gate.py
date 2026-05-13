import argparse
import logging
from pathlib import Path

from utils.logging_utils import setup_logging
from utils.asset_utils import get_ml_client, register_dataframes_as_asset
from data.preprocessing import (
    clean,
    compute_baseline_stats,
    split,
    infer_schema,
    load_data,
)

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
        "--output-asset-name",
        type=str,
        default="modelling_assets",
        help="Name of the created data asset containing train, test, and validation data",
    )
    parser.add_argument(
        "--output-version-path",
        type=str,
        required=True,
        help="Path to write the registered asset version number to",
    )

    return parser.parse_args()


def run(args: argparse.Namespace) -> str:

    setup_logging()

    # Step 1: Load data asset ----------------
    df = load_data(args.raw_data)
    logger.info(f"Data shape: {df.shape}")

    # Step 2: Clean data ---------------------
    df_clean = clean(df)
    logger.info(f"Cleaned data shape: {df_clean.shape}")

    # Step 3: Split Data ---------------------
    train_df, val_df, test_df = split(df_clean)

    # Step 4: Compute and Log Baselines:
    num_cols, _ = infer_schema(df_clean)
    logger.info(f"Baseline Stats: {compute_baseline_stats(df_clean, num_cols)}")

    # Step 5: Upload Processed Splits to Path
    dataframes = {"train.csv": train_df, "test.csv": test_df, "validation.csv": val_df}
    ml_client = get_ml_client()
    registered = register_dataframes_as_asset(
        ml_client,
        dataframes,
        name=args.output_asset_name,
        description="Cleaned data for modelling purposes",
    )
    # Write version to output file for the orchestrator to read
    output_path = Path(args.output_version_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(str(registered.version))
    logger.info(f"Wrote asset version {registered.version} to {output_path}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
