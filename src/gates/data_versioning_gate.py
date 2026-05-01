import argparse
import json
import logging
import sys
import tempfile
import uuid
from pathlib import Path
import pandas as pd

log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Data versioning gate")
    parser.add_argument(
        "--raw-data",
        type=str,
        required=True,
        help="Mounted input path to raw dataset (Azure ML uri_folder)",
    )

    parser.add_argument(
        "--output-base-uri",
        type=str,
        required=True,
        help="Base blob path for versioned dataset (e.g. azureml://datastores/...)",
    )

    parser.add_argument(
        "--data-asset-name",
        type=str,
        default="training_data_assets",
        help="Name of the Azure ML data asset to register",
    )

    parser.add_argument(
        "--output-uuid-file",
        type=str,
        required=True,
        help="Path to write generated dataset UUID (AML output file)",
    )

    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    data_path = Path(args.raw_data)
    df = pd.read_csv(data_path)
    print(df)


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
