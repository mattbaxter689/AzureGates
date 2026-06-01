import argparse
import logging

from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Drift Detection Gate")
    p.add_argument(
        "--training-data",
        type=str,
        required=True,
        help="Mounted input path for gold star data",
    )
    p.add_argument(
        "--drift-detected",
        type=str,
        required=True,
        help="Mounted input path for new data to check",
    )
    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    drift_output_path = Path(args.drift_detected)
    drift_output = drift_output_path.read_text()
    logger.info(f"Drift output from drift gate: {drift_output}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
