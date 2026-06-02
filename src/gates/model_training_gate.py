import argparse
import logging

from data.preprocessing import load_drift_output
from utils.logging_utils import setup_logging

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

    setup_logging()

    result = load_drift_output(args.drift_detected)
    logger.info(f"Drift output from drift gate: {result}")


def main() -> None:
    logger.info("Parsing arguments")
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
