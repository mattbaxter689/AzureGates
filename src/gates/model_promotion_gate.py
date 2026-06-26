import argparse
import logging

from utils.logging_utils import setup_logging
from data.preprocessing import load_final_run_output

# from pathlib import Path
# import mlflow
# from mlflow.tracking import MlflowClient

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model Promotion Gate")
    p.add_argument(
        "--final-run-id",
        type=str,
        required=True,
        help="Run id from final model training round",
    )

    return p.parse_args()


def run(args: argparse.Namespace) -> None:

    setup_logging()

    run_id = load_final_run_output(args.final_run_id)
    logger.info(f"Loaded run id from final model training: {run_id}")


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
