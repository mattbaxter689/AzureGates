import argparse
import logging

from data.preprocessing import (
    load_drift_output,
    encode_features,
    infer_schema,
    load_data,
    TARGET_COL,
)
from model.tuner import run_tuning
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

    if result == "drifted":

        train, val, test = load_data(args.training_data)
        num_cols, cat_cols = infer_schema(train)

        train_tf, train_target, transformer, label_encoder = encode_features(
            train, num_cols, cat_cols, fit=True
        )
        val_tf, val_target, _, _ = encode_features(
            val,
            num_cols,
            cat_cols,
            fit=False,
            transformer=transformer,
            label_encoder=label_encoder,
        )
        test_tf, test_target, _, _ = encode_features(
            test,
            num_cols,
            cat_cols,
            fit=False,
            transformer=transformer,
            label_encoder=label_encoder,
        )

        num_classes = train[TARGET_COL].nunique()

        study = run_tuning(
            train_tf,
            train_target,
            val_tf,
            val_target,
            num_classes=num_classes,
            max_epochs=20,
        )
        logger.info(f"Best params: {study.best_params}")


def main() -> None:
    logger.info("Parsing arguments")
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
