import argparse
import logging
import tempfile
from pathlib import Path

import joblib

from data.preprocessing import (
    load_drift_output,
    encode_features,
    infer_schema,
    load_data,
    TARGET_COL,
)
from model.tuner import run_tuning, final_training_run
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
    p.add_argument(
        "--final-run", type=str, required=True, help="The final run id for model"
    )
    return p.parse_args()


def run(args: argparse.Namespace) -> str:

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

        # Save encoder and scaler to load with pyfunc model
        temp_dir = tempfile.gettempdir()
        scaler_path = Path(temp_dir) / "scaler.joblib"
        encoder_path = Path(temp_dir) / "label_encoder.joblib"

        joblib.dump(transformer, scaler_path)
        joblib.dump(label_encoder, encoder_path)
        logger.info(f"Persisted transformer and scaler to temp location: {temp_dir}")

        num_classes = train[TARGET_COL].nunique()

        study = run_tuning(
            train_tf,
            train_target,
            val_tf,
            val_target,
            num_classes=num_classes,
            n_trials=5,
        )
        logger.info(f"Best params: {study.best_params}")
        logger.info(f"Best params F1-score: {study.best_value}")

        logger.info("Starting final training run with best parameters")
        best_params = study.best_params
        final_run_id, best_val_f1 = final_training_run(
            train_tf,
            train_target,
            val_tf,
            val_target,
            test_tf,
            test_target,
            num_classes=num_classes,
            best_params=best_params,
            scaler_path=scaler_path,
            encoder_path=encoder_path,
        )

        logger.info(f"Final training complete. Best F1-score: {best_val_f1}")

        output_path = Path(args.final_run)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "run_id.txt").write_text(str(final_run_id))
        logger.info(f"Wrote run id: {final_run_id}")


def main() -> None:
    logger.info("Parsing arguments")
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
