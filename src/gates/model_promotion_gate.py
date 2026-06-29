import argparse
import logging
import mlflow
import mlflow.pyfunc
import time
from sklearn.metrics import f1_score
from mlflow.client import MlflowClient

from utils.logging_utils import setup_logging
from data.preprocessing import TARGET_COL, load_data, load_final_run_output
from utils.mlflow_utils import validate_and_promote_challenger

# from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Model Promotion Gate")
    p.add_argument(
        "--final-run-id",
        type=str,
        required=True,
        help="Run id from final model training round",
    )
    p.add_argument(
        "--processed-data",
        type=str,
        required=True,
        help="Processed data from data version step",
    )

    return p.parse_args()


def run(args: argparse.Namespace) -> None:

    setup_logging()
    client = MlflowClient()

    model_name = "sleep_disorder_classifier"

    run_id = load_final_run_output(args.final_run_id)
    logger.info(f"Loaded run id from final model training: {run_id}")

    challenger_version = validate_and_promote_challenger(run_id, model_name)

    logger.info("Loading test data")
    _, _, test = load_data(args.processed_data)
    X_test = test.drop(columns=[TARGET_COL])
    y_true = test[TARGET_COL]

    challenger_model = mlflow.pyfunc.load_model(f"models:/{model_name}/Staging")

    start_time = time.time()
    challenger_preds = challenger_model.predict(X_test)
    challenger_latency = (time.time() - start_time) / len(X_test)
    challenger_f1 = f1_score(y_true, challenger_preds, average="macro")

    try:
        champion_model = mlflow.pyfunc.load_model(f"models:/{model_name}/Production")
        champion_preds = champion_model.predict(X_test)
        champion_f1 = f1_score(y_true, champion_preds, average="macro")
        is_first_deployment = False
    except Exception:
        logger.info("No active champion found. Initializing first deployment track.")
        champion_f1 = 0.0
        is_first_deployment = True

    if challenger_latency > 0.015:
        logger.info(
            f"Promotion Denied: Latency too high ({challenger_latency*1000:.2f}ms)."
        )
        return None

    if is_first_deployment or (challenger_f1 > champion_f1):
        logger.info("Promoting Challenger model to Champion")

        client.transition_model_version_stage(
            name=model_name,
            version=challenger_version,
            stage="Production",
            archive_existing_versions=True,
        )

        client.set_model_version_tag(
            name=model_name,
            version=challenger_version,
            key="deployment_role",
            value="champion",
        )
        return None
    else:
        logger.info("Challenger did not beat current Champion.")
        return None


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
