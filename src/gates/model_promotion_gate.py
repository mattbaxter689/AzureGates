import argparse
import logging
import mlflow
import mlflow.pyfunc
import time
import json
from pathlib import Path
from sklearn.metrics import f1_score
from mlflow.client import MlflowClient

from utils.logging_utils import setup_logging
from data.preprocessing import TARGET_COL, load_data, load_final_run_output
from utils.mlflow_utils import tag_challenger, resolve_version_by_role


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
    p.add_argument(
        "--decision-output",
        type=str,
        required=True,
        help="Output folder containing gate decision",
    )

    return p.parse_args()


def run(args: argparse.Namespace) -> None:

    setup_logging()
    client = MlflowClient()

    run_id = load_final_run_output(args.final_run_id)
    logger.info(f"Loaded run id from final model training: {run_id}")

    # register the new model and tag as challenger for now
    challenger_version = tag_challenger(client, run_id)
    logger.info(f"Registered challenger version {challenger_version}")

    logger.info("Loading test data")
    _, _, test = load_data(args.processed_data)
    X_test = test.drop(columns=[TARGET_COL])
    y_true = test[TARGET_COL]

    challenger_model = mlflow.pyfunc.load_model(
        f"models:/sleep_disorder_classifier/{challenger_version}"
    )

    start_time = time.time()
    challenger_preds = challenger_model.predict(X_test)
    challenger_latency = (time.time() - start_time) / len(X_test)
    challenger_f1 = f1_score(y_true, challenger_preds, average="macro")
    logger.info(
        f"Challenger v{challenger_version}: f1={challenger_f1:.4f}, "
        f"latency={challenger_latency * 1000:.2f}ms"
    )

    champion_mv = resolve_version_by_role(
        client, "sleep_disorder_classifier", "champion"
    )
    is_first_deployment = champion_mv is None
    champion_version = None
    champion_f1 = None

    if is_first_deployment:
        logger.info("No active champion found. Treating as first deployment")
    else:
        champion_version = champion_mv.version
        champion_model = mlflow.pyfunc.load_model(
            f"models:/sleep_disorder_classifier/{champion_version}"
        )
        champion_preds = champion_model.predict(X_test)
        champion_f1 = f1_score(y_true, champion_preds, average="macro")
        logger.info(f"Champion v{champion_version}: f1={champion_f1:.4f}")

    latency_ok = challenger_latency <= args.latency_threshold
    beats_champion = is_first_deployment or (challenger_f1 > champion_f1)

    if not latency_ok:
        decision, reason = "reject", (
            f"Latency {challenger_latency*1000:.2f}ms exceeds "
            f"{args.latency_threshold*1000:.2f}ms budget"
        )
    elif is_first_deployment:
        decision, reason = "promote_first", "No champion currently deployed"
    elif beats_champion:
        decision, reason = "shadow", (
            f"Challenger f1 {challenger_f1:.4f} beats champion f1 {champion_f1:.4f}"
        )
    else:
        decision, reason = "reject", (
            f"Challenger f1 {challenger_f1:.4f} did not beat champion f1 {champion_f1:.4f}"
        )

    logger.info(f"Gate decision: {decision} ({reason})")

    output = {
        "decision": decision,
        "reason": reason,
        "model_name": "sleep_disorder_classifier",
        "challenger_version": challenger_version,
        "champion_version": champion_version,
        "challenger_f1": challenger_f1,
        "champion_f1": champion_f1,
        "challenger_latency_ms": challenger_latency * 1000,
    }

    output_path = Path(args.decision_output)
    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "decision.json", "w") as f:
        json.dump(output, f, indent=2)


def main() -> None:
    args = parse_args()
    run(args)


if __name__ == "__main__":
    main()
