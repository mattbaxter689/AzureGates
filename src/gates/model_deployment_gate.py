import argparse
from collections.abc import Callable
import json
import logging
from pathlib import Path

from mlflow.client import MlflowClient
from utils.logging_utils import setup_logging
from utils.asset_utils import get_ml_client
from deploy.deploy import Decision, handle_promote_first, handle_reject, handle_shadow


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Executes the promotion / shadow-deployment decision"
    )
    p.add_argument(
        "--decision-input",
        type=str,
        required=True,
        help="Folder containing decision from promotion gate",
    )
    p.add_argument("--mirror-pct", type=int, default=10)
    p.add_argument("--instance-type", type=str, default="Standard_DS3_v2")

    return p.parse_args()


def run(args: argparse.Namespace) -> None:
    setup_logging()
    with open(Path(args.decision_input / "decision.json")) as f:
        decision: Decision = json.load(f)

    mlflow_client = MlflowClient()
    ml_client = get_ml_client()

    dispatch: dict[str, Callable[[], None]] = {
        "promote_first": lambda: handle_promote_first(
            ml_client, mlflow_client, decision, args
        ),
        "shadow": lambda: handle_shadow(ml_client, mlflow_client, decision, args),
        "reject": lambda: handle_reject(ml_client, mlflow_client, decision),
    }

    action = dispatch.get(decision["decision"])
    if action is None:
        raise ValueError(f"Unknown decision: {decision['decision']}")
    action()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
