import argparse
import logging


logger = logging.getLogger(__name__)

CHAMPION_DEPLOYMENT = "champion"
CHALLENGER_DEPLOYMENT = "challenger"


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
