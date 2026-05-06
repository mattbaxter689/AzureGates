import logging
import sys


def setup_logging(level: int = logging.INFO) -> None:
    """Configure a root logger with clean formatting for AML job logs"""

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=level,
        format=fmt,
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )

    # Suppress noisy Azure SDK transport logs
    for noisy in ("azure.core", "azure.identity", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
