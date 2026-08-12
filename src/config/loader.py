import os
import re
from pathlib import Path

import yaml

from config.config_models import OrchestratorConfig

ORCHESTRATOR_ROOT = Path(__file__).parent.parent.parent
_VAR_PATTERN = re.compile(r"\$\{([A-Z0-9_]+)\}")


def _substitute_env(obj):
    if isinstance(obj, dict):
        return {k: _substitute_env(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_substitute_env(v) for v in obj]
    if isinstance(obj, str):

        def replace(match):
            var_name = match.group(1)
            try:
                return os.environ[var_name]
            except KeyError:
                raise RuntimeError(f"Missing required environment variable: {var_name}")

        return _VAR_PATTERN.sub(replace, obj)
    return obj


def load_config(path: str) -> OrchestratorConfig:
    raw = yaml.safe_load((ORCHESTRATOR_ROOT / path).read_text())
    substituted = _substitute_env(raw)
    return OrchestratorConfig.model_validate(substituted)
