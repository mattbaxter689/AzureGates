from pydantic import BaseModel, model_validator, Field
from typing import Literal, Union


class CategoricalParam(BaseModel):
    """
    Will hold the categorical data parsed from yaml
    """

    type: Literal["categorical"]
    choices: list[int | float | str]


class FloatParam(BaseModel):
    """
    Will hold float parameters and their low and high values
    parsed from yaml
    """

    type: Literal["float"]
    low: float
    high: float
    log: bool = False

    @model_validator(mode="after")
    def check_bounds(self) -> "FloatParam":
        if self.low >= self.high:
            raise ValueError(f"low ({self.low}) must be < high ({self.high})")
        if self.log and self.low <= 0:
            raise ValueError("log-scale range requires low > 0")
        return self


ParamSpec = Union[CategoricalParam, FloatParam]


class OptunaConfig(BaseModel):
    n_trials: int
    max_epochs: int
    startup_trials: int
    warmup_steps: int


class TuningConfig(BaseModel):
    hyperparams: dict[str, ParamSpec] = Field(discriminator=None)
    optuna: OptunaConfig
