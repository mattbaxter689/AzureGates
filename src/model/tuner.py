import os
import mlflow
import logging
import pandas as pd
import lightning.pytorch as pl
from optuna.integration import PyTorchLightningPruningCallback
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers.mlflow import MLFlowLogger
import optuna
from typing import Callable

from model.classifier import SleepClassifier
from data.dataset import make_dataloaders

PARENT_RUN_ID = os.getenv("MLFLOW_RUN_ID")
TRACKING_URI = mlflow.get_tracking_uri()

mlflow.pytorch.autolog(disable=True)
mlflow.autolog(disable=True)

logger = logging.getLogger(__name__)


def make_objective(
    train_df: pd.DataFrame,
    train_target: pd.Series,
    val_df: pd.DataFrame,
    val_target: pd.Series,
    num_classes: int,
    max_epochs: int = 20,
) -> Callable:
    """
    Function that helps create the objective study for optuna
    """

    def objective(trial: optuna.Trial) -> float:
        mlflow.set_tracking_uri(TRACKING_URI)

        with mlflow.start_run(run_id=PARENT_RUN_ID) as parent_run:
            current_experiment_id = parent_run.info.experiment_id

            with mlflow.start_run(
                run_name=f"trial_{trial.number}",
                experiment_id=current_experiment_id,
                nested=True,
            ) as child_run:

                # sample the parameters
                hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
                batch_size = trial.suggest_categorical("batch_size", [128, 256, 512])
                dropout = trial.suggest_float("dropout", 0.1, 0.5)
                lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
                weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

                mlflow.log_params(
                    {
                        "hidden_dim": hidden_dim,
                        "batch_size": batch_size,
                        "dropout": dropout,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "trial_number": trial.number,
                    }
                )
                mlflow.set_tag("mlflow.runName", f"trial_{trial.number}")

                train_loader, val_loader, _ = make_dataloaders(
                    train_df,
                    train_target,
                    val_df,
                    val_target,
                    val_df,
                    val_target,
                    batch_size=batch_size,
                    num_workers=3,
                )

                model = SleepClassifier(
                    input_dim=len(train_df.columns),
                    num_classes=num_classes,
                    hidden_dim=hidden_dim,
                    lr=lr,
                    dropout=dropout,
                    weight_decay=weight_decay,
                )

                pruning_cb = PyTorchLightningPruningCallback(trial, monitor="val_f1")
                early_stop = EarlyStopping(monitor="val_f1", mode="max", patience=3)
                mlflow_logger = MLFlowLogger(
                    tracking_uri=TRACKING_URI, run_id=child_run.info.run_id
                )

                trainer = pl.Trainer(
                    max_epochs=max_epochs,
                    accelerator="auto",
                    devices=1,
                    enable_progress_bar=False,
                    enable_model_summary=False,
                    logger=mlflow_logger,
                    callbacks=[pruning_cb, early_stop],
                    enable_checkpointing=False,
                )

                try:
                    trainer.fit(model, train_loader, val_loader)
                except optuna.exceptions.TrialPruned:
                    logger.info(
                        f"Trial number {trial.number} pruned for early stopping"
                    )
                    raise

                val_f1 = trainer.callback_metrics.get("val_f1", 0.0)
                return float(val_f1)

    return objective


def run_tuning(
    train_df: pd.DataFrame,
    train_target: pd.Series,
    val_df: pd.DataFrame,
    val_target: pd.Series,
    num_classes: int,
    max_epochs: int = 25,
    n_trials: int = 10,
) -> optuna.Study:
    """
    Function to run the hyperparameter tuning on optuna
    """

    study = optuna.create_study(
        direction="maximize",
        study_name="sleep-classification-tuner",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=1),
        sampler=optuna.samplers.TPESampler(seed=42),
    )

    objective = make_objective(
        train_df, train_target, val_df, val_target, num_classes, max_epochs
    )

    logger.info(f"Starting Optuna tuning: {n_trials}")
    study.optimize(objective, n_trials, show_progress_bar=False)

    logger.info(
        f"Tuning complete. Best trial {study.best_trial.number}  val_f1={study.best_value:.4f}"
    )

    return study
