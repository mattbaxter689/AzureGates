import gc
import logging
import os
from collections.abc import Callable
from datetime import UTC, datetime

import lightning.pytorch as pl
import mlflow
import optuna
import pandas as pd
import torch
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers.mlflow import MLFlowLogger
from optuna.integration import PyTorchLightningPruningCallback

from data.dataset import make_dataloaders
from model.callbacks import (
    MlflowArtifactCallback,
    make_checkpoint,
    make_early_stopping,
)
from model.classifier import SleepClassifier

os.environ["MLFLOW_DISABLE_LOGGED_MODELS"] = "true"

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

        # CUDA errors requires this for my GPU training
        gc.collect()
        torch.cuda.empty_cache()

        with mlflow.start_run(
            run_name=f"trial_{trial.number}",
            tags={"mlflow.parentRunId": PARENT_RUN_ID},
            nested=True,
        ) as child_run:
            params = {
                "hidden_dim": trial.suggest_categorical("hidden_dim", [64, 128, 256]),
                "batch_size": trial.suggest_categorical("batch_size", [128, 256, 512]),
                "dropout": trial.suggest_float("dropout", 0.1, 0.5),
                "lr": trial.suggest_float("lr", 1e-4, 1e-2, log=True),
                "weight_decay": trial.suggest_float(
                    "weight_decay", 1e-5, 1e-3, log=True
                ),
            }

            hidden_dim = params["hidden_dim"]
            batch_size = params["batch_size"]
            dropout = params["dropout"]
            lr = params["lr"]
            weight_decay = params["weight_decay"]

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

            try:
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

                # 4. Bind the Lightning logger strictly to this child run ID
                mlflow_logger = MLFlowLogger(
                    tracking_uri=TRACKING_URI, run_id=child_run.info.run_id
                )

                mlflow.log_params(params)

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
                mlflow.log_metric("final_val_f1", float(val_f1))

                return float(val_f1)

            finally:
                del model
                del trainer
                gc.collect()
                torch.cuda.empty_cache()

    return objective


def run_tuning(
    train_df: pd.DataFrame,
    train_target: pd.Series,
    val_df: pd.DataFrame,
    val_target: pd.Series,
    num_classes: int,
    max_epochs: int = 50,
    n_trials: int = 20,
) -> optuna.Study:
    """
    Function to run the hyperparameter tuning on optuna
    """

    study = optuna.create_study(
        direction="maximize",
        study_name="sleep-classification-tuner",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=3),
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


def final_training_run(
    train_df: pd.DataFrame,
    train_target: pd.Series,
    val_df: pd.DataFrame,
    val_target: pd.Series,
    test_df: pd.DataFrame,
    test_target: pd.Series,
    num_classes: int,
    best_params: dict[str, str | int | float],
    scaler_path: str,
    encoder_path: str,
    ckpt_dir: str = "checkpoints",
    max_epochs: int = 50,
) -> tuple[str, float]:
    """
    Train final model using best parameters from Optuna
    trials.
    """

    train_loader, val_loader, test_loader = make_dataloaders(
        train_df,
        train_target,
        val_df,
        val_target,
        test_df,
        test_target,
        best_params["batch_size"],
        num_workers=2,
    )

    model = SleepClassifier(
        input_dim=len(train_df.columns),
        num_classes=num_classes,
        hidden_dim=best_params["hidden_dim"],
        lr=best_params["lr"],
        dropout=best_params["dropout"],
        weight_decay=best_params["weight_decay"],
    )

    ckpt_cb = make_checkpoint(ckpt_dir)
    early_stopping = make_early_stopping(patience=5)
    artifact_db = MlflowArtifactCallback(
        ckpt_cb, scaler_path=str(scaler_path), label_encoder_path=str(encoder_path)
    )

    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    with mlflow.start_run(
        run_name=f"final_training-{timestamp}",
        nested=True,
        tags={"mlflow.parentRunId": PARENT_RUN_ID},
    ) as run:
        mlflow_logger = MLFlowLogger(
            tracking_uri=TRACKING_URI,
            run_id=run.info.run_id,
        )
        mlflow.log_params(best_params)

        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            callbacks=[ckpt_cb, early_stopping, artifact_db],
            logger=mlflow_logger,
            enable_progress_bar=True,
        )
        trainer.fit(model, train_loader, val_loader)
        best_val_f1 = float(trainer.callback_metrics.get("val_f1", 0.0))

        test_results = trainer.test(model, test_loader, verbose=False)
        if test_results:
            test_metrics = test_results[0]

            mlflow.log_metrics(test_metrics)
            logger.info(f"Testing complete. Metrics: {test_metrics}")

        mlflow.log_metric("best_val_f1", best_val_f1)

        run_id = run.info.run_id
    return run_id, best_val_f1
