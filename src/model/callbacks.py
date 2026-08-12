import logging
import os

import lightning.pytorch as pl
import mlflow
from lightning.pytorch.callbacks import (
    Callback,
    EarlyStopping,
    ModelCheckpoint,
)

from model.pyfunc_wrapper import SleepRiskPredictor

logger = logging.getLogger(__name__)


def make_early_stopping(patience: int = 10, monitor: str = "val_f1") -> EarlyStopping:
    """Stop training when val_f1 does not improve for `patience` epochs."""
    return EarlyStopping(
        monitor=monitor,
        mode="max",
        patience=patience,
        verbose=True,
        min_delta=1e-4,
    )


def make_checkpoint(
    dirpath: str,
    monitor: str = "val_f1",
    filename: str = "best-{epoch:02d}-{val_f1:.4f}",
    save_top_k: int = 1,
) -> ModelCheckpoint:
    """
    Save the top-k checkpoints by `monitor` metric.
    The `best_model_path` attribute gives the path to the best checkpoint.
    """
    return ModelCheckpoint(
        dirpath=dirpath,
        filename=filename,
        monitor=monitor,
        mode="max",
        save_top_k=save_top_k,
        save_last=True,
        verbose=True,
    )


class MlflowArtifactCallback(Callback):
    """
    At the end of training the final model, automatically log the
    best checkpoint as well a scaler and label encoder as a unified
    pyfunc model
    """

    def __init__(
        self,
        checkpoint_callback: ModelCheckpoint,
        scaler_path: str,
        label_encoder_path: str,
    ) -> None:
        super().__init__()
        self._ckpt_cb = checkpoint_callback
        self._scaler_path = scaler_path
        self._label_encoder_path = label_encoder_path

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:

        best_ckpt_path = self._ckpt_cb.best_model_path

        if not mlflow.active_run():
            logger.warning("No active mlflow run - model not logged to mlflow")
            return

        if best_ckpt_path:
            logger.info("Packaging model and scaler into custom PyFunc model")

            # define the artifacts to map to mlflow
            artifacts = {
                "checkpoint": best_ckpt_path,
                "scaler": self._scaler_path,
                "label_encoder": self._label_encoder_path,
            }

            mlflow.pyfunc.log_model(
                artifact_path="model",
                python_model=SleepRiskPredictor(),
                artifacts=artifacts,
                code_path=[os.path.join(os.path.dirname(__file__), "classifier.py")],
            )

            logger.info("Successfully logged unified PyFunc asset to MlFlow")
        else:
            logger.error(
                "ModelCheckpoint callback did not return a valid best path. PyFunc logging aborted"
            )
