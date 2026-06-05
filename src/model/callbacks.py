import logging
import mlflow
import lightning.pytorch as pl
from lightning.pytorch.callbacks import (
    Callback,
    ModelCheckpoint,
    EarlyStopping,
    LearningRateMonitor,
)

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


def make_lr_monitor() -> LearningRateMonitor:
    return LearningRateMonitor(logging_interval="epoch")


class MlflowArtifactCallback(Callback):
    """
    At the end of training the final model, automatically log the best checkpoint as the mlflow
    artifact. Requires an active mlflow run
    """

    def __init__(self, checkpoint_callback: ModelCheckpoint) -> None:
        super().__init__()
        self._ckpt_cb = checkpoint_callback

    # TODO: create a special Pyfunc to load my scalers and model
    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        best_path = self._ckpt_cb.best_model_path
        if best_path and mlflow.active_run():
            logger.info(f"Logged best checkpoint to mlflow: {best_path}")
        elif not mlflow.active_run():
            logger.warn("No active mlflow run - checkpoint not logged to mlflow")
