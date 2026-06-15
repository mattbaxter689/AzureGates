from typing import Any
import logging
import lightning.pytorch as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score

logger = logging.getLogger(__name__)


class SleepClassifier(pl.LightningModule):
    """
    Sleep data classifier for sleep disorder
    risk
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        dropout: float = 0.2,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lr = lr
        self.dropout = dropout
        self.weight_decay = weight_decay

        # Inpute layer
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Residual block for skip connection
        self.res_block = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Final classification output layer
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
        else:
            self.class_weights = None

        metric_kwargs = dict(num_classes=num_classes, average="macro")
        self.train_acc = MulticlassAccuracy(**metric_kwargs)
        self.val_acc = MulticlassAccuracy(**metric_kwargs)
        self.val_f1 = MulticlassF1Score(**metric_kwargs)
        self.test_acc = MulticlassAccuracy(**metric_kwargs)
        self.test_f1 = MulticlassF1Score(**metric_kwargs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h1 = self.input_layer(x)
        h2 = h1 + self.res_block(h1)

        return self.output_layer(h2)

    def _loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(
            logits, targets, weight=self.class_weights, label_smoothing=0.1
        )

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self._loss(logits, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.train_acc.update(logits, y)  # accumulate, don't log yet
        return loss

    def on_train_epoch_end(self):
        self.log("train_acc", self.train_acc.compute().item(), prog_bar=True)
        self.train_acc.reset()

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self._loss(logits, y)
        self.val_acc.update(logits, y)
        self.val_f1.update(logits, y)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        val_acc = self.val_acc.compute()
        val_f1 = self.val_f1.compute()
        self.log("val_acc", val_acc, prog_bar=True)
        self.log("val_f1", val_f1, prog_bar=True)
        self.val_acc.reset()
        self.val_f1.reset()

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        self.test_acc.update(logits, y)
        self.test_f1.update(logits, y)

    def on_test_epoch_end(self):
        self.log("test_acc", self.test_acc.compute())
        self.log("test_f1", self.test_f1.compute())
        self.test_acc.reset()
        self.test_f1.reset()

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.trainer.max_epochs)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler},
        }
