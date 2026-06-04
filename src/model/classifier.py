import logging
import mlflow
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adam import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


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
        class_weights: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["class_weights"])

        self.input_dim = input_dim
        self.num_classes = num_classes
        self.lr = lr
        self.dropout = dropout

        self.net = nn.Sequential(
            [
                # 2 layer model for now
                nn.Linear(input_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                # final block
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.GELU(),
                nn.Dropout(dropout),
                # output layer
                nn.Linear(hidden_dim // 2, num_classes),
            ]
        )
