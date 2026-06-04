import torch
import pandas as pd
from torch.utils.data import DataLoader, Dataset


class SleepDataset(Dataset):
    """
    Torch dataset for sleep data classification
    """

    def __init__(self, df: pd.DataFrame, target: pd.Series) -> None:
        self.X = torch.tensor(df.values, dtype=torch.float32)
        self.y = torch.tensor(target.values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.X[idx], self.y[idx]


def make_dataloaders(
    train_df: pd.DataFrame,
    train_target: pd.Series,
    val_df: pd.DataFrame,
    val_target: pd.Series,
    test_df: pd.DataFrame,
    test_target: pd.DataFrame,
    batch_size: int = 64,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Helper function to create data loaders for all datasets
    """
    kwargs = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    train_loadder = DataLoader(
        SleepDataset(train_df, train_target), shuffle=True, **kwargs
    )
    val_loader = DataLoader(SleepDataset(val_df, val_target), shuffle=True, **kwargs)
    test_loader = DataLoader(SleepDataset(test_df, test_target), shuffle=True, **kwargs)

    return train_loadder, val_loader, test_loader
