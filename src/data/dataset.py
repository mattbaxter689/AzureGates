import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


class SleepDataset(Dataset):
    """
    Torch dataset for sleep data classification
    """

    def __init__(self, df: pd.DataFrame, target: pd.Series) -> None:
        self.X = torch.tensor(df.to_numpy(dtype=np.float32), dtype=torch.float32)

        if hasattr(target, "to_numpy"):
            # If it's a pandas Series
            target_arr = target.to_numpy(dtype=np.int64)
        else:
            # If it's already a numpy array
            target_arr = target.astype(np.int64)

        self.y = torch.tensor(target_arr, dtype=torch.long)

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
    kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
    }
    train_loader = DataLoader(
        SleepDataset(train_df, train_target), shuffle=True, **kwargs
    )
    val_loader = DataLoader(SleepDataset(val_df, val_target), shuffle=False, **kwargs)
    test_loader = DataLoader(
        SleepDataset(test_df, test_target), shuffle=False, **kwargs
    )

    return train_loader, val_loader, test_loader
