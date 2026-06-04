import torch
import pandas as pd
from torch.utils.data import Dataset


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
