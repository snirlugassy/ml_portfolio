import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class StockReturnsDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, window_size=5) -> None:
        super().__init__()
        # preprocess dataset
        business_days_index = pd.date_range(start=dataset.index.min(), end=dataset.index.max(), freq='B')
        df = dataset['Adj Close'].reindex(business_days_index, fill_value=np.nan)
        self.returns = df.pct_change(1, fill_method="ffill")   # TODO: maybe .dropna()
        self.window_size = window_size
        self.window_index = []
        for w in self.returns.rolling(window_size):
            if len(w) < window_size:
                continue
            self.window_index.append(w.index)

    def __len__(self):
        return len(self.window_index)

    def __getitem__(self, index):
        window = self.df.loc[self.window_index[index]]
        window = torch.from_numpy(window.values).float()
        return window[:-1], window[-1]
