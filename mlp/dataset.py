import random

import torch
import pandas as pd

from torch.utils.data import Dataset


RANDOM_NOISE_PROB = 0.5
RANDOM_DELETE_PROB = 0.1
RANDOM_DELETE_PERCENT = 0.02
RANDOM_NEGATIVE_MARKET_PROB = 0.0


class SingleReturnsDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, apply_augment=True, pct_change=1) -> None:
        super().__init__()
        # preprocess dataset
        self.dataset = dataset['Adj Close'].fillna(method="ffill")
        self.returns = self.dataset.pct_change(pct_change).fillna(0.0)[pct_change:]
        print("number of nan", self.returns.isna().sum().sum())
        self.apply_augment = apply_augment
        
    def __len__(self):
        return len(self.returns) - 1

    def __getitem__(self, index):
        i, j = self.returns.index[index], self.returns.index[index + 1]
        x = self.returns.loc[i].values
        y = self.returns.loc[j].values

        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()

        if self.apply_augment:
            # add random noise
            if random.random() < RANDOM_NOISE_PROB:
                x = x + (torch.rand_like(x) - 0.5) / 1000

            # random zero stock return
            if random.random() < RANDOM_DELETE_PROB:
                m = (torch.rand_like(x) < 1 - RANDOM_DELETE_PERCENT)
                x = x * m
                # y = y * m # this is optional

            # random negative market
            if random.random() < RANDOM_NEGATIVE_MARKET_PROB:
                x = -x
                y = -y

        return x, y
