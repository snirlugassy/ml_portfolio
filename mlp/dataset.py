import torch
import pandas as pd

from torch.utils.data import Dataset


class SingleReturnsDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, apply_augment=True) -> None:
        super().__init__()
        # preprocess dataset
        self.returns = dataset.pct_change(1, fill_method="ffill")[1:]
        self.apply_augment = apply_augment
        # self.bernoulli_dist = torch.distributions.Bernoulli(0.95)
        
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
            x = x + (torch.rand_like(x) - 0.5) / 1000

            # random zero stock return
            m = (torch.rand_like(x) < 0.98)
            x = x * m
        return x, y
