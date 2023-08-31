import os
import argparse
import random
from uuid import uuid4

import torch
from torch.utils.data import Dataset, DataLoader

import numpy as np
import pandas as pd
from tqdm import tqdm

STOCKS_TICKERS_FILE = "stocks.txt"
NETWORK_WEIGHTS_FILE = "model_best_10a0ad56.pth"

NETWORK_CONFIG = {
  "output_dir": "./mlp_v1/",
  "input_dim": 503,
  "epochs": 200,
  "seed": 56,
  "depth": 5,
  "batch_size": 256,
  "pct_change": 1,
  "dropout": 0.0,
  "learning_rate": 0.0001,
  "normalization_factor": 0.0,
  "normalization_order": 1,
  "step_lr_gamma": 0.9,
  "step_lr_every": 15,
  "use_batch_norm": True,
  "use_skip_connection": True,
  "apply_augments": True,
  "device": "cuda:0"
}


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


class RRNet(torch.nn.Module):
    def __init__(self, dim, depth, dropout=0.3, batch_norm=True, skip_connection=True) -> None:
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.activation = torch.nn.ELU
        self.dropout = dropout
        self.use_batch_norm = batch_norm
        self.use_skip_connection = skip_connection

        _mlps = []
        for _ in range(depth):
            _mlps.append(torch.nn.Linear(dim, dim))
            _mlps.append(self.activation())
            if self.use_batch_norm:
                _mlps.append(torch.nn.BatchNorm1d(self.dim))
            _mlps.append(torch.nn.Dropout(dropout))
        _mlps.append(torch.nn.Linear(dim, dim)) # last layer without activation
        self.mlps = torch.nn.Sequential(*_mlps)

    def forward(self, x):
        y = self.mlps(x)

        if self.use_skip_connection:
            y = x + y

        y = y / y.sum(dim=1).unsqueeze(1)

        return y


def num_params(module):
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def test_model(model, loader, device):
    """
    using the network predicted returns to evaluate the model
    """
    S = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            w = model(x)
            r = w * y
            s = torch.std(r, dim=-1) / torch.mean(r, dim=-1)
            S.append(s)
    S = torch.concat(S)
    return torch.mean(S).item()


class Portfolio:
    def __init__(self, weights=np.NaN):
        torch.manual_seed(NETWORK_CONFIG["seed"])
        self.net = RRNet(
            dim=NETWORK_CONFIG['input_dim'],
            depth=NETWORK_CONFIG['depth'],
            dropout=NETWORK_CONFIG['dropout'],
            batch_norm=NETWORK_CONFIG["use_batch_norm"],
            skip_connection=NETWORK_CONFIG['use_skip_connection']
        )
        self.net.load_state_dict(torch.load(NETWORK_WEIGHTS_FILE))
        self.net = self.net.eval()

        with open(STOCKS_TICKERS_FILE, "r") as f:
            self.stocks = f.readlines()
            self.stocks = [s.strip() for s in self.stocks]

    def train(self, train_data:pd.DataFrame):
        config = argparse.Namespace(**NETWORK_CONFIG)
 
        torch.manual_seed(config.seed)
        random.seed(config.seed)

        experiment_id = str(uuid4()).replace("-", "")[:8]
        print("EXPERIMENT:", experiment_id)

        experiment_dir = os.path.join(config.output_dir, experiment_id)
        os.makedirs(experiment_dir, exist_ok=False)

        train_dataset = SingleReturnsDataset(train_data, apply_augment=config.apply_augments, pct_change=config.pct_change)
        train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        print("train dataset length", len(train_dataset))

        self.net = self.net.to(config.device)
        self.net.train()
        
        print("number of parameters:", num_params(self.net))

        # OPTIMIZATION
        optimizer = torch.optim.Adam(self.net.parameters(), lr=config.learning_rate, maximize=True)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config.step_lr_every, 
            gamma=config.step_lr_gamma
        )


        # test_sharpe_ratio = []
        best_epoch_sharpe = (-1, -1) # iteration, share ratio

        nf = config.normalization_factor
        nord = config.normalization_order

        with tqdm(total=config.epochs * len(train_loader), postfix={"epoch":0, "lr": lr_scheduler.get_last_lr()}) as progress:
            for epoch in range(config.epochs):
                progress.set_postfix(epoch=(epoch + 1), lr=lr_scheduler.get_last_lr())
                for i, (X, y) in enumerate(train_loader):
                    X = X.to(config.device)
                    y = y.to(config.device)

                    # forward
                    output = self.net(X)
                    r = output * y
                    sharpe_ratio = torch.mean(torch.std(r, dim=-1) / torch.mean(r, dim=-1) + nf * torch.norm(output, p=nord, dim=-1))

                    # backward
                    optimizer.zero_grad()
                    sharpe_ratio.backward()
                    optimizer.step()

                    # print statistics
                    progress.update()

                # THIS IS MISSING SINCE WE ONLY GET TRAINING DATA AND NOT TESTING DATA
                # evaluate on test set
                # ts = test_model(self.net, test_loader, config.device)

                # if ts > best_epoch_sharpe[1]:
                #     best_epoch_sharpe = (epoch, ts)
                #     torch.save(self.net.state_dict(), os.path.join(experiment_dir, "model_best_sharpe.pth"))

                # test_sharpe_ratio.append(ts)

                lr_scheduler.step()

        print(f"Best epoch {best_epoch_sharpe[0]} with Sharpe ratio {best_epoch_sharpe[1]}")

        # save last weights to file
        torch.save(self.net.state_dict(), os.path.join(experiment_dir, "model_last.pth"))


    def get_portfolio(self, train_data:pd.DataFrame):
        if 'Adj Close' in train_data.columns:
            returns = train_data['Adj Close'].fillna(method="ffill").pct_change(1).fillna(0.0)[1:]
        else:
            returns = train_data.fillna(0.0)

        returns = returns[self.stocks]

        R = returns.iloc[-1].values
        R = torch.from_numpy(R).float().unsqueeze(0)

        with torch.no_grad():
            portfolio = self.net(R).squeeze(0)

        return np.array(portfolio)
 


class NetworkPortfolio(Portfolio):
    pass
