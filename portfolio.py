import torch
import numpy as np
import pandas as pd

from classic import find_mvp_portfolio, find_tangent_portfolio
from mlp.network import RRNet


class Portfolio:
    def __init__(self, net: torch.nn.Module, stocks:list[str]):
        self.net = net.eval()
        self.stocks = stocks


    def train(self, train_data:pd.DataFrame):
        pass

    def get_portfolio(self, train_data:pd.DataFrame):
        df = train_data['Adj Close'] # market close value
        all_stocks = list(df.columns)
        df = df.fillna(method='ffill') # attempt to fill nan values
        df = df[self.stocks]

        # df = df.dropna(axis=1) # remove stocks with nan values that couldn't be filled
        # stocks = list(df.columns)

        returns = df.pct_change(1, fill_method="ffill")[1:]
        R = returns.iloc[-1].values
        R = torch.from_numpy(R).float().unsqueeze(0)

        with torch.no_grad():
            portfolio = self.net(R).squeeze(0)

        w = torch.zeros(len(all_stocks))
        for v,s in zip(portfolio, self.stocks):
            i = all_stocks.index(s)
            w[i] = v
        
        # return even weights portfolio
        # return np.ones(len(all_stocks)) / len(all_stocks)

        return np.array(w)
