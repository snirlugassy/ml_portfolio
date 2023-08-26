import torch
import numpy as np
import pandas as pd

from classic import find_mvp_portfolio, find_tangent_portfolio
from mlp.network import RRNet

NETWORK_WEIGHTS = "/home/snirlugassy/ml_portfolio/experiments/ea3bb3b9/model_last.pth"
NETWORK_DEPTH = 5
NETWORK_DIM = 493

class Portfolio:
    def __init__(self, net: torch.nn.Module):
        # self.net = RRNet(dim=NETWORK_DIM, depth=NETWORK_DEPTH)
        # self.net.load_state_dict(torch.load(weights))
        # self.net = self.net.eval()

        self.net = net.eval()


    def train(self, train_data:pd.DataFrame):
        pass

    def get_portfolio(self, train_data:pd.DataFrame):
        train_data = train_data['Adj Close'] # market close value
        all_stocks = list(train_data.columns)
        train_data = train_data.fillna(method='ffill') # attempt to fill nan values
        train_data = train_data.dropna(axis=1) # remove stocks with nan values that couldn't be filled
        stocks = list(train_data.columns)
        
        returns = train_data.pct_change(1, fill_method="ffill")[1:]
        R = returns.iloc[-1].values
        R = torch.from_numpy(R).float().unsqueeze(0)

        with torch.no_grad():
            portfolio = self.net(R).squeeze(0)

        w = torch.zeros(len(all_stocks))
        for v,s in zip(portfolio, stocks):
            i = all_stocks.index(s)
            w[i] = v
        
        # return even weights portfolio
        # return np.ones(len(all_stocks)) / len(all_stocks)

        return np.array(w)
