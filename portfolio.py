import torch
import numpy as np
import pandas as pd


class Portfolio:
    def __init__(self, net: torch.nn.Module, stocks:list[str], seed=42):
        torch.manual_seed(seed)
        self.net = net.eval()
        self.stocks = stocks

    def train(self, train_data:pd.DataFrame):
        pass

    def get_portfolio(self, train_data:pd.DataFrame):
        df = train_data['Adj Close'] # market close value
        all_stocks = list(df.columns)
        df = df.fillna(method='ffill') # attempt to fill nan values
        df = df[self.stocks]

        returns = df.pct_change(1)[1:]
        R = returns.iloc[-1].values
        R = torch.from_numpy(R).float().unsqueeze(0)

        with torch.no_grad():
            portfolio = self.net(R).squeeze(0)

        # return the portfolio to the original size
        w = torch.zeros(len(all_stocks))
        for v,s in zip(portfolio, self.stocks):
            i = all_stocks.index(s)
            w[i] = v
        
        return np.array(w)


class NetworkPortfolio(Portfolio):
    pass