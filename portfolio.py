import numpy as np
import pandas as pd

from classic import find_mvp_portfolio, find_tangent_portfolio


class Portfolio:
    def __init__(self, weights=np.NaN):
        pass
    
    def train(self, train_data:pd.DataFrame):
        pass

    def get_portfolio(self, train_data:pd.DataFrame):
        business_days_index = pd.date_range(start=train_data.index.min(), end=train_data.index.max(), freq='B')
        adj_close = train_data['Adj Close'].reindex(business_days_index, fill_value=np.nan)
        returns = adj_close.pct_change(1, fill_method="ffill")   # TODO: maybe .dropna()
        x_mvp = find_mvp_portfolio(returns)
        return x_mvp

        # return even weights portfolio
        # return np.ones(len(train_data['Adj Close'].columns)) / len(train_data['Adj Close'].columns)
