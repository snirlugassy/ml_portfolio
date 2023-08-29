import json
import os
import pickle
import sys
from glob import glob

import numpy as np
import pandas as pd
import torch
from sklearn import linear_model


def predict_next_day_price(price_vec, next_day_factor=1.5):
    w = len(price_vec)
    t = np.linspace(0,1,w)

    # LARS
    inp = t.reshape(-1,1)
    # model = linear_model.Lars()
    # model = linear_model.BayesianRidge()
    # model = linear_model.ElasticNet()
    model = linear_model.Lasso()
    model.fit(inp, np.array(price_vec))

    next_day_t = np.array([[next_day_factor]])
    y_pred_next_day = model.predict(next_day_t)

    return y_pred_next_day

def predict_next_day_market(df, window_size, next_day_factor=1.5):
    V = df[-window_size:]

    stocks = V.columns
    next_day_market = np.zeros(len(stocks))

    for i, stock in enumerate(stocks):
        v = V[stock].values
        p = predict_next_day_price(v, next_day_factor=next_day_factor)
        next_day_market[i] = p

    return next_day_market


class Portfolio:
    def __init__(self):
        pass

    def train(self, train_data:pd.DataFrame):
        pass

    def get_portfolio(self, train_data:pd.DataFrame):
        df = train_data['Adj Close']
        market = predict_next_day_market(df, window_size=14, next_day_factor=2)
        last_day = df.iloc[-1].to_numpy()
        r = (market / last_day) - 1
        
        th = 0.2
        r_pos = 1 * (r > th)
        r_neg = 1 * (r < -th)
        r = r_pos + r_neg

        if sum(r) == 0:
            # even weights
            return np.ones(len(r)) / len(r)
        
        return r / r.sum()


START_DATE = '2018-07-01'
END_TRAIN_DATE = '2023-05-31'
END_TEST_DATE = '2023-06-30'

def get_data():
    with open("dataset.pkl", "rb") as f:
        data = pickle.load(f)
    return data


def test_portfolio():
    full_train = get_data()
    returns = []
    strategy = Portfolio()
    for test_date in pd.date_range(END_TRAIN_DATE, END_TEST_DATE):
        if test_date not in full_train.index:
            continue
        train = full_train[full_train.index < test_date]
        cur_portfolio = strategy.get_portfolio(train)
        if not np.isclose(cur_portfolio.sum(), 1):
            raise ValueError(f'The sum of the portfolio should be 1, not {cur_portfolio.sum()}')
        test_data = full_train['Adj Close'].loc[test_date].to_numpy()
        prev_test_data = train['Adj Close'].iloc[-1].to_numpy()
        test_data = test_data / prev_test_data - 1
        cur_return = cur_portfolio @ test_data
        returns.append({'date': test_date, 'return': cur_return})
    returns = pd.DataFrame(returns).set_index('date')
    mean_return, std_returns = float(returns.mean()), float(returns.std())
    sharpe = mean_return / std_returns
    print("Sharp Ratio: ", sharpe)

    # portfolio variance
    cov_matrix = full_train['Adj Close'].pct_change().cov()
    port_variance = np.dot(cur_portfolio.T, np.dot(cov_matrix, cur_portfolio))
    print("Portfolio Variance: ", port_variance)

if __name__ == '__main__':
    test_portfolio()
