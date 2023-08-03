import pandas as pd
import yfinance as yf
import numpy as np
from portfolio import Portfolio
import pickle


# START_DATE = '2017-08-01'
# END_TEST_DATE = '2022-09-30'
# END_TRAIN_DATE = '2022-08-31'

START_DATE = '2018-07-01'
END_TRAIN_DATE = '2023-05-31'
END_TEST_DATE = '2023-06-30'

def get_data():
    with open("train_dataset.pkl", "rb") as f:
        data = pickle.load(f)
    return data


def test_portfolio( portfolio):
    print("Loading data...")
    full_train = get_data()
    returns = []
    strategy = portfolio()
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
    test_portfolio(Portfolio)
