import pandas as pd
import numpy as np


def calc_portfolio_performance(portfolio_weights: np.array, returns: pd.DataFrame):
    """
    Calculate the expected mean and standard deviation of returns for a portfolio.
    :param weights: A numpy array with the weights of the portfolio.
    :param returns: A pandas dataframe with the returns of the stocks.
    :return: A tuple with the mean and standard deviation of portfolio returns.
    """
    mean_returns = returns.mean()
    portfolio_returns = mean_returns @ portfolio_weights
    portfolio_std = np.sqrt(portfolio_weights.T @ returns.cov() @ portfolio_weights)
    return portfolio_returns, portfolio_std
