import pandas as pd
import numpy as np


def find_mvp_portfolio(returns: pd.DataFrame):
    """
    Find the portfolio with the minimum variance.
    :param returns: A pandas dataframe with the returns of the stocks.
    :return: A numpy array with the weights of the portfolio.
    """
    cov_matrix = returns.cov()
    inv_cov_matrix = np.linalg.pinv(cov_matrix)
    e = np.ones(len(cov_matrix))
    x_mvp = inv_cov_matrix @ e / (e @ inv_cov_matrix @ e)
    # x_mvp = x_mvp * 1*(np.abs(x_mvp) > eps) # ignore very small investments
    x_mvp /= x_mvp.sum() # renormalize the weights
    return x_mvp


def find_tangent_portfolio(returns: pd.DataFrame):
    """
    Find the tangent portfolio with the maximum Sharpe ratio.
    :param returns: A pandas dataframe with the returns of the stocks.
    :return: A numpy array with the weights of the portfolio.
    """
    cov_matrix = returns.cov()
    inv_cov_matrix = pd.DataFrame(np.linalg.pinv(cov_matrix.values), cov_matrix.columns, cov_matrix.index)
    ones = np.ones(len(cov_matrix))
    mean_returns = returns.mean()
    x_tan = inv_cov_matrix @ mean_returns / (ones.T @ inv_cov_matrix @ mean_returns)
    return x_tan
