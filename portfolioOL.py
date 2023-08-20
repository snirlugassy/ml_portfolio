import numpy as np
import pandas as pd
import pickle

from classic import find_mvp_portfolio, find_tangent_portfolio
from Portfolio1DCNN import Portfolio1DCNN

hypothesis = [
find_mvp_portfolio,
find_tangent_portfolio,
Portfolio1DCNN
]

class Portfolio:
    def __init__(self,gamma = 0.5 ,weights=np.NaN):
        self.hypothesis = [hypotheses() for hypotheses in hypothesis]
        self.weights = {hypothesis: 0 for hypothesis in self.hypothesis}
        self.gamma = gamma
        pass

    def train(self, train_data: pd.DataFrame, window_size = 31):

        for start in range(len(train_data) - window_size + 1):
            results = []

            window_data = train_data.iloc[start:start + window_size - 1]
            last_day_data = train_data.iloc[start + window_size - 1]

            # update weights
            for hypothesis in self.hypothesis:
                portfolio_weights = hypothesis.get_portfolio(window_data)
                recommendation = np.dot(portfolio_weights, last_day_data)
                results.append((hypothesis, recommendation))

            sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
            for i, (hypothesis, recommendation) in enumerate(sorted_results):
                self.weights[hypothesis] = self.weights[hypothesis] * self.gamma ** i

            # normalize weights
            total_weight = sum(self.weights.values())
            for hypothesis in self.hypothesis:
                self.weights[hypothesis] /= total_weight

        # store weights
        with open("portfolio_weights.pkl", "wb") as f:
            pickle.dump(self.weights, f)

        pass

    def get_portfolio(self, train_data: pd.DataFrame):
        # load weights
        with open("portfolio_weights.pkl", "rb") as f:
            self.weights = pickle.load(f)

        # get portfolio
        business_days_index = pd.date_range(start=train_data.index.min(), end=train_data.index.max(), freq='B')
        adj_close = train_data['Adj Close'].reindex(business_days_index, fill_value=np.nan)
        returns = adj_close.pct_change(1, fill_method="ffill")
        portfolioOL =  sum([self.weights[hypothesis] * hypothesis.get_portfolio(train_data) for hypothesis in self.hypothesis])

         # normalize weights to sum to 1
        portfolioOL /= portfolioOL.sum()
        return portfolioOL


if __name__ == '__main__':
    with open("train_dataset.pkl", "rb") as f:
        data = pickle.load(f)
    portfolio = Portfolio()
    portfolio.train(data)
    print(portfolio.get_portfolio(data))