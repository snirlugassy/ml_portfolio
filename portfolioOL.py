import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
import sys

from classic import mvp_portfolio
from portfolio2DCNN import portfolio2CNN

hypothesis = [
    mvp_portfolio(),
    mvp_portfolio(10),
    mvp_portfolio(20),
    mvp_portfolio(30),
    mvp_portfolio(40),
    mvp_portfolio(50),
    portfolio2CNN(),
    # portfolio2CNN()
]


class Portfolio:
    def __init__(self, weights=np.NaN, gamma=0.5):
        self.hypothesis = hypothesis
        self.weights =  {h: 1 / len(self.hypothesis) for h in self.hypothesis}
        self.gamma = self._load_gamma()
        self.prediction_history = self._load_prediction_history()
        pass

    def _load_gamma(self):
        try:
            with open("best_gamma.pkl", "rb") as f:
                self.gamma = pickle.load(f)
        except (FileNotFoundError, EOFError):
            self.gamma = 0.5

    def _load_prediction_history(self):
        try:
            with open("prediction_history.pkl", "rb") as f:
                return pickle.load(f)
        except (FileNotFoundError, EOFError):
            return {}

    def _save_prediction_history(self):
        with open("prediction_history.pkl", "wb") as f:
            pickle.dump(self.prediction_history, f)

    def train_gamma(self, train_data, gamma_values=[0.1, 0.3, 0.5, 0.7, 0.9], window_size=31):
        # Split the data into training and validation sets
        train_data, validation_data = train_test_split(train_data, test_size=0.2, shuffle=False)

        best_gamma = None
        best_performance = float('-inf')  # Starting with a very low value

        for gamma in gamma_values:
            self.gamma = gamma
            self.train(train_data, window_size=window_size)
            performance = self.validate(validation_data, window_size=window_size)

            if performance > best_performance:
                best_performance = performance
                best_gamma = gamma

        self.gamma = best_gamma
        with open("best_gamma.pkl", "wb") as f:
            pickle.dump(self.gamma, f)
        return best_gamma

    def validate(self, validation_data, window_size=31):
        total_reward = 0
        for start in range(len(validation_data) - window_size + 1):
            window_data = validation_data.iloc[start:start + window_size - 1]
            last_day_data = validation_data.iloc[start + window_size - 1]

            # Get portfolio for the window
            portfolio_weights = self.get_portfolio(window_data)
            recommendation = np.dot(portfolio_weights, last_day_data)

            total_reward += recommendation
        return total_reward / (len(validation_data) - window_size + 1)

    def train(self, train_data: pd.DataFrame, window_size=31):
        end_index = len(train_data) - window_size + 1
        print(f"current gamma: {self.gamma}")
        for start in range(end_index):
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

            sys.stdout.write('\r')
            sys.stdout.write(
                "[%-20s] %d%% - Current wights: %s" % ('=' * int(i / int(end_index / 20)), i, self.weights))
            sys.stdout.flush()

        # store weights
        with open("portfolioOL_weights.pkl", "wb") as f:
            pickle.dump(self.weights, f)

        print(f"Model weights have been saved at portfolioOL_weights.pkl")
        pass

    def _adjust_weights_based_on_outcome(self, actual_returns):
        # Adjust weights based on the difference between predicted and actual returns
        # This is a basic adjustment, and you can modify this based on your strategy
        for hypothesis in self.hypothesis:
            predicted_portfolio = hypothesis.get_portfolio(actual_returns)
            prediction_error = np.dot(predicted_portfolio - actual_returns, actual_returns)
            self.weights[hypothesis] -= self.gamma * prediction_error

        # Normalize weights after adjustment
        total_weight = sum(self.weights.values())
        for hypothesis in self.hypothesis:
            self.weights[hypothesis] /= total_weight

    def get_portfolio(self, train_data: pd.DataFrame):
        # load weights
        with open("portfolioOL_weights.pkl", "rb") as f:
            self.weights = pickle.load(f)
        # Assuming the last date in train_data is the current date
        current_date = train_data.index[-1]
        if current_date in self.prediction_history:
            # If we've already made a prediction for this date, adjust weights
            actual_returns = train_data.iloc[-1]  # Get the actual returns for the day
            self._adjust_weights_based_on_outcome(actual_returns)

        # get portfolio
        portfolioOL = sum(
            [self.weights[hypothesis] * hypothesis.get_portfolio(train_data) for hypothesis in self.hypothesis])

        # normalize weights to sum to 1
        portfolioOL /= portfolioOL.sum()

        # save prediction history
        self.prediction_history[current_date] = portfolioOL
        self._save_prediction_history()

        return portfolioOL


if __name__ == '__main__':
    with open("train_dataset.pkl", "rb") as f:
        data = pickle.load(f)
    portfolio = Portfolio()
    portfolio.train_gamma(data)
    portfolio.train(data)
    print(portfolio.get_portfolio(data))
