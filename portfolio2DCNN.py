import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler
import pickle

path = "CNNportfolio_weights.pt"
num_stocks = 12


class CNN2D(nn.Module):
    def __init__(self, lookback=30):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=num_stocks, kernel_size=(lookback, num_stocks),
                               stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_stocks, num_stocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class MultiStockPredictorCNN:
    def __init__(self, lookback=30, lr=0.01, epochs=100):
        self.lookback = lookback  # number of days to look back
        self.epochs = epochs
        self.lr = lr
        self.model = CNN2D(lookback)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

    def train_wights(self, training_stock_df):
        for epoch in range(self.epochs):
            total_loss = 0
            training_days = training_stock_df.shape[0]
            for i in range(0, training_days - self.lookback):
                self.optimizer.zero_grad()
                training_stock_df.fillna(0, inplace=True)
                training_stock = training_stock_df.values
                input_window = training_stock[i:i + self.lookback]
                ground_truth = training_stock[i + self.lookback]

                input_window = torch.tensor(input_window, dtype=torch.float).unsqueeze(0).unsqueeze(1)
                ground_truth = torch.tensor(ground_truth, dtype=torch.float)  # shape: (num_stocks,)

                output = self.model(input_window).squeeze()  # Remove batch and channel dimensions
                loss = self.criterion(ground_truth, output)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 20 == 0:  # Print progress every 20 epochs
                print(f"Epoch: {epoch + 1}/{self.epochs}, Loss: {total_loss}")
        torch.save(self.model.state_dict(), path)
        print(f"Model weights have been saved at {path}")

    def get_portfolio(self, recent_data: pd.DataFrame):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()  # set model to evaluation mode
        return self.model(recent_data.iloc[-31:, -num_stocks:])



class Portfolio1DCNN:
    def __init__(self, lookback=60, epochs=100, batch_size=32, lr=0.01):
        self.lookback = lookback  # number of days to look back
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        # self.model = CNN1D(lookback)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

    def pre_train(self, stock_df, save_path):
        self.model.train()

        for idx, column in enumerate(stock_df.columns):
            print(f"Training on stock: {column}")
            stock_prices = stock_df[column].dropna()
            X, y = self.pre_process(stock_prices)
            X = torch.tensor(X, dtype=torch.float).view(-1, 1, self.lookback)  # reshape into (batch_size, 1, lookback)
            y = torch.tensor(y, dtype=torch.float).view(-1, 1)  # reshape into (batch_size, 1)

            for epoch in range(self.epochs):
                self.optimizer.zero_grad()
                output = self.model(X)
                loss = self.criterion(output, y)
                loss.backward()
                self.optimizer.step()

                if (epoch + 1) % 20 == 0:  # Print progress every 20 epochs
                    print(f"Epoch: {epoch + 1}/{self.epochs}, Loss: {loss.item()}")

        torch.save(self.model.state_dict(), save_path)
        print(f"Model weights have been saved at {save_path}")

    def get_portfolio(self, recent_data: pd.DataFrame):
        self.model.load_state_dict(torch.load(path))
        self.model.eval()  # set model to evaluation mode

        weights = []
        for column in recent_data.columns:
            # Preprocess the most recent data for each stock
            last_values = recent_data[column].iloc[-self.lookback:]
            scaled_data = self.scaler.transform(last_values.values.reshape(-1, 1))
            X_test = torch.tensor(scaled_data, dtype=torch.float).view(1, 1,
                                                                       self.lookback)  # reshape into (1, 1, lookback)

            # Use the model to predict on this data
            with torch.no_grad():
                prediction = self.model(X_test)

            # Convert the model's prediction to a portfolio recommendation
            weights.append(prediction.item())

        # Normalize the weights to sum up to 1
        total_weight = sum(weights)
        normalized_weights = [weight / total_weight for weight in weights]

        # Return a dictionary with stock names as keys and their respective weights as values
        portfolio = dict(zip(recent_data.columns, normalized_weights))

        return portfolio


if __name__ == '__main__':
    with open("train_dataset.pkl", "rb") as f:
        data = pickle.load(f)

    adj_close = data['Adj Close']
    adj_close = adj_close.iloc[-31:, -num_stocks:]
    returns = adj_close.pct_change(1, fill_method="ffill")
    print(data)

    portfolio = MultiStockPredictorCNN()
    portfolio.train_wights(returns)
    # print(portfolio.get_portfolio())
