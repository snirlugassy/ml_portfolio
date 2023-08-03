import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from sklearn.preprocessing import MinMaxScaler
import pickle

path = "portfolio_weights.pt"

class CNN1D(nn.Module):
    def __init__(self, lookback):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, 5)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear((lookback-4)*32, 1) # (lookback - kernel_size + 1) * number_of_filters

    def forward(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class Portfolio1DCNN:
    def __init__(self, lookback=60, epochs=100, batch_size=32, lr=0.01):
        self.lookback = lookback # number of days to look back
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = lr
        self.model = CNN1D(lookback)
        self.scaler = MinMaxScaler(feature_range=(0,1))
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)
        
    def pre_process(self, data):
        scaled_data = self.scaler.fit_transform(data.values.reshape(-1,1))
        X, y = [], []
        for i in range(self.lookback, len(scaled_data)):
            X.append(scaled_data[i-self.lookback:i, 0])
            y.append(scaled_data[i, 0])
        return np.array(X), np.array(y)

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

                if (epoch+1) % 20 == 0:  # Print progress every 20 epochs
                    print(f"Epoch: {epoch+1}/{self.epochs}, Loss: {loss.item()}")

        torch.save(self.model.state_dict(), save_path)
        print(f"Model weights have been saved at {save_path}")

    def get_portfolio(self):
        self.model.load_state_dict(torch.load(path))
        self.model.eval() # set model to evaluation mode
        return self.model.state_dict()

if __name__ == '__main__':
    with open("train_dataset.pkl", "rb") as f:
        data = pickle.load(f)
    portfolio = Portfolio1DCNN()
    portfolio.pre_train(data, path)
    print(portfolio.get_portfolio())
