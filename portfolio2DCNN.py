import os
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import tqdm
from sklearn.preprocessing import MinMaxScaler
from torch.optim import Adam

## Config
num_stocks = 503
output_dir = "./experiments/cnn_v1/"
os.makedirs(output_dir, exist_ok=True)

def build_checkpoint_path(suffix):
    return os.path.join(output_dir, f"CNNportfolio_weights_{suffix}.pth")

seed = 42
torch.manual_seed(seed) # reproducability


## Network
class CNN2D(nn.Module):
    def __init__(self, lookback=30):
        super(CNN2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10*num_stocks, kernel_size=(lookback, num_stocks),
                               stride=1, padding=0)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(10* num_stocks, num_stocks)

    def forward(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x


class portfolio2CNN:
    def __init__(self, lookback=30, lr=0.0001, epochs=100, device="cuda:0", checkpoint_path=None):
        self.device = device

        self.lookback = lookback  # number of days to look back
        self.epochs = epochs
        self.lr = lr
        self.model = CNN2D(lookback)
        self.model = self.model.to(self.device)

        if checkpoint_path is not None:
            self.model.load_state_dict(torch.load(checkpoint_path))

        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.criterion = nn.MSELoss()
        self.optimizer = Adam(self.model.parameters(), lr=self.lr)

    def train_wights(self, training_stock_df):
        print("Training model...")
        for epoch in tqdm.trange(self.epochs):
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

                input_window = input_window.to(self.device)
                ground_truth = ground_truth.to(self.device)

                output = self.model(input_window).squeeze()  # Remove batch and channel dimensions
                loss = self.criterion(ground_truth, output)

                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()

            if (epoch + 1) % 5 == 0:  # Print progress
                tqdm.tqdm.write(f"Loss: {total_loss}")

            if (epoch + 1) % 20 == 0:  # Save checkpoint
                torch.save(self.model.state_dict(), build_checkpoint_path(epoch))
    
        torch.save(self.model.state_dict(), build_checkpoint_path("FINAL"))

    def get_portfolio(self, recent_data: pd.DataFrame):
        self.model.eval()  # set model to evaluation mode
        return self.model(recent_data.iloc[-31:, -num_stocks:])


if __name__ == '__main__':
    with open("train_dataset.pkl", "rb") as f:
        data = pickle.load(f)

    adj_close = data['Adj Close']
    # adj_close = adj_close.iloc[-31:, -num_stocks:]
    returns = adj_close.pct_change(1, fill_method="ffill")
    print(data)

    portfolio = portfolio2CNN()
    portfolio.train_wights(returns)
    # print(portfolio.get_portfolio())
