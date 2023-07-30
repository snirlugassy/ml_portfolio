import os
import sys
import argparse

import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd

from dataset import StockReturnsDataset

# CONSTANTS
DATASET_COLUMNS = ['Close', 'Low', 'Adj Close', 'Volume', 'High', 'Open']
START_DATE = '2018-07-01'
END_TRAIN_DATE = '2023-05-31'
END_TEST_DATE = '2023-06-30'


# DEFAULT HYPERPARAMETERS
WINDOW_SIZE = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 10
HIDDEN_SIZE = 64
NUM_LAYERS = 2
DROPOUT = 0.2


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--experiment", type=str, required=True, help="Experiment name, will be used as output folder name")
    args.add_argument("--train-dataset", type=str, default="./train_dataset.pkl")
    args.add_argument("--test-dataset", type=str, default="./train_dataset.pkl")

    args.add_argument("--window_size", type=int, default=WINDOW_SIZE)
    args.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
    args.add_argument("--epochs", type=int, default=EPOCHS)
    args.add_argument("--dropout", type=float, default=DROPOUT)

    args.add_argument("--hidden_size", type=int, default=HIDDEN_SIZE)
    args.add_argument("--num_layers", type=int, default=NUM_LAYERS)

    args.add_argument("--seed", type=int, default=42)
    args.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    args = args.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_df = pd.read_pickle(args.train_dataset)
    test_df = pd.read_pickle(args.test_dataset)

    train_dataset = StockReturnsDataset(train_df, window_size=args.window_size)
    test_dataset = StockReturnsDataset(test_df, window_size=args.window_size)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # TODO: build model

    # TODO: optimizer + loss function

    # TODO: train loop

    # TODO: evaluation and results