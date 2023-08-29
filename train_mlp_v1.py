import os
import sys
import json
import pickle
import argparse
import random
from uuid import uuid4

import torch
from torch.utils.data import DataLoader

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm

from mlp.network import RRNet
from mlp.dataset import SingleReturnsDataset
from mlp.utils import num_params


def test_model(model, loader, device):
    """
    using the network predicted returns to evaluate the model
    """
    S = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            w = model(x)
            r = w * y
            s = torch.std(r, dim=-1) / torch.mean(r, dim=-1)
            S.append(s)
    S = torch.concat(S)
    return torch.mean(S).item()


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--output-dir", type=str, required=True)
    args.add_argument("--epochs", type=int, required=True)
    args.add_argument("--seed", type=int, required=True)
    args.add_argument("--depth", type=int, required=True)
    args.add_argument("--batch_size", type=int, default=64)
    args.add_argument("--pct-change", type=int, default=1)
    args.add_argument("--dropout", type=float, default=0.0)
    args.add_argument("--learning-rate", type=float, default=0.001)
    args.add_argument("--normalization-factor", type=float, default=0.01)
    args.add_argument("--normalization-order", type=int, default=1)
    args.add_argument("--step-lr-gamma", type=float, default=0.9)
    args.add_argument("--step-lr-every", type=int, default=10)
    args.add_argument("--use_batch_norm", action="store_true")
    args.add_argument("--use_skip_connection", action="store_true")
    args.add_argument("--apply_augments", action="store_true")
    args.add_argument("--device", type=str, default="cuda:0")
    args = args.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    experiment_id = str(uuid4()).replace("-", "")[:8]
    print("EXPERIMENT:", experiment_id)

    experiment_dir = os.path.join(args.output_dir, experiment_id)
    os.makedirs(experiment_dir, exist_ok=False)

    with open(os.path.join(experiment_dir, "args.json"), "w") as f:
        json.dump(args.__dict__, f, indent=2)

    # DATA LOADING
    with open("train_dataset.pkl", "rb") as f:
        train_df = pickle.load(f)

    with open("test_dataset.pkl", "rb") as f:
        test_df = pickle.load(f)

    train_dataset = SingleReturnsDataset(train_df, apply_augment=args.apply_augments, pct_change=args.pct_change)
    test_dataset = SingleReturnsDataset(test_df, apply_augment=args.apply_augments, pct_change=args.pct_change)

    print("train dataset length", len(train_dataset))
    print("test dataset length", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    # NETWORK
    dim = len(train_dataset.returns.columns)

    net = RRNet(
        dim=dim, 
        depth=args.depth, 
        activation=torch.nn.ELU, # TODO: change and add to argparser 
        dropout=args.dropout,
        batch_norm=args.use_batch_norm,
        skip_connection=args.use_skip_connection
    ).to(args.device)

    print("number of parameters:", num_params(net))

    # OPTIMIZATION
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate, maximize=True)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=args.step_lr_every, 
        gamma=args.step_lr_gamma
    )

    net.train()

    test_sharpe_ratio = []
    best_epoch_sharpe = (-1, -1) # iteration, share ratio

    nf = args.normalization_factor
    nord = args.normalization_order

    with tqdm(total=args.epochs * len(train_loader), postfix={"epoch":0, "lr": lr_scheduler.get_last_lr()}) as progress:
        for epoch in range(args.epochs):
            progress.set_postfix(epoch=(epoch + 1), lr=lr_scheduler.get_last_lr())
            for i, (X, y) in enumerate(train_loader):
                X = X.to(args.device)
                y = y.to(args.device)

                # forward
                output = net(X)
                r = output * y
                sharpe_ratio = torch.mean(torch.std(r, dim=-1) / torch.mean(r, dim=-1) + nf * torch.norm(output, p=nord, dim=-1))

                # backward
                optimizer.zero_grad()
                sharpe_ratio.backward()
                optimizer.step()

                # print statistics
                progress.update()

            # evaluate on test set
            ts = test_model(net, test_loader, args.device)

            if ts > best_epoch_sharpe[1]:
                best_epoch_sharpe = (epoch, ts)
                torch.save(net.state_dict(), os.path.join(experiment_dir, "model_best_sharpe.pth"))

            test_sharpe_ratio.append(ts)

            lr_scheduler.step()

    print(f"Best epoch {best_epoch_sharpe[0]} with Sharpe ratio {best_epoch_sharpe[1]}")

    # save test results to file
    with open(os.path.join(experiment_dir, "test_sharpe_ratio.txt"), "w") as f:
        f.writelines([str(i) + "\n" for i in test_sharpe_ratio])

    # save last weights to file
    torch.save(net.state_dict(), os.path.join(experiment_dir, "model_last.pth"))

    plt.figure(figsize=(10,5))
    plt.plot(test_sharpe_ratio, label="test")
    plt.ylabel("Sharpe Ratio")
    plt.xlabel("Epoch")
    plt.axhline(0)
    plt.savefig(os.path.join(experiment_dir, "test_share_ratio.png"))
    plt.close()
