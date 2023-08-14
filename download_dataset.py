import pandas as pd
import yfinance as yf
import numpy as np
import pickle


def download_dataset(start_date, end_date):
    wiki_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    sp_tickers = wiki_table[0]
    tickers = [ticker.replace('.', '-') for ticker in sp_tickers['Symbol'].to_list()]
    data = yf.download(tickers, start_date, end_date)
    return data


if __name__ == '__main__':
    START_DATE = '2018-07-01'
    END_TRAIN_DATE = '2023-05-31'
    END_TEST_DATE = '2023-06-30'

    dataset = download_dataset(START_DATE, END_TEST_DATE)
    print(f"Dataset shape: {dataset.shape}")

    with open("dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)
    
    train_data = dataset[dataset.index < END_TRAIN_DATE]
    print(f"Train data shape: {train_data.shape}")
    with open("train_dataset.pkl", "wb") as f:
        pickle.dump(train_data, f)

    test_data = dataset[dataset.index >= END_TRAIN_DATE]
    print(f"Test data shape: {test_data.shape}")
    with open("test_dataset.pkl", "wb") as f:
        pickle.dump(test_data, f)