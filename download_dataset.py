import pandas as pd
import yfinance as yf
import numpy as np
import pickle

start_date = '2012-01-01'
end_date = '2023-07-07'

sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]

tickers = []
for sector, sector_tickers in sp500_tickers.groupby("GICS Sector"):
    tickers.append(sector_tickers.sample(5))
tickers = pd.concat(tickers)

data = yf.download(tickers["Symbol"].tolist(), start=start_date, end=end_date)

with open("dataset.pkl", "wb") as f:
    pickle.dump(data, f)