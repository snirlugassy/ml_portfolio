#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
plt.style.use("ggplot")


# In[2]:


np.random.seed(42)


# S&P 500 stocks data

# In[3]:


start_date = '2017-01-01'
end_date = '2023-01-01'

sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp500_tickers


# Randomly sampling 5 stocks from each GICS sector

# In[4]:


tickers = []
for sector, sector_tickers in sp500_tickers.groupby("GICS Sector"):
    tickers.append(sector_tickers.sample(5))
tickers = pd.concat(tickers)
tickers.head()


# Using Yahoo Finance to fetch data of the sampled stocks between 01/01/2017 - 01/01/2023

# In[5]:


data = yf.download(tickers["Symbol"].tolist(), start=start_date, end=end_date)
data


# In[6]:


features = list({x for x,y in data.columns})
features


# Preprocessing of the adjusted closing price data

# In[7]:


stock_adj_close = data['Adj Close']
all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
stock_adj_close = stock_adj_close.reindex(all_weekdays)
stock_adj_close = stock_adj_close.fillna(method='ffill')
stock_adj_close


# In[8]:


stock_adj_close.describe()


# Calculating the relative returns using pandas "pct_change" function (the percentage of change between each row and the previous row)
# 
# $$ 
# r_{rel} = {{p_t - p_{t-1}} \over {p_{t-1}}}
# $$

# In[9]:


relative_returns = stock_adj_close.pct_change(1)
relative_returns


# In[55]:


rr_stds, rr_means = relative_returns.std(axis=0), relative_returns.mean(axis=0)
plt.figure(figsize=(7,7))
plt.title("Relative Returns")
plt.scatter(rr_stds, rr_means)
plt.xlabel("STD of relative return")
plt.ylabel("Mean of relative return")
plt.show()


# In the log returns we are adding +1 in order to avoid -inf
# 
# 
# $$ 
# log(1+r_{rel}) = log(1 + {{p_t - p_{t-1}} \over {p_{t-1}}}) = log(1 + {{p_t} \over {p_{t-1}}} - {{p_{t-1}} \over {p_{t-1}}}) = log({{p_t} \over {p_{t-1}}} )
# $$

# In[53]:


log_relative_returns = np.log(1+stock_adj_close).diff()

log_rr_stds, log_rr_means = log_relative_returns.std(axis=0), log_relative_returns.mean(axis=0)

plt.figure(figsize=(7,7))

plt.title("Log Relative Returns")
plt.scatter(xs, log_rr_means)
plt.xlabel("STD of log relative return")
plt.ylabel("Mean of log relative return")
plt.show()


# Finding the efficient frontiers

# In[56]:


N = len(rr_stds)
efficient_frontiers = []
for i in range(N):
    is_efficient = True
    for j in range(N):
        if i == j:
            continue

        if rr_stds[i] > rr_stds[j] and rr_means[i] <= rr_means[j]:
            is_efficient = False
            break

        if rr_stds[i] >= rr_stds[j] and rr_means[i] < rr_means[j]:
            is_efficient = False
            break
    
    if is_efficient:
        efficient_frontiers.append(i)

print(f"{len(efficient_frontiers)} Efficient frontier:", stock_adj_close.columns[efficient_frontiers].tolist())


# In[57]:


plt.figure(figsize=(7,7))
plt.title("Relative Returns with efficient frontier")

plt.scatter(rr_stds, rr_means, c="blue")

for i in efficient_frontiers:
    plt.scatter(rr_stds[i], rr_means[i], c="red")

plt.legend(["not efficient", "efficient"], loc="best")

plt.xlabel("STD of relative return")
plt.ylabel("Mean of relative return")
plt.show()


# In[58]:


covariance_matrix = relative_returns.cov()

plt.figure(figsize=(10,10))
plt.title("Stocks relative returns covariance (heatmap)")
plt.imshow(covariance_matrix.values)
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.show()


# In[59]:


correlation_matrix = relative_returns.corr()

plt.figure(figsize=(10,10))
plt.title("Stocks relative returns correlation (heatmap)")
plt.imshow(correlation_matrix.values, vmin=-1, vmax=1)
plt.xticks([])
plt.yticks([])
plt.colorbar()
plt.show()


# Finding the Minimal Variance Portfolio:
# 
# $$
# X = {{C^{-1} e} \over {e C^{-1} e}}
# $$

# In[61]:


cov_inv = np.linalg.inv(covariance_matrix)
e = np.ones(cov_inv.shape[0])
X_mvp = cov_inv @ e / (e @ cov_inv @ e)
X_mvp


# In[62]:


tickers = relative_returns.columns.tolist()

plt.figure(figsize=(20,5))
plt.title("MV Portfolio")
plt.bar(tickers, X_mvp)
plt.xticks(rotation=90)
plt.xlabel("Stock")
plt.ylabel("Stock portfolio weight in MVP")
plt.show()

