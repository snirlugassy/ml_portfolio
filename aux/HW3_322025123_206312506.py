#!/usr/bin/env python
# coding: utf-8

# In[283]:


import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
plt.style.use("ggplot")


# S&P 500 stocks data

# In[284]:


start_date = '2017-01-01'
end_date = '2022-01-01'

sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp500_tickers


# Randomly sampling 5 stocks from each GICS sector

# In[285]:


np.random.seed(42)
tickers = []
for sector, sector_tickers in sp500_tickers.groupby("GICS Sector"):
    tickers.append(sector_tickers.sample(5))
tickers = pd.concat(tickers)
tickers.head()


# Using Yahoo Finance to fetch data of the sampled stocks between 01/01/2017 - 01/01/2023

# In[286]:


data = yf.download(tickers["Symbol"].tolist(), start=start_date, end=end_date)
data


# In[287]:


features = list({x for x,y in data.columns})
features


# Preprocessing of the adjusted closing price data

# In[288]:


adj_close = data['Adj Close']
all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
adj_close = adj_close.reindex(all_weekdays)
adj_close = adj_close.fillna(method='ffill')
adj_close


# In[289]:


adj_close.describe()


# Calculating the relative returns using pandas "pct_change" function (the percentage of change between each row and the previous row)
# 
# $$ 
# r_{rel} = {{p_t - p_{t-1}} \over {p_{t-1}}}
# $$

# In[290]:


relative_returns = adj_close.pct_change(1)
relative_returns


# In[291]:


rr_stds, rr_means = relative_returns.std(axis=0), relative_returns.mean(axis=0)
plt.figure(figsize=(7,7))
plt.scatter(rr_stds, rr_means)
plt.xlabel("STD of relative return")
plt.ylabel("Mean of relative return")
plt.show()


# ## Question B

# Finding the Minial Variance Portfolio

# In[292]:


C = relative_returns.cov()
C_inv = np.linalg.inv(C)
e = np.ones(C_inv.shape[0])

X_mvp = C_inv @ e / (e @ C_inv @ e)
# X_mvp = X_mvp * 1*(np.abs(X_mvp) > eps) # ignore very small investments
X_mvp /= X_mvp.sum() # renormalize the weights
X_mvp.sum(), X_mvp


# Finding the tangent portfolio without risk-free assents (basket portfolio)
# 
# - Solve $Cy = R \Rightarrow y = C^{-1} R$
# - Normalize $ x = {y \over \Sigma y} $

# In[293]:


R = rr_means.values

X_tan_basket = C_inv @ R
# X_tan_basket = X_tan_basket * 1*(np.abs(X_tan_basket) > eps) # ignore very small investments
X_tan_basket /= X_tan_basket.sum()
X_tan_basket.sum(), X_tan_basket


# ## Question C

# The risky portfolio curve
# 
# The efficient frontier of risky assets lies on the curve:
# 
# $$ 
# X(\gamma) = (1-\gamma) \cdot X_{tan} + \gamma \cdot X_{mvp}
# $$

# In[294]:


def get_efficient_portfolio(gamma:float):
    return (1-gamma) * X_tan_basket + gamma * X_mvp

eff_frontier = [get_efficient_portfolio(x) for x in np.linspace(-5, 1, 100)]
eff_rrs = [float(x@R) for x in eff_frontier]
eff_stds = [float(np.sqrt(x@C@x)) for x in eff_frontier]

plt.figure(figsize=(7,7))
plt.title("The tangent portfolio curve")
plt.plot(eff_stds, eff_rrs)
plt.xlabel("risk (var)")
plt.ylabel("return")
plt.show()


# ## Question D

# In[295]:


X_mvp_return = X_mvp @ R
X_mvp_std = np.sqrt(X_mvp @ C @ X_mvp)

X_tan_basket_return = X_tan_basket @ R
X_tan_basket_std = np.sqrt(X_tan_basket @ C @ X_tan_basket)


# In[296]:


plt.figure(figsize=(7,7))
plt.title("Tangent portfolio curve")

# portfolios
plt.scatter(rr_stds, rr_means, c="blue")
plt.scatter(X_mvp_std, X_mvp_return, color="orange", label="MVP")
plt.scatter(X_tan_basket_std, X_tan_basket_return, color="red", label="Basket")

# curve
plt.plot(eff_stds, eff_rrs)

plt.legend(loc="best")

plt.xlabel("std")
plt.ylabel("return")

plt.xlim((-0.001, 0.05))
plt.ylim((-0.001,0.007))

plt.axvline(x=0, color='black')
plt.axhline(y=0, color="black")

plt.show()


# ## Question E
# 
# Tangent portfolio using risk free assert with $ R_f=0.05 / 250$

# In[297]:


R_f = 0.05 / 250
X_tan = C_inv @ (R-R_f)
# X_tan = X_tan * 1*(X_tan > eps) # ignore very small investments
X_tan /= X_tan.sum()
X_tan.sum(), X_tan


# ## Question F

# Finding the investment line:
# 
# $$
# R_p = R_f + m \cdot \sigma_p  
# $$
# 
# $$
# m = (\bar R_A - R_f) / \sigma_A
# $$

# In[298]:


R_tan = X_tan @ R
SD_tan = np.sqrt(X_tan @ C @ X_tan)
slope = (R_tan - R_f) / SD_tan

print(R_f, slope)


# The investment line:
# 
# $$
# R_p = 0.0002 + 0.1621 * \sigma_p
# $$

# ## Question G

# In[299]:


def inv_line(s):
    return R_f + s * slope

x0 = 0
y0 = inv_line(x0)

plt.figure(figsize=(15,10))
plt.title("Tangent portfolio and investment line")

# portfolios
plt.scatter(rr_stds, rr_means, c="blue")
plt.scatter(X_mvp_std, X_mvp_return, color="orange", label="MVP")
plt.scatter(X_tan_basket_std, X_tan_basket_return, color="red", label="Basket")
plt.scatter(SD_tan, R_tan, color="green", label="Tangent")

# curve
plt.plot(eff_stds, eff_rrs)

# investment line
plt.axline((x0,y0), slope=slope, color="purple", label="investment line")

plt.legend(loc="best")

plt.xlabel("std")
plt.ylabel("return")

plt.xlim((-0.001, 0.05))
plt.ylim((-0.001,0.007))

plt.axvline(x=0, color='black')
plt.axhline(y=0, color="black")

plt.show()


# ## Question H
# 
# Q: How does the investment line relate to the efficient frontier?
# 
# A: The investment line is tangent to the efficient frontier curve. Intersecting with the curve at the risk-return point of the tangent portfolio (with risk-free assets)

# ## Question I

# In[300]:


def calc_mvp_portfolio(window):    
    C = window.cov()
    C_inv = np.linalg.inv(C)
    e = np.ones(C_inv.shape[0])
    _mvp = C_inv @ e / (e @ C_inv @ e)
    _mvp /= _mvp.sum() # renormalize the weights

    window_mean = window.mean(axis=0)
    _er = _mvp @ window_mean
    _std = np.sqrt(_mvp @ C @ _mvp)

    return _mvp, _er, _std


# In[301]:


relative_returns


# In[302]:


np.any(relative_returns[2:-2])


# In[303]:


N = len(relative_returns)

# iterate on windows of 500 days
# 500 + 2 to ignore NaN first 2 days
# N-1 since the last window doesn't have "next day"

p500_data = []
for i in tqdm(range(500 + 2, N-1)):
    rr_window = relative_returns.iloc[i-500:i]
    rr_window = rr_window.dropna(how="all", axis=1) # remove columns with no values
    next_day_returns = relative_returns[rr_window.columns].iloc[i+1]
    window_mean = rr_window.mean(axis=0)
    
    C = rr_window.cov()
    C_inv = np.linalg.inv(C)
    e = np.ones(C_inv.shape[0])
    
    # mvp
    x_mvp = C_inv @ e / (e @ C_inv @ e)
    x_mvp /= x_mvp.sum() # renormalize the weights

    # tangent basket
    x_basket = C_inv @ window_mean
    x_basket /= x_basket.sum()

    p500_data.append({
        "mvp_er": x_mvp @ window_mean,
        "mvp_std": np.sqrt(x_mvp @ C @ x_mvp),
        "basket_er": x_basket @ window_mean,
        "basket_std": np.sqrt(x_basket @ C @ x_basket),
        "true_mvp_return": next_day_returns.values @ x_mvp,
        "true_basket_return": next_day_returns.values @ x_basket,
    })


# ### Question I.2
# 
# MVP mean return and return STD using the rolling window:

# In[304]:


mvp_window_returns = np.array([x['mvp_er'] for x in p500_data if not np.isnan(x['mvp_er'])])
mvp_window_stds = np.array([x['mvp_std'] for x in p500_data if not np.isnan(x['mvp_std'])])

print("MVP mean return", mvp_window_returns.mean())
print("MVP mean return std", mvp_window_stds.mean())


# Tangent basket portfolio mean return and return STD using the rolling window:

# In[305]:


basket_window_returns = np.array([x['basket_er'] for x in p500_data if not np.isnan(x['basket_er'])])
basket_window_stds = np.array([x['basket_std'] for x in p500_data if not np.isnan(x['basket_std'])])

print("Tangent Basket mean return", basket_window_returns.mean())
print("Tangent Basket mean return std", basket_window_stds.mean())


# In[306]:


true_mvp_returns = np.array([x['true_mvp_return'] for x in p500_data if not np.isnan(x['true_mvp_return'])])
print("True MVP mean return", true_mvp_returns.mean())
print("True MVP mean return std", true_mvp_returns.std())

true_basket_returns = np.array([x['true_basket_return'] for x in p500_data if not np.isnan(x['true_basket_return'])])
print("True basket mean return", true_basket_returns.mean())
print("True basket mean return std", true_basket_returns.std())


# ## Question I.3

# In[308]:


plt.figure(figsize=(15,10))
plt.title("Comparing MVP and Basket using sliding window of 500 days")

# portfolios
# MVP on all data
plt.scatter(X_mvp_std, X_mvp_return, color="blue", label="MVP (all)", marker="*", s=100)

# Tangent basket on all data
plt.scatter(X_tan_basket_std, X_tan_basket_return, color="purple", label="Basket (all)", marker="*", s=100)

# MVP on sliding window of 500 days
plt.scatter(mvp_window_stds.mean(), mvp_window_returns.mean(), color="blue", label="MVP (window)", s=100)

# Tangent basket on sliding window of 500 days
plt.scatter(basket_window_stds.mean(), basket_window_returns.mean(), color="purple", label="Basket (window)", s=100)

# True values on sliding window of 500 days
plt.scatter(true_mvp_returns.std(), true_mvp_returns.mean(), color="blue", label="True MVP (window)", marker="+", s=100)
plt.scatter(true_basket_returns.std(), true_basket_returns.mean(), color="purple", label="True Basket (window)", marker="+", s=100)


plt.legend(loc="best")

plt.xlabel("std")
plt.ylabel("return")

plt.axvline(x=0, color='black')
plt.axhline(y=0, color="black")

plt.show()


# ## Question J
# 
# Q: What is the difference between expected to actual value point?
# 
# A: In the MVP the actual value had a **better return than expected**.  
# However, in the best basket, for both sliding window and all data, the **actual return was worst than expected**
# 
# We can try to explaing this using the nature of both portfolios:
# - MVP is more conservative, looking to minimize risk and sticking to stocks that are more certain on their return.
# - Best basket is theoretically optimal, and seeking efficiency, however the stock market usually doesn't conform to theoretical promises (if that was the case everyone would apply the same theoretical solution and "win", that is impossible in markets)
# 
# ## Question K
# 
# Q: What is a possible solution?
# 
# A: Either stick to the MVP and minimize your risks, or estimate the expected return better, e.g., using more complex model, such neural network or large decision trees.
