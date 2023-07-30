#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import yfinance as yf
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import cvxpy

plt.style.use("ggplot")


# ## Homework 4
# 
# Question A

# In[2]:


start_date = '2017-01-01'
end_date = '2022-01-01'

sp500_tickers = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
sp500_tickers


# In[3]:


np.random.seed(42)
tickers = []
for sector, sector_tickers in sp500_tickers.groupby("GICS Sector"):
    tickers.append(sector_tickers.sample(5))
tickers = pd.concat(tickers)
tickers.head()


# In[4]:


data = yf.download(tickers["Symbol"].tolist(), start=start_date, end=end_date)
data


# In[5]:


adj_close = data['Adj Close']
all_weekdays = pd.date_range(start=start_date, end=end_date, freq='B')
adj_close = adj_close.reindex(all_weekdays)
adj_close = adj_close.fillna(method='ffill')
# adj_close = adj_close.dropna(axis=0)
adj_close


# In[6]:


adj_close.describe()


# In[7]:


relative_returns = adj_close.pct_change(1)
relative_returns


# Question B

# In[8]:


def get_min_var_portfolio(train_data):
    C = train_data.cov()
    C_inv = np.linalg.inv(C)
    e = np.ones(C_inv.shape[0])
    X_mvp = C_inv @ e / (e @ C_inv @ e)
    X_mvp /= X_mvp.sum() # renormalize the weights
    return X_mvp

def get_mean_std(portfolio, data):
    C = data.cov()
    R = data.mean(axis=0).values
    portfolio_returns = portfolio @ R
    portfolio_std = np.sqrt(portfolio @ C @ portfolio)
    return portfolio_returns, portfolio_std


# Question C

# In[9]:


x_mvp = get_min_var_portfolio(relative_returns)
print("MVP portfolio:", x_mvp)


# In[10]:


mvp_mean_rr, mvp_std_rr = get_mean_std(x_mvp, relative_returns)
print("MVP Mean:", mvp_mean_rr)
print("MVP STD:", mvp_std_rr)


# Question D
# 
# We will solve the following optimization:
# 
# $$
# min_x : x^T C x + \tau \cdot ||x||_1
# $$
# 

# In[11]:


C = relative_returns.cov().values
n = C.shape[0]
C.shape, type(C)


# In[12]:


def find_min_var_l1(train_data, tau):
    C = train_data.cov().values
    x = cvxpy.Variable(len(C))
    c = cvxpy.Constant(C)
    obj = cvxpy.Minimize( cvxpy.quad_form(x, c, assume_PSD=True) + tau * cvxpy.norm1(x))
    constraints = [ cvxpy.sum(x) == 1, x >= 0 ]
    prob = cvxpy.Problem(obj, constraints)
    prob.solve()
    return x.value


# In[13]:


x_l1_reg = find_min_var_l1(relative_returns, 0.01)
x_l1_reg


# Question E

# In[14]:


get_rr_by_year = lambda year: relative_returns[relative_returns.index.year == year]
get_rr_by_year(2021)


# In[15]:


results = {}
for year in [2019, 2020, 2021]:
    results[year] = {}
    prev_2_year = relative_returns[(relative_returns.index.year >= (year - 2)) & (relative_returns.index.year < (year))]
    prev_2_year = prev_2_year.dropna(axis=1, how="all") # remove stocks without any data
    current_year_rr = relative_returns[relative_returns.index.year == year][prev_2_year.columns]
    
    x_mvp = get_min_var_portfolio(prev_2_year)
    results[year]['mvp_mean'], results[year]['mvp_std'] = get_mean_std(x_mvp, prev_2_year)
    results[year]['mvp_mean_actual'], results[year]['mvp_std_actual'] = get_mean_std(x_mvp, current_year_rr)

    for tau in [0.01, 0.1, 0.5]:
        x_l1_reg = find_min_var_l1(prev_2_year, tau)
        results[year][f'l1_reg_{tau}_mean'], results[year][f'l1_reg_{tau}_std'] = get_mean_std(x_l1_reg, prev_2_year)
        results[year][f'l1_reg_{tau}_mean_actual'], results[year][f'l1_reg_{tau}_std_actual'] = get_mean_std(x_l1_reg, current_year_rr)

results = pd.DataFrame(results)
results


# Question F

# In[16]:


three_year_avg = results.mean(axis=1)
three_year_avg


# Question G

# In[17]:


plt.figure(figsize=(10,10))

# overall - green
# expected - blue
# actual - purple

plt.scatter(mvp_std_rr, mvp_mean_rr, color="green", label="overall MVP")
plt.scatter(three_year_avg['mvp_std'], three_year_avg['mvp_mean'], color="blue", label="Avg. expected 3-year MVP", marker="+")
plt.scatter(three_year_avg['mvp_std_actual'], three_year_avg['mvp_mean_actual'], color="purple", label="Avg. actual 3-year MVP", marker="+")

plt.scatter(three_year_avg['l1_reg_0.01_std'], three_year_avg['l1_reg_0.01_mean'], color="blue", label="Reg. MVP t=0.01", marker="s")
plt.scatter(three_year_avg['l1_reg_0.1_std'], three_year_avg['l1_reg_0.1_mean'], color="blue", label="Reg. MVP t=0.1", marker="p")
plt.scatter(three_year_avg['l1_reg_0.5_std'], three_year_avg['l1_reg_0.5_mean'], color="blue", label="Reg. MVP t=0.5", marker="d")

plt.scatter(three_year_avg['l1_reg_0.01_std_actual'], three_year_avg['l1_reg_0.01_mean_actual'], color="purple", label="Actual Reg. MVP t=0.01", marker="s")
plt.scatter(three_year_avg['l1_reg_0.1_std_actual'], three_year_avg['l1_reg_0.1_mean_actual'], color="purple", label="Actual Reg. MVP t=0.1", marker="p")
plt.scatter(three_year_avg['l1_reg_0.5_std_actual'], three_year_avg['l1_reg_0.5_mean_actual'], color="purple", label="Actual Reg. MVP t=0.5", marker="d")


plt.title("Comparing risk-return for different portfolios")

plt.xlabel("std")
plt.ylabel("return")

# plt.axhline(0, color="black")
# plt.axvline(0, color="black")
plt.legend()
plt.show()


# Question H
# 
# Q: What is the optimal $ \tau $ among the given options?
# 
# A: $ \tau = 0.5 $ is better than $ \tau = 0.1 $, However, There is no actual optimal $ \tau $. 
# 
# It can be selected based on the risk-return preference of the investor (lower risk $ \tau = 0.01$, higher risk $ \tau = 0.5$)

# Question I
# 
# Q: How would you improve this result?
# 
# A: 
# 
# - We would use a rolling / expanding window instead of a single window to estimate the Mean and STD of returns for different $ \tau $.
# - We would try to consider more $ \tau $ values and than interpolate the Mean and STD as a function of $ \tau $.
# - Try different regularzation methods, e.g., L2 norm / L-infinity norm.
# - (Best option) **Adding the expected return to the optimiztion problem**, that is:
# 
# $$
# min_x : x^T C x - x^T \bar {R} + \tau \cdot ||x||_1
# $$
# 
# Than we will both minimize the risk and maximize the reward at the same time (full quad form)
# 
