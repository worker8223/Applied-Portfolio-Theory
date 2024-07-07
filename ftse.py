#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install yfinance


# In[2]:


import yfinance as yf

def download_stock_data(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
    return data

tickers = ['^FTSE', 'BP', 'SHEL.L', 'AZN', 'BARC.L', 'LYG', 'NWG', 'HSBC']
start_date = '2010-01-01'
end_date = '2024-06-30'  # 'today' will automatically be interpreted as today's date

stock_data = download_stock_data(tickers, start_date, end_date)
print(stock_data.head())


# In[3]:


print(stock_data)


# In[4]:


import pandas as pd
import numpy as np
from scipy.stats import mode
from math import sqrt

# Calculate mean
mean = stock_data.mean()

# Calculate median
median = stock_data.median()

# Calculate mode
mode_result = stock_data.mode().iloc[0]

# Calculate daily variance
variance = stock_data.var()

# Calculate daily standard deviation
std_dev = stock_data.std()

# Calculate annualized volatility (assuming 252 trading days in a year)
annual_volatility = std_dev * np.sqrt(252)

# Calculate historical volatility (standard deviation of daily returns)
daily_returns = stock_data.pct_change().dropna()
historical_volatility = daily_returns.std()

# Calculate skewness
skewness = daily_returns.skew()

# Calculate kurtosis
kurtosis = daily_returns.kurtosis()

# Combine all statistics into a DataFrame
statistics = pd.DataFrame({
    'Mean': mean,
    'Median': median,
    'Mode': mode_result,
    'Daily Variance': variance,
    'Daily Standard Deviation': std_dev,
    'Annualized Volatility': annual_volatility,
    'Historical Volatility': historical_volatility,
    'Skewness': skewness,
    'Kurtosis': kurtosis
})

# Print statistics
print(statistics)


# In[5]:


# Calculate percentage daily returns
percentage_returns = stock_data.pct_change().dropna()

# Print percentage daily returns
print(percentage_returns.head())


# In[6]:


import numpy as np

# Calculate log returns
log_returns = np.log(stock_data / stock_data.shift(1)).dropna()

# Print log returns
print(log_returns.head())


# In[7]:


import matplotlib.pyplot as plt

# Plot histogram distribution of daily percentage returns for each ticker
for ticker in percentage_returns.columns:
    plt.figure(figsize=(20, 20))
    plt.hist(percentage_returns[ticker], bins=30, alpha=0.7, color='blue')
    plt.title(f'Distribution of Daily Percentage Returns for {ticker}')
    plt.xlabel('Percentage Returns')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()


# In[8]:


# Calculate covariance matrix
covariance_matrix = percentage_returns.cov()

# Calculate correlation matrix
correlation_matrix = percentage_returns.corr()

# Calculate inverse covariance matrix
inverse_covariance_matrix = np.linalg.inv(covariance_matrix)

# Print covariance matrix
print("Covariance Matrix:")
print(covariance_matrix)

# Print correlation matrix
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Print inverse covariance matrix
print("\nInverse Covariance Matrix:")
print(inverse_covariance_matrix)


# In[30]:


import seaborn as sns

# Plot covariance matrix
plt.figure(figsize=(20, 18))
sns.heatmap(covariance_matrix, annot=True, cmap='coolwarm', fmt=".4f", linewidths=.5)
plt.title('Covariance Matrix of Daily Percentage Returns')
plt.show()

# Plot correlation matrix
plt.figure(figsize=(20, 18))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Matrix of Daily Percentage Returns')
plt.show()

# Plot inverse covariance matrix
plt.figure(figsize=(20, 18))
sns.heatmap(inverse_covariance_matrix, annot=True, cmap='coolwarm', fmt=".4f", linewidths=.5)
plt.title('Inverse Covariance Matrix')
plt.show()


# In[10]:


# Calculate daily variance of daily returns
daily_variance = daily_returns.var()

# Calculate daily standard deviation of daily returns
daily_std_dev = daily_returns.std()

# Calculate skewness of daily returns
daily_skewness = daily_returns.skew()

# Calculate kurtosis of daily returns
daily_kurtosis = daily_returns.kurtosis()

# Combine statistics into a DataFrame
daily_statistics = pd.DataFrame({
    'Daily Variance': daily_variance,
    'Daily Standard Deviation': daily_std_dev,
    'Skewness': daily_skewness,
    'Kurtosis': daily_kurtosis
})

# Print statistics
print(daily_statistics)


# In[11]:


# Calculate weekly volatility
weekly_volatility = daily_returns.resample('W').std()

# Calculate monthly volatility
monthly_volatility = daily_returns.resample('M').std()

# Print weekly volatility
print("Weekly Volatility:")
print(weekly_volatility)

# Print monthly volatility
print("\nMonthly Volatility:")
print(monthly_volatility)


# In[12]:


# Plot weekly volatility
plt.figure(figsize=(12, 6))
for ticker in weekly_volatility.columns:
    plt.plot(weekly_volatility.index, weekly_volatility[ticker], label=ticker)

plt.title('Weekly Volatility of Daily Returns')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.show()

# Plot monthly volatility
plt.figure(figsize=(12, 6))
for ticker in monthly_volatility.columns:
    plt.plot(monthly_volatility.index, monthly_volatility[ticker], label=ticker)

plt.title('Monthly Volatility of Daily Returns')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.show()


# In[13]:


# Plot weekly and monthly volatility overlapped for each ticker
for ticker in weekly_volatility.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(weekly_volatility.index, weekly_volatility[ticker], label='Weekly Volatility', color='blue')
    plt.plot(monthly_volatility.index, monthly_volatility[ticker], label='Monthly Volatility', color='red')
    plt.title(f'Volatility of Daily Returns for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    plt.show()


# In[14]:


# Calculate annualized volatility
annualized_volatility = daily_returns.std() * np.sqrt(252)

# Calculate historical volatility
historical_volatility = daily_returns.std()

# Combine statistics into a DataFrame
volatility_statistics = pd.DataFrame({
    'Annualized Volatility': annualized_volatility,
    'Historical Volatility': historical_volatility
})

# Print statistics
print(volatility_statistics)


# In[15]:


# Calculate annualized standard deviation of log returns
annualized_log_std = log_returns.std() * np.sqrt(252)

# Plot annualized standard deviation of log returns for each ticker
plt.figure(figsize=(12, 6))
plt.bar(annualized_log_std.index, annualized_log_std.values, color='blue')
plt.title('Annualized Standard Deviation of Log Returns for Each Ticker')
plt.xlabel('Ticker')
plt.ylabel('Annualized Standard Deviation')
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.show()


# In[16]:


# Calculate rolling 30-day historical volatility moving average
rolling_volatility_ma = daily_returns.rolling(window=30).std() * np.sqrt(252)

# Plot rolling 30-day historical volatility moving average for each ticker
plt.figure(figsize=(12, 6))
for ticker in rolling_volatility_ma.columns:
    plt.plot(rolling_volatility_ma.index, rolling_volatility_ma[ticker], label=ticker)

plt.title('Rolling 30-Day Historical Volatility Moving Average')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.show()


# In[17]:


# Plot rolling 30-day historical volatility moving average for each ticker
for ticker in rolling_volatility_ma.columns:
    plt.figure(figsize=(12, 6))
    plt.plot(rolling_volatility_ma.index, rolling_volatility_ma[ticker], label=ticker)
    plt.title(f'Rolling 30-Day Historical Volatility Moving Average for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    plt.show()


# In[18]:


# Define thresholds for volatility and correlation
volatility_threshold = 0.1  # Example threshold for volatility
correlation_threshold = 0.5  # Example threshold for correlation

# Identify periods of high volatility
high_volatility_periods = rolling_volatility_ma[rolling_volatility_ma > volatility_threshold]

# Identify periods of high correlation of risk
correlation_of_risk = rolling_volatility_ma.corr()
high_correlation_of_risk_periods = correlation_of_risk[correlation_of_risk > correlation_threshold]

# Print high volatility periods
print("High Volatility Periods:")
print(high_volatility_periods)

# Print high correlation of risk periods
print("\nHigh Correlation of Risk Periods:")
print(high_correlation_of_risk_periods)


# In[19]:


# Plot rolling 30-day historical volatility moving average for each ticker
for ticker in rolling_volatility_ma.columns:
    plt.figure(figsize=(12, 6))
    
    # Plot rolling volatility
    plt.plot(rolling_volatility_ma.index, rolling_volatility_ma[ticker], label='Volatility')
    
    # Highlight periods of high volatility
    high_volatility_periods = rolling_volatility_ma[ticker][rolling_volatility_ma[ticker] > volatility_threshold]
    plt.fill_between(high_volatility_periods.index, 0, high_volatility_periods.values, color='red', alpha=0.3, label='High Volatility')
    
    plt.title(f'Rolling 30-Day Historical Volatility Moving Average for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot correlation of risk
    plt.figure(figsize=(12, 6))
    
    # Plot correlation of risk
    plt.plot(correlation_of_risk.index, correlation_of_risk[ticker], label='Correlation of Risk')
    
    # Highlight periods of high correlation of risk
    high_correlation_of_risk_periods = correlation_of_risk[ticker][correlation_of_risk[ticker] > correlation_threshold]
    plt.fill_between(high_correlation_of_risk_periods.index, 0, high_correlation_of_risk_periods.values, color='orange', alpha=0.3, label='High Correlation of Risk')
    
    plt.title(f'Correlation of Risk for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Correlation')
    plt.legend()
    plt.grid(True)
    plt.show()


# In[20]:


import arch
from arch import arch_model

# Create an empty dictionary to store GARCH models
garch_models = {}

# Fit GARCH(1,1) model for each ticker
for ticker in daily_returns.columns:
    # Specify GARCH model
    garch_model = arch_model(daily_returns[ticker], vol='Garch', p=1, q=1)
    
    # Fit the model
    garch_result = garch_model.fit(disp='off')
    
    # Store the fitted model
    garch_models[ticker] = garch_result

    # Print model summary
    print(f"GARCH(1,1) Model Summary for {ticker}:")
    print(garch_result.summary())
    print("\n")

# Rescale data
scaled_returns = daily_returns * 100

# Fit GARCH(1,1) model for each ticker
for ticker in scaled_returns.columns:
    # Specify GARCH model
    garch_model = arch_model(scaled_returns[ticker], vol='Garch', p=1, q=1)
    
    # Fit the model
    garch_result = garch_model.fit(disp='off')
    
    # Store the fitted model
    garch_models[ticker] = garch_result

    # Print model summary
    print(f"GARCH(1,1) Model Summary for {ticker}:")
    print(garch_result.summary())
    print("\n")


# In[21]:


from statsmodels.tsa.stattools import adfuller

# Perform ADF test for stationarity for each ticker
for ticker in daily_returns.columns:
    result = adfuller(daily_returns[ticker])
    print(f"ADF test results for {ticker}:")
    print(f"ADF Statistic: {result[0]}")
    print(f"P-value: {result[1]}")
    print(f"Critical Values:")
    for key, value in result[4].items():
        print(f"  {key}: {value}")
    print("\n")


# In[22]:


from statsmodels.tsa.stattools import adfuller

# Perform ADF test for each ticker
for ticker in daily_returns.columns:
    result = adfuller(daily_returns[ticker])
    print(f"ADF Test for {ticker}:")
    print('ADF Statistic:', result[0])
    print('p-value:', result[1])
    print('Critical Values:')
    for key, value in result[4].items():
        print(f'   {key}: {value}')
    print("\n")


# In[23]:


import numpy as np
import statsmodels.api as sm

# Define the lag order for the AR model
lag_order = 1  # You can adjust this value

# Fit AR model for each ticker
for ticker in daily_returns.columns:
    # Create lagged variables
    X = sm.add_constant(daily_returns[ticker].shift(1).dropna())
    y = daily_returns[ticker].iloc[1:]

    # Fit AR model
    model = sm.OLS(y, X)
    results = model.fit()

    # Extract residuals
    residuals = results.resid

    # Calculate variance of residuals as a proxy for volatility
    volatility = np.sqrt(np.mean(residuals**2))

    print(f"Volatility (MSE) for {ticker}: {volatility}")


# In[24]:


import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Define the window size for the rolling average
window = 30

# Calculate rolling average MSE volatility for each ticker
rolling_volatility_ma = pd.DataFrame()
for ticker in daily_returns.columns:
    # Create lagged variables
    X = sm.add_constant(daily_returns[ticker].shift(1).dropna())
    y = daily_returns[ticker].iloc[1:]

    # Fit AR model
    model = sm.OLS(y, X)
    results = model.fit()

    # Extract residuals
    residuals = results.resid

    # Calculate squared residuals
    squared_residuals = residuals ** 2

    # Calculate rolling mean of squared residuals
    rolling_volatility_ma[ticker] = squared_residuals.rolling(window=window).mean()

# Plot rolling average MSE volatility for each ticker
plt.figure(figsize=(12, 6))
for ticker in rolling_volatility_ma.columns:
    plt.plot(rolling_volatility_ma.index, rolling_volatility_ma[ticker], label=ticker)

plt.title('30-Day Rolling Average MSE Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.show()


# In[25]:


import matplotlib.pyplot as plt

# Calculate 30-day rolling average historical volatility for each ticker
rolling_historical_volatility_ma = daily_returns.rolling(window=window).std() * np.sqrt(window)

# Plot rolling average MSE volatility and rolling average historical volatility for each ticker
plt.figure(figsize=(12, 6))
for ticker in rolling_volatility_ma.columns:
    plt.plot(rolling_volatility_ma.index, rolling_volatility_ma[ticker], label=f'{ticker} MSE')
    plt.plot(rolling_historical_volatility_ma.index, rolling_historical_volatility_ma[ticker], linestyle='--', label=f'{ticker} Historical')

plt.title('30-Day Rolling Average Volatility')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.legend()
plt.grid(True)
plt.show()


# In[26]:


# Plot separate graphs for each ticker
for ticker in daily_returns.columns:
    plt.figure(figsize=(12, 6))

    # Plot rolling average MSE volatility
    plt.plot(rolling_volatility_ma.index, rolling_volatility_ma[ticker], label='MSE Volatility')

    # Plot rolling average historical volatility
    plt.plot(rolling_historical_volatility_ma.index, rolling_historical_volatility_ma[ticker], linestyle='--', label='Historical Volatility')

    plt.title(f'30-Day Rolling Average Volatility for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.legend()
    plt.grid(True)
    plt.show()


# In[27]:


# Plot GARCH volatility forecasts for each ticker
for ticker, garch_result in garch_models.items():
    plt.figure(figsize=(22, 16))
    
    # Plot conditional volatility forecasts
    garch_result.plot(annualize='D')
    
    plt.title(f'GARCH Volatility Forecast for {ticker}')
    plt.xlabel('Date')
    plt.ylabel('Volatility')
    plt.grid(True)
    plt.show()


# In[33]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Mock data setup (replace this with actual data)
tickers = ['AZN', 'BARC.L', 'BP', 'HSBC', 'LYG', 'NWG', 'SHEL.L', '^FTSE']
data = pd.DataFrame(np.random.randn(1000, len(tickers)), columns=tickers)  # Replace with actual data

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Define market returns (FTSE)
market_returns = returns['^FTSE']

# Calculate betas
betas = {}
for ticker in tickers:
    if ticker != '^FTSE':
        cov = np.cov(returns[ticker], market_returns)
        betas[ticker] = cov[0, 1] / cov[1, 1]

# Expected market return
market_return = market_returns.mean()

# Risk-free rate (assume a small positive rate, e.g., 0.01)
risk_free_rate = 0.01

# Calculate expected returns using CAPM
capm_returns = {}
for ticker, beta in betas.items():
    capm_returns[ticker] = risk_free_rate + beta * (market_return - risk_free_rate)

capm_returns = pd.Series(capm_returns)

# Risk Parity Model
def calculate_portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def calculate_risk_contributions(weights, cov_matrix):
    portfolio_var = calculate_portfolio_variance(weights, cov_matrix)
    marginal_contrib = np.dot(cov_matrix, weights)
    risk_contrib = weights * marginal_contrib / portfolio_var
    return risk_contrib

def risk_parity_objective(weights, cov_matrix):
    risk_contrib = calculate_risk_contributions(weights, cov_matrix)
    risk_diffs = risk_contrib - np.mean(risk_contrib)
    return np.sum(risk_diffs**2)

# Initial guess for the weights
initial_weights = np.ones(len(tickers) - 1) / (len(tickers) - 1)

# Constraints: weights sum to 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Bounds for the weights
bounds = tuple((0, 1) for _ in range(len(tickers) - 1))

# Optimization for Risk Parity
risk_parity_result = minimize(risk_parity_objective, initial_weights, args=cov_matrix.iloc[:-1, :-1],
                              method='SLSQP', bounds=bounds, constraints=constraints)

risk_parity_weights = risk_parity_result.x

# Ensure the Risk Parity weights align correctly with the tickers
risk_parity_weights = np.append(risk_parity_weights, [0])  # Append 0 for FTSE as it's not part of the optimization

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

# CAPM Allocation
ax[0].bar(capm_returns.index, capm_weights)
ax[0].set_title('CAPM Allocation')
ax[0].set_ylabel('Weights')
ax[0].set_xlabel('Assets')

# Risk Parity Allocation
ax[1].bar(tickers, risk_parity_weights)
ax[1].set_title('Risk Parity Allocation')
ax[1].set_ylabel('Weights')
ax[1].set_xlabel('Assets')

plt.tight_layout()
plt.show()


# In[34]:


pip install numpy pandas matplotlib yfinance


# In[38]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize
import yfinance as yf

# Define the tickers of the stocks and the market index (e.g., FTSE)
tickers = ['AZN', 'BARC.L', 'BP', 'HSBC', 'LYG', 'NWG', 'SHEL.L', '^FTSE']

# Download historical data
data = yf.download(tickers, start='2010-01-01', end='2024-06-30')['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Separate market returns (FTSE)
market_returns = returns['^FTSE']

# Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Calculate betas
betas = {}
for ticker in tickers:
    if ticker != '^FTSE':
        cov = np.cov(returns[ticker], market_returns)
        beta = cov[0, 1] / cov[1, 1]
        betas[ticker] = beta

# Expected market return
market_return = market_returns.mean()

# Risk-free rate (assume a small positive rate, e.g., 0.01)
risk_free_rate = 0.05

# Calculate expected returns using CAPM
capm_returns = {}
for ticker, beta in betas.items():
    capm_returns[ticker] = risk_free_rate + beta * (market_return - risk_free_rate)

capm_returns = pd.Series(capm_returns)

# Risk Parity Model
def calculate_portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def calculate_risk_contributions(weights, cov_matrix):
    portfolio_var = calculate_portfolio_variance(weights, cov_matrix)
    marginal_contrib = np.dot(cov_matrix, weights)
    risk_contrib = weights * marginal_contrib / portfolio_var
    return risk_contrib

def risk_parity_objective(weights, cov_matrix):
    risk_contrib = calculate_risk_contributions(weights, cov_matrix)
    risk_diffs = risk_contrib - np.mean(risk_contrib)
    return np.sum(risk_diffs**2)

# Initial guess for the weights
initial_weights = np.ones(len(tickers) - 1) / (len(tickers) - 1)

# Constraints: weights sum to 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Bounds for the weights
bounds = tuple((0, 1) for _ in range(len(tickers) - 1))

# Optimization for Risk Parity
risk_parity_result = minimize(risk_parity_objective, initial_weights, args=cov_matrix.iloc[:-1, :-1],
                              method='SLSQP', bounds=bounds, constraints=constraints)

risk_parity_weights = risk_parity_result.x

# Ensure the Risk Parity weights align correctly with the tickers
risk_parity_weights = np.append(risk_parity_weights, [0])  # Append 0 for FTSE as it's not part of the optimization

# Plotting
fig, ax = plt.subplots(1, 2, figsize=(14, 7))

# CAPM Allocation
ax[0].bar(capm_returns.index, capm_returns.values)
ax[0].set_title('CAPM Allocation')
ax[0].set_ylabel('Weights')
ax[0].set_xlabel('Assets')

# Risk Parity Allocation
ax[1].bar(tickers, risk_parity_weights)
ax[1].set_title('Risk Parity Allocation')
ax[1].set_ylabel('Weights')
ax[1].set_xlabel('Assets')

plt.tight_layout()
plt.show()



# In[42]:


# Define the tickers of the stocks and the market index (e.g., FTSE)
tickers = ['AZN', 'BARC.L', 'BP', 'HSBC', 'LYG', 'NWG', 'SHEL.L', '^FTSE']

# Download historical data
data = yf.download(tickers, start='2010-01-01', end='2024-06-30')['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Separate market returns (FTSE)
market_returns = returns['^FTSE']
# Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Calculate betas
betas = {}
for ticker in tickers:
    if ticker != '^FTSE':
        cov = np.cov(returns[ticker], market_returns)
        beta = cov[0, 1] / cov[1, 1]
        betas[ticker] = beta

# Expected market return
market_return = market_returns.mean()

# Risk-free rate (UK risk-free rate = 4.3%)
risk_free_rate = 0.043  # 4.3% annual rate

# Calculate expected returns using CAPM
capm_returns = {}
for ticker, beta in betas.items():
    capm_returns[ticker] = risk_free_rate + beta * (market_return - risk_free_rate)

capm_returns = pd.Series(capm_returns)
# Plot CAPM and Market Line (CML)
plt.figure(figsize=(10, 6))

# Plot individual assets (CAPM)
plt.scatter(betas.values(), capm_returns.values, marker='o', label='Assets')

# Plot market portfolio (FTSE)
plt.scatter(1, market_return, marker='o', color='red', label='Market (FTSE)')

# Plot Capital Market Line (CML)
x_cml = np.linspace(0, max(betas.values()) * 1.1, 100)
y_cml = risk_free_rate + x_cml * (market_return - risk_free_rate)
plt.plot(x_cml, y_cml, color='blue', linestyle='-', linewidth=2, label='Capital Market Line')

# Point of tangency (market portfolio)
plt.scatter(1, market_return, marker='*', color='red', s=200, label='Tangency Portfolio')

# Annotate tangency portfolio
plt.annotate('Tangency Portfolio', xy=(1, market_return), xytext=(1.2, market_return + 0.005),
             arrowprops=dict(facecolor='black', arrowstyle='->'), fontsize=12)

# Add labels and title
plt.xlabel('Beta')
plt.ylabel('Expected Return')
plt.title('CAPM and Capital Market Line (CML)')
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.show()


# In[44]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize
# Define the tickers of the stocks and the market index (e.g., FTSE)
tickers = ['AZN', 'BARC.L', 'BP', 'HSBC', 'LYG', 'NWG', 'SHEL.L', '^FTSE']

# Download historical data
data = yf.download(tickers, start='2010-01-01', end='2024-06-30')['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Separate market returns (FTSE)
market_returns = returns['^FTSE']

# Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

# Calculate betas
betas = {}
for ticker in tickers:
    if ticker != '^FTSE':
        cov = np.cov(returns[ticker], market_returns)
        beta = cov[0, 1] / cov[1, 1]
        betas[ticker] = beta

# Expected market return
market_return = market_returns.mean()

# Risk-free rate (UK risk-free rate = 4.3%)
risk_free_rate = 0.043  # 4.3% annual rate
def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def capm_objective(weights, capm_returns, target_return):
    expected_return = np.sum(capm_returns * weights)
    return portfolio_variance(weights, cov_matrix.iloc[:-1, :-1]) + np.abs(expected_return - target_return) * 1000
# Initial guess for the weights
initial_weights = np.ones(len(tickers) - 1) / (len(tickers) - 1)

# Constraints: weights sum to 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Bounds for the weights
bounds = tuple((0, 1) for _ in range(len(tickers) - 1))

# Optimization for CAPM Efficient Frontier
target_return = 0.10  # 10% return target
capm_allocation = minimize(capm_objective, initial_weights, args=(capm_returns.values, target_return),
                           method='SLSQP', bounds=bounds, constraints=constraints)

capm_weights = capm_allocation.x
# Display results
print("CAPM Efficient Portfolio Allocation for Target Return of {:.1f}%:".format(target_return * 100))
for i, ticker in enumerate(tickers[:-1]):
    print("{}: {:.2f}%".format(ticker, capm_weights[i] * 100))

expected_return = np.sum(capm_returns * capm_weights)
print("\nExpected Return: {:.2f}%".format(expected_return * 100))
print("Portfolio Variance: {:.6f}".format(portfolio_variance(capm_weights, cov_matrix.iloc[:-1, :-1])))


# In[45]:


def calculate_portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def calculate_risk_contributions(weights, cov_matrix):
    portfolio_var = calculate_portfolio_variance(weights, cov_matrix)
    marginal_contrib = np.dot(cov_matrix, weights)
    risk_contrib = weights * marginal_contrib / portfolio_var
    return risk_contrib

def risk_parity_objective(weights, cov_matrix):
    risk_contrib = calculate_risk_contributions(weights, cov_matrix)
    risk_diffs = risk_contrib - np.mean(risk_contrib)
    return np.sum(risk_diffs**2)

# Initial guess for the weights
initial_weights = np.ones(len(tickers) - 1) / (len(tickers) - 1)

# Constraints: weights sum to 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Bounds for the weights
bounds = tuple((0, 1) for _ in range(len(tickers) - 1))

# Optimization for Risk Parity
risk_parity_result = minimize(risk_parity_objective, initial_weights, args=cov_matrix.iloc[:-1, :-1],
                              method='SLSQP', bounds=bounds, constraints=constraints)

risk_parity_weights = risk_parity_result.x

# Ensure the Risk Parity weights align correctly with the tickers
risk_parity_weights = np.append(risk_parity_weights, [0])  # Append 0 for FTSE as it's not part of the optimization
def efficient_frontier(capm_returns, cov_matrix, risk_free_rate):
    def capm_objective(weights, capm_returns, target_return):
        expected_return = np.sum(capm_returns * weights)
        return portfolio_variance(weights, cov_matrix.iloc[:-1, :-1]) + np.abs(expected_return - target_return) * 1000

    # Range of target returns for the efficient frontier
    target_returns = np.linspace(0, 0.3, 100)  # Adjust the range as needed

    efficient_portfolios = []
    for target_return in target_returns:
        capm_allocation = minimize(capm_objective, initial_weights, args=(capm_returns.values, target_return),
                                   method='SLSQP', bounds=bounds, constraints=constraints)
        efficient_portfolios.append(capm_allocation.x)

    return efficient_portfolios, target_returns

# Calculate CAPM returns for all assets except the market index
capm_returns = pd.Series({ticker: risk_free_rate + betas[ticker] * (market_return - risk_free_rate)
                          for ticker in tickers[:-1]})

# Calculate efficient frontier portfolios
efficient_portfolios, target_returns = efficient_frontier(capm_returns, cov_matrix, risk_free_rate)
# Calculate volatility for each efficient portfolio
volatilities = []
for weights in efficient_portfolios:
    volatility = np.sqrt(portfolio_variance(weights, cov_matrix.iloc[:-1, :-1]))
    volatilities.append(volatility)

# Plot efficient frontier
plt.figure(figsize=(10, 6))
plt.scatter(volatilities, target_returns, c=target_returns, cmap='viridis', marker='o')
plt.title('Efficient Portfolio Frontier for Risk Parity Allocation')
plt.xlabel('Portfolio Volatility (Standard Deviation)')
plt.ylabel('Expected Return')
plt.colorbar(label='Target Return')
plt.grid(True)
plt.show()


# In[47]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from scipy.optimize import minimize

# Define the tickers of the stocks
tickers = ['AZN', 'BARC.L', 'BP', 'HSBC', 'LYG', 'NWG', 'SHEL.L', '^FTSE']

# Download historical data
data = yf.download(tickers, start='2010-01-01', end='2024-06-30')['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Separate market returns (FTSE)
market_returns = returns['^FTSE']

# Calculate mean returns and covariance matrix
mean_returns = returns.mean()
cov_matrix = returns.cov()

def calculate_portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def calculate_risk_contributions(weights, cov_matrix):
    portfolio_var = calculate_portfolio_variance(weights, cov_matrix)
    marginal_contrib = np.dot(cov_matrix, weights)
    risk_contrib = weights * marginal_contrib / portfolio_var
    return risk_contrib

def risk_parity_objective(weights, cov_matrix):
    risk_contrib = calculate_risk_contributions(weights, cov_matrix)
    risk_diffs = risk_contrib - np.mean(risk_contrib)
    return np.sum(risk_diffs**2)

# Initial guess for the weights
initial_weights = np.ones(len(tickers) - 1) / (len(tickers) - 1)

# Constraints: weights sum to 1
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Bounds for the weights
bounds = tuple((0, 1) for _ in range(len(tickers) - 1))

# Optimization for Risk Parity
risk_parity_result = minimize(risk_parity_objective, initial_weights, args=cov_matrix.iloc[:-1, :-1],
                              method='SLSQP', bounds=bounds, constraints=constraints)

risk_parity_weights = risk_parity_result.x
risk_parity_weights = np.append(risk_parity_weights, [0])  # Append 0 for FTSE as it's not part of the optimization

# Calculate CAPM returns for all assets except the market index
capm_returns = pd.Series({ticker: risk_free_rate + betas[ticker] * (market_return - risk_free_rate)
                          for ticker in tickers[:-1]})

# Define functions for efficient frontier
def portfolio_variance(weights, cov_matrix):
    return np.dot(weights.T, np.dot(cov_matrix, weights))

def capm_objective(weights, capm_returns, target_return):
    expected_return = np.sum(capm_returns * weights)
    return portfolio_variance(weights, cov_matrix.iloc[:-1, :-1]) + np.abs(expected_return - target_return) * 1000

def efficient_frontier(capm_returns, cov_matrix, risk_free_rate):
    target_returns = np.linspace(0, 0.3, 100)  # Adjust the range as needed
    efficient_portfolios = []
    for target_return in target_returns:
        capm_allocation = minimize(capm_objective, initial_weights, args=(capm_returns.values, target_return),
                                   method='SLSQP', bounds=bounds, constraints=constraints)
        efficient_portfolios.append(capm_allocation.x)
    return efficient_portfolios, target_returns

# Calculate efficient frontier portfolios
efficient_portfolios, target_returns = efficient_frontier(capm_returns, cov_matrix, risk_free_rate)

# Calculate volatilities for each efficient portfolio
volatilities = []
for weights in efficient_portfolios:
    volatility = np.sqrt(portfolio_variance(weights, cov_matrix.iloc[:-1, :-1]))
    volatilities.append(volatility)

# Plotting the efficient portfolio frontier
plt.figure(figsize=(10, 6))
plt.scatter(volatilities, target_returns, c=target_returns, cmap='viridis', marker='o', label='Efficient Frontier')

# Annotate each point with stock names
for i, ticker in enumerate(tickers[:-1]):
    plt.annotate(ticker, (volatilities[i], target_returns[i]), textcoords="offset points", xytext=(5,-5), ha='center')

plt.colorbar(label='Target Return')
plt.title('Efficient Portfolio Frontier for Risk Parity Allocation')
plt.xlabel('Portfolio Volatility (Standard Deviation)')
plt.ylabel('Expected Return')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[49]:


import numpy as np
import pandas as pd
import yfinance as yf

# Define the tickers of the stocks and the market index (e.g., FTSE)
tickers = ['AZN', 'BARC.L', 'BP', 'HSBC', 'LYG', 'NWG', 'SHEL.L', '^FTSE']

# Download historical data from Yahoo Finance
data = yf.download(tickers, start='2010-01-01', end='2024-06-30')['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate daily volatility (standard deviation of daily returns)
daily_volatility = returns.std()

# Calculate daily variance (square of daily volatility)
daily_variance = daily_volatility ** 2

# Calculate volatility of volatility (standard deviation of daily variances)
vol_of_vol = daily_variance.std()

# Display the results
print("Daily Volatility:")
print(daily_volatility)
print("\nDaily Variance:")
print(daily_variance)
print("\nVolatility of Volatility (Vol of Vol):")
print(vol_of_vol)



# In[50]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Define the tickers of the stocks and the market index (e.g., FTSE)
tickers = ['AZN', 'BARC.L', 'BP', 'HSBC', 'LYG', 'NWG', 'SHEL.L', '^FTSE']

# Download historical data from Yahoo Finance
data = yf.download(tickers, start='2010-01-01', end='2024-06-30')['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate daily volatility (standard deviation of daily returns)
daily_volatility = returns.std()

# Calculate daily variance (square of daily volatility)
daily_variance = daily_volatility ** 2

# Calculate volatility of volatility (standard deviation of daily variances)
vol_of_vol = daily_variance.std()

# Plotting the Volatility of Volatility
fig, ax = plt.subplots(figsize=(10, 6))

# Plotting bars for each ticker
ax.bar(daily_variance.index, daily_variance, label='Daily Variance')

# Plotting the Volatility of Volatility
ax.axhline(vol_of_vol, color='r', linestyle='--', linewidth=2, label='Volatility of Volatility')

# Formatting
ax.set_xlabel('Ticker')
ax.set_ylabel('Volatility')
ax.set_title('Volatility of Volatility (Vol of Vol) for Stocks and FTSE')
ax.legend()

plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[61]:


import numpy as np
import pandas as pd
import yfinance as yf

# Define the tickers of the stocks and the market index (e.g., FTSE)
tickers = ['AZN', 'BARC.L', 'BP', 'HSBC', 'LYG', 'NWG', 'SHEL.L', '^FTSE']

# Download historical data from Yahoo Finance
data = yf.download(tickers, start='2010-01-01', end='2024-06-30')['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate daily volatility (standard deviation of daily returns)
daily_volatility = returns.std()

# Annualize daily volatility
trading_days_per_year = 252
annual_volatility = daily_volatility * np.sqrt(trading_days_per_year)

# Calculate volatility of volatility (vol of vol)
vol_of_vol = annual_volatility.std()

# Print results
print(f"Daily Volatility (Standard Deviation of Daily Returns):\n{daily_volatility}")
print(f"\nAnnualized Volatility (Standard Deviation of Annual Returns):\n{annual_volatility}")
print(f"\nVolatility of Volatility (Vol of Vol):\n{vol_of_vol}")


# In[62]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Define the tickers of the stocks and the market index (e.g., FTSE)
tickers = ['AZN', 'BARC.L', 'BP', 'HSBC', 'LYG', 'NWG', 'SHEL.L', '^FTSE']

# Download historical data from Yahoo Finance
data = yf.download(tickers, start='2010-01-01', end='2024-06-30')['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate daily volatility (standard deviation of daily returns)
daily_volatility = returns.std()

# Annualize daily volatility
trading_days_per_year = 252
annual_volatility = daily_volatility * np.sqrt(trading_days_per_year)

# Calculate volatility of volatility (vol of vol)
vol_of_vol = annual_volatility.std()

# Plotting
plt.figure(figsize=(12, 6))

# Plot Daily Volatility
plt.plot(daily_volatility.index, daily_volatility.values, label='Daily Volatility', marker='o')

# Plot Annualized Volatility
plt.plot(annual_volatility.index, annual_volatility.values, label='Annualized Volatility', marker='o')

# Plot Volatility of Volatility (Vol of Vol)
plt.axhline(vol_of_vol, color='r', linestyle='--', label='Volatility of Volatility (Vol of Vol)')

# Formatting
plt.title('Volatility Measures')
plt.xlabel('Stocks')
plt.ylabel('Volatility')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.show()



# In[63]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Define the tickers of the stocks and the market index (e.g., FTSE)
tickers = ['AZN', 'BARC.L', 'BP', 'HSBC', 'LYG', 'NWG', 'SHEL.L', '^FTSE']

# Download historical data from Yahoo Finance
data = yf.download(tickers, start='2010-01-01', end='2024-06-30')['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate rolling 21-day standard deviation of daily returns
rolling_volatility_21d = returns.rolling(window=21).std()

# Annualize volatility by multiplying by the square root of 252
annualized_volatility_21d = rolling_volatility_21d * np.sqrt(252)

# Plotting
plt.figure(figsize=(12, 6))

# Plot Annualized Volatility
for ticker in tickers:
    plt.plot(annualized_volatility_21d.index, annualized_volatility_21d[ticker], label=ticker)

# Formatting
plt.title('Annualized Volatility (21-day rolling window)')
plt.xlabel('Date')
plt.ylabel('Annualized Volatility')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.show()



# In[64]:


import numpy as np
import pandas as pd
import yfinance as yf

# Define the tickers of the stocks and the market index (e.g., FTSE)
tickers = ['AZN', 'BARC.L', 'BP', 'HSBC', 'LYG', 'NWG', 'SHEL.L', '^FTSE']

# Download historical data from Yahoo Finance
data = yf.download(tickers, start='2010-01-01', end='2024-06-30')['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate daily volatility (standard deviation of daily returns)
daily_volatility = returns.std()

# Annualize daily volatility
trading_days_per_year = 252
annual_volatility = daily_volatility * np.sqrt(trading_days_per_year)

# Calculate volatility of volatility (vol of vol)
vol_of_vol = annual_volatility.std()

# Print results
print(f"Daily Volatility (Standard Deviation of Daily Returns):\n{daily_volatility}")
print(f"\nAnnualized Volatility (Standard Deviation of Annual Returns):\n{annual_volatility}")
print(f"\nVolatility of Volatility (Vol of Vol):\n{vol_of_vol}")


# In[66]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Define the tickers of the stocks and the market index (e.g., FTSE)
tickers = ['AZN', 'BARC.L', 'BP', 'HSBC', 'LYG', 'NWG', 'SHEL.L', '^FTSE']

# Download historical data from Yahoo Finance
data = yf.download(tickers, start='2010-01-01', end='2024-06-30')['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate historical volatility (21 days)
historical_volatility = returns.rolling(window=21).std()

# Annualize historical volatility
trading_days_per_year = 252
annualized_volatility = historical_volatility * np.sqrt(trading_days_per_year)

# Calculate volatility of volatility (vol of vol)
vol_of_vol = annualized_volatility.std()

# Plotting
plt.figure(figsize=(12, 6))

# Plot Historical Volatility (21 days)
for ticker in tickers:
    plt.plot(historical_volatility.index, historical_volatility[ticker], label=f'{ticker} - Historical Volatility (21 days)', alpha=0.8)

# Plot Annualized Volatility
for ticker in tickers:
    plt.plot(annualized_volatility.index, annualized_volatility[ticker], label=f'{ticker} - Annualized Volatility', linestyle='--', alpha=0.8)

# Plot Volatility of Volatility (Vol of Vol)
plt.axhline(vol_of_vol.mean(), color='r', linestyle='-', label='Mean Volatility of Volatility (Vol of Vol)')
plt.fill_between(vol_of_vol.index, vol_of_vol.min(), vol_of_vol.max(), color='r', alpha=0.1)

# Formatting
plt.title('Volatility Measures')
plt.xlabel('Date')
plt.ylabel('Volatility')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()

plt.show()

# Print Vol of Vol
print(f"Volatility of Volatility (Vol of Vol): {vol_of_vol.mean()}")


# In[73]:


import pandas as pd
import numpy as np
from scipy.stats import mode
import yfinance as yf

# Define the tickers of the stocks and the market index (e.g., FTSE)
tickers = ['^FTSE', 'BP', 'SHEL.L', 'AZN', 'BARC.L', 'LYG', 'NWG', 'HSBC']
start_date = '2010-01-01'
end_date = '2024-06-30'  # 'today' will automatically be interpreted as today's date

# Download historical data from Yahoo Finance
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Initialize a DataFrame to store results
statistics = pd.DataFrame(index=['Mean', 'Median', 'Mode', 'Daily Variance', 'Daily Standard Deviation',
                                 'Annualized Volatility', 'Historical Volatility', 'Skewness', 'Kurtosis'])

# Calculate statistics for each ticker
for ticker in tickers:
    # Calculate mean
    mean = returns[ticker].mean()
    
    # Calculate median
    median = returns[ticker].median()
    
    
    
    # Calculate daily variance
    variance = returns[ticker].var()
    
    # Calculate daily standard deviation
    std_dev = returns[ticker].std()
    
    # Calculate daily returns
    daily_returns = returns[ticker]
    
    # Calculate annualized volatility (assuming 252 trading days in a year)
    annual_volatility = std_dev * np.sqrt(252)
    
    # Calculate historical volatility (standard deviation of daily returns)
    historical_volatility = daily_returns.std()
    
    # Calculate skewness
    skewness = daily_returns.skew()
    
    # Calculate kurtosis
    kurtosis = daily_returns.kurtosis()
    
    # Store results in DataFrame
    statistics[ticker] = [mean, median, mode_result, variance, std_dev, annual_volatility,
                          historical_volatility, skewness, kurtosis]

# Print statistics
print(statistics)


# In[72]:


import pandas as pd
import numpy as np
from scipy.stats import mode
import yfinance as yf

# Define the tickers of the stocks and the market index (e.g., FTSE)
tickers = ['^FTSE', 'BP', 'SHEL.L', 'AZN', 'BARC.L', 'LYG', 'NWG', 'HSBC']
start_date = '2010-01-01'
end_date = '2024-06-30'  # 'today' will automatically be interpreted as today's date

# Download historical data from Yahoo Finance
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Initialize a DataFrame to store results
statistics = pd.DataFrame(index=['Mean', 'Median', 'Mode', 'Daily Variance', 'Daily Standard Deviation',
                                 'Annualized Volatility', 'Historical Volatility', 'Skewness', 'Kurtosis'])

# Calculate statistics for each ticker
for ticker in tickers:
    # Calculate mean
    mean = returns[ticker].mean()
    
    # Calculate median
    median = returns[ticker].median()
    
  
    
    # Calculate daily variance
    variance = returns[ticker].var()
    
    # Calculate daily standard deviation
    std_dev = returns[ticker].std()
    
    # Calculate daily returns
    daily_returns = returns[ticker]
    
    # Calculate annualized volatility (assuming 252 trading days in a year)
    annual_volatility = std_dev * np.sqrt(252)
    
    # Calculate historical volatility (standard deviation of daily returns)
    historical_volatility = daily_returns.std()
    
    # Calculate skewness
    skewness = daily_returns.skew()
    
    # Calculate kurtosis
    kurtosis = daily_returns.kurtosis()
    
    # Store results in DataFrame
    statistics[ticker] = [mean, median, mode_result, variance, std_dev, annual_volatility,
                          historical_volatility, skewness, kurtosis]

# Print statistics
print(statistics)


# In[74]:


import pandas as pd
import numpy as np
from scipy.stats import skew, kurtosis
import yfinance as yf

# Define the tickers of the stocks and the market index (e.g., FTSE)
tickers = ['^FTSE', 'BP', 'SHEL.L', 'AZN', 'BARC.L', 'LYG', 'NWG', 'HSBC']
start_date = '2010-01-01'
end_date = '2024-06-30'  # 'today' will automatically be interpreted as today's date

# Download historical data from Yahoo Finance
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Initialize a DataFrame to store results
statistics = pd.DataFrame(index=['Mean', 'Median', 'Daily Variance', 'Daily Standard Deviation',
                                 'Annualized Volatility', 'Historical Volatility', 'Skewness', 'Kurtosis'])

# Calculate statistics for each ticker
for ticker in tickers:
    # Calculate mean
    mean = returns[ticker].mean()
    
    # Calculate median
    median = returns[ticker].median()
    
    # Calculate daily variance
    variance = returns[ticker].var()
    
    # Calculate daily standard deviation
    std_dev = returns[ticker].std()
    
    # Calculate annualized volatility (assuming 252 trading days in a year)
    annual_volatility = std_dev * np.sqrt(252)
    
    # Calculate historical volatility (standard deviation of daily returns)
    historical_volatility = std_dev
    
    # Calculate skewness
    skewness_value = skew(returns[ticker])
    
    # Calculate kurtosis
    kurtosis_value = kurtosis(returns[ticker])
    
    # Store results in DataFrame
    statistics[ticker] = [mean, median, variance, std_dev, annual_volatility,
                          historical_volatility, skewness_value, kurtosis_value]

# Print statistics
print("Descriptive Statistics for Stock Tickers:")
print(statistics)


# In[77]:


import pandas as pd
import yfinance as yf

# Define the tickers of the stocks and the market index (e.g., FTSE)
tickers = ['^FTSE', 'BP', 'SHEL.L', 'AZN', 'BARC.L', 'LYG', 'NWG', 'HSBC']
start_date = '2010-01-01'
end_date = '2024-06-30'  # 'today' will automatically be interpreted as today's date

# Download historical data from Yahoo Finance
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate mean and median for each ticker
statistics = pd.DataFrame(index=['Mean', 'Median'])

for ticker in tickers:
    # Calculate mean
    mean_price = data[ticker].mean()
    
    # Calculate median
    median_price = data[ticker].median()
    
    # Store results in DataFrame
    statistics[ticker] = [mean_price, median_price]

# Print statistics
print("Mean and Median of Stock Prices:")
print(statistics)


# In[78]:


import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Define the tickers of the stocks and the market index (e.g., FTSE)
tickers = ['^FTSE', 'BP', 'SHEL.L', 'AZN', 'BARC.L', 'LYG', 'NWG', 'HSBC']
start_date = '2010-01-01'
end_date = '2024-06-30'  # 'today' will automatically be interpreted as today's date

# Download historical data from Yahoo Finance
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate annualized volatility (assuming 252 trading days in a year)
daily_returns = data.pct_change().dropna()

trading_days_per_year = 252
annual_volatility = daily_returns.std() * np.sqrt(trading_days_per_year)

# Plotting
plt.figure(figsize=(12, 6))

# Plot Annualized Volatility
plt.bar(annual_volatility.index, annual_volatility.values, color='blue')

# Formatting
plt.title('Annualized Volatility of Stocks and Index')
plt.xlabel('Ticker')
plt.ylabel('Annualized Volatility')
plt.xticks(rotation=45)
plt.grid(True)
plt.tight_layout()

plt.show()


# In[79]:


import pandas as pd
import yfinance as yf

# Define the tickers of the stocks and the market index (e.g., FTSE)
tickers = ['^FTSE', 'BP', 'SHEL.L', 'AZN', 'BARC.L', 'LYG', 'NWG', 'HSBC']
start_date = '2010-01-01'
end_date = '2024-06-30'

# Download historical data from Yahoo Finance
data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']

# Calculate daily returns
returns = data.pct_change().dropna()

# Calculate historical volatility (standard deviation of daily returns)
historical_volatility = returns.std()

# Calculate average historical volatility
average_historical_volatility = historical_volatility.mean()

print(f"Average Historical Volatility: {average_historical_volatility:.6f}")


# In[ ]:




