#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats

# Define the tickers and date range
tickers = ["MB.MI", "ISP.MI", "UCG.MI", "SR2000.MI", "XEON.MI", "3ITL.MI", "XFFE.MI", "FTSEMIB.MI", "FBK.MI", "IF.MI", "BGN.MI"]
start_date = "2005-01-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

# Function to download data for a list of tickers
def download_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = yf.download(ticker, start=start_date, end=end_date)
        except Exception as e:
            print(f"Error downloading data for {ticker}: {str(e)}")
    return data

# Function to calculate historical volatility
def calculate_volatility(data, window):
    # Calculate daily returns
    returns = data['Adj Close'].pct_change().dropna()
    
    # Calculate rolling standard deviation
    rolling_sd = returns.rolling(window=window).std()
    
    # Annualize volatility
    annualized_volatility = rolling_sd * (252**0.5)
    
    return annualized_volatility

# Function to plot PDF and PMF for historical volatility
def plot_pdf_pmf(ticker, volatility, window):
    # Plot PDF
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.hist(volatility, bins=30, density=True, alpha=0.6, color='g', edgecolor='black')
    plt.title(f'{ticker} - {window}-day Annualized Historical Volatility PDF')
    plt.xlabel('Volatility')
    plt.ylabel('Density')
    plt.grid(True)
    
    # Plot PMF (using Poisson approximation for discrete distribution)
    if window == 5:
        lambda_ = volatility.mean()
        poisson_dist = stats.poisson(mu=lambda_)
        x = range(0, int(max(volatility))+1)
        pmf_values = poisson_dist.pmf(x)
        
        plt.subplot(2, 1, 2)
        plt.plot(x, pmf_values, 'bo-', markersize=5, label='PMF')
        plt.title(f'{ticker} - {window}-day Annualized Historical Volatility PMF')
        plt.xlabel('Volatility')
        plt.ylabel('Probability')
        plt.grid(True)
        plt.tight_layout()

# Download data
data = download_data(tickers, start_date, end_date)

# Calculate and plot PDF and PMF for each ticker
for ticker, df in data.items():
    if not df.empty:
        try:
            # Calculate 5-day and 21-day annualized volatility
            vol_5d = calculate_volatility(df, 5)
            vol_21d = calculate_volatility(df, 21)
            
            # Plot PDF and PMF
            plot_pdf_pmf(ticker, vol_5d, 5)
            plot_pdf_pmf(ticker, vol_21d, 21)
            
            plt.show()  # Show plots for each ticker
        except Exception as e:
            print(f"Error processing data for {ticker}: {str(e)}")
    else:
        print(f"No data available for {ticker}")



# In[2]:


import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import scipy.stats as stats

# Define the tickers and date range
tickers = ["MB.MI", "ISP.MI", "UCG.MI", "SR2000.MI", "XEON.MI", "3ITL.MI", "XFFE.MI", "FTSEMIB.MI", "FBK.MI", "IF.MI", "BGN.MI"]
start_date = "2005-01-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

# Function to download data for a list of tickers
def download_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = yf.download(ticker, start=start_date, end=end_date)
        except Exception as e:
            print(f"Error downloading data for {ticker}: {str(e)}")
    return data

# Function to calculate historical volatility
def calculate_volatility(data, window):
    try:
        # Calculate daily returns
        returns = data['Adj Close'].pct_change()
        
        # Remove rows with NaNs
        returns.dropna(inplace=True)
        
        # Calculate rolling standard deviation
        rolling_sd = returns.rolling(window=window).std()
        
        # Annualize volatility
        annualized_volatility = rolling_sd * (252**0.5)
        
        return annualized_volatility
    
    except Exception as e:
        print(f"Error calculating volatility: {str(e)}")
        return None

# Function to plot PDF and PMF for historical volatility
def plot_pdf_pmf(ticker, volatility, window):
    if volatility is not None:
        # Plot PDF
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.hist(volatility, bins=30, density=True, alpha=0.6, color='g', edgecolor='black')
        plt.title(f'{ticker} - {window}-day Annualized Historical Volatility PDF')
        plt.xlabel('Volatility')
        plt.ylabel('Density')
        plt.grid(True)
        
        # Plot PMF (using Poisson approximation for discrete distribution)
        if window == 5:
            lambda_ = volatility.mean()
            poisson_dist = stats.poisson(mu=lambda_)
            x = range(0, int(max(volatility))+1)
            pmf_values = poisson_dist.pmf(x)
            
            plt.subplot(2, 1, 2)
            plt.plot(x, pmf_values, 'bo-', markersize=5, label='PMF')
            plt.title(f'{ticker} - {window}-day Annualized Historical Volatility PMF')
            plt.xlabel('Volatility')
            plt.ylabel('Probability')
            plt.grid(True)
            plt.tight_layout()
        
        plt.show()
    else:
        print(f"No volatility data available for {ticker}")

# Download data
data = download_data(tickers, start_date, end_date)

# Calculate and plot PDF and PMF for each ticker
for ticker, df in data.items():
    if not df.empty:
        try:
            # Calculate 5-day and 21-day annualized volatility
            vol_5d = calculate_volatility(df, 5)
            vol_21d = calculate_volatility(df, 21)
            
            # Plot PDF and PMF
            plot_pdf_pmf(ticker, vol_5d, 5)
            plot_pdf_pmf(ticker, vol_21d, 21)
            
        except Exception as e:
            print(f"Error processing data for {ticker}: {str(e)}")
    else:
        print(f"No data available for {ticker}")


# In[13]:


import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Define the tickers and date range
tickers = ["MB.MI", "ISP.MI", "UCG.MI", "SR2000.MI", "XEON.MI", "3ITL.MI", "XFFE.MI", "FTSEMIB.MI", "FBK.MI", "IF.MI", "BGN.MI"]
start_date = "2005-01-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

# Function to download data for a list of tickers
def download_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = yf.download(ticker, start=start_date, end=end_date)
        except Exception as e:
            print(f"Error downloading data for {ticker}: {str(e)}")
    return data

# Function to calculate historical volatility
def calculate_volatility(data, window):
    try:
        # Calculate daily returns
        returns = data['Adj Close'].pct_change()
        
        # Calculate rolling standard deviation
        rolling_sd = returns.rolling(window=window).std()
        
        # Annualize volatility
        annualized_volatility = rolling_sd * (252**0.5)
        
        return annualized_volatility
    
    except Exception as e:
        print(f"Error calculating volatility: {str(e)}")
        return None

# Download data
data = download_data(tickers, start_date, end_date)

# Calculate and plot historical volatility for each ticker
for ticker, df in data.items():
    if not df.empty:
        try:
            # Calculate 5-day and 21-day annualized volatility
            vol_5d = calculate_volatility(df, 5)
            vol_21d = calculate_volatility(df, 21)
            
            # Plotting
            plt.figure(figsize=(10, 6))
            
            # Plot 5-day volatility
            plt.subplot(2, 1, 1)
            plt.plot(vol_5d, label='5-day Volatility', color='blue')
            plt.title(f'{ticker} - 5-day Annualized Historical Volatility')
            plt.xlabel('Date')
            plt.ylabel('Volatility')
            plt.legend()
            plt.grid(True)
            
            # Plot 21-day volatility
            plt.subplot(2, 1, 2)
            plt.plot(vol_21d, label='21-day Volatility', color='red')
            plt.title(f'{ticker} - 21-day Annualized Historical Volatility')
            plt.xlabel('Date')
            plt.ylabel('Volatility')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            print(f"Error processing data for {ticker}: {str(e)}")
    else:
        print(f"No data available for {ticker}")


# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf

# Define the tickers and date range
tickers = ["MB.MI", "ISP.MI", "UCG.MI", "SR2000.MI", "XEON.MI", "3ITL.MI", "XFFE.MI", "FTSEMIB.MI", "FBK.MI", "IF.MI", "BGN.MI"]
start_date = "2005-01-01"
end_date = "2024-07-06"  # Update end date as needed

# Function to download historical data for a ticker
def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Adj Close'].pct_change().dropna()  # Calculate daily returns

# Bayesian volatility forecasting function
def bayesian_volatility_forecasting(returns):
    # Prior distribution parameters (normal distribution for simplicity)
    prior_mu = 0.02  # Prior mean
    prior_sigma = 0.05  # Prior standard deviation
    
    # Likelihood function parameters (normal distribution)
    likelihood_sigma = np.std(returns)  # Standard deviation of returns as likelihood parameter
    
    # Prior distribution
    prior_distribution = norm(loc=prior_mu, scale=prior_sigma)
    
    # Posterior distribution parameters
    posterior_mu = (prior_mu / prior_sigma**2 + np.sum(returns) / likelihood_sigma**2) / (1 / prior_sigma**2 + len(returns) / likelihood_sigma**2)
    posterior_sigma = np.sqrt(1 / (1 / prior_sigma**2 + len(returns) / likelihood_sigma**2))
    
    # Posterior distribution
    posterior_distribution = norm(loc=posterior_mu, scale=posterior_sigma)
    
    # Parameter estimation (mean of the posterior distribution)
    posterior_mean = posterior_mu
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Prior distribution plot
    x = np.linspace(prior_distribution.ppf(0.001), prior_distribution.ppf(0.999), 100)
    plt.plot(x, prior_distribution.pdf(x), 'g-', lw=2, alpha=0.6, label='Prior Distribution')
    
    # Posterior distribution plot
    x = np.linspace(posterior_distribution.ppf(0.001), posterior_distribution.ppf(0.999), 100)
    plt.plot(x, posterior_distribution.pdf(x), 'b-', lw=2, alpha=0.6, label='Posterior Distribution')
    
    plt.title(f'Bayesian Volatility Forecasting - Ticker: {ticker}')
    plt.xlabel('Volatility')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print summary statistics
    print(f"Ticker: {ticker}")
    print(f"Posterior mean volatility: {posterior_mean}")
    print(f"Posterior 95% credible interval: {posterior_distribution.interval(0.95)}")

# Iterate over each ticker, download data, and perform Bayesian volatility forecasting
for ticker in tickers:
    try:
        # Download data for the ticker
        returns = download_data(ticker, start_date, end_date)
        
        # Perform Bayesian volatility forecasting
        bayesian_volatility_forecasting(returns)
    
    except Exception as e:
        print(f"Error processing data for {ticker}: {str(e)}")


# In[3]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf

# Define the tickers and date range
tickers = ["MB.MI", "ISP.MI", "UCG.MI", "SR2000.MI", "XEON.MI", "3ITL.MI", "XFFE.MI", "FTSEMIB.MI", "FBK.MI", "IF.MI", "BGN.MI"]
start_date = "2005-01-01"
end_date = "2024-07-06"  # Update end date as needed

# Function to download historical data for a ticker
def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Adj Close'].pct_change().dropna()  # Calculate daily returns

# Function to calculate prior mean and standard deviation
def calculate_prior_parameters(returns):
    prior_mean = np.mean(returns)  # Prior mean as the mean of historical returns
    prior_std = np.std(returns)    # Prior standard deviation as the standard deviation of historical returns
    return prior_mean, prior_std

# Iterate over each ticker, download data, calculate prior parameters, and print
for ticker in tickers:
    try:
        # Download data for the ticker
        returns = download_data(ticker, start_date, end_date)
        
        # Calculate prior parameters
        prior_mean, prior_std = calculate_prior_parameters(returns)
        
        # Print results
        print(f"Ticker: {ticker}")
        print(f"Prior Mean: {prior_mean}")
        print(f"Prior Standard Deviation: {prior_std}")
        print("-" * 50)
    
    except Exception as e:
        print(f"Error processing data for {ticker}: {str(e)}")


# In[8]:


pip install QuantLib-Python


# In[4]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf

# Define the tickers and date range
tickers = ["MB.MI", "ISP.MI", "UCG.MI", "SR2000.MI", "XEON.MI", "3ITL.MI", "XFFE.MI", "FTSEMIB.MI", "FBK.MI", "IF.MI", "BGN.MI"]
start_date = "2005-01-01"
end_date = "2024-07-06"  # Update end date as needed

# Function to download historical data for a ticker
def download_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data['Adj Close'].pct_change().dropna()  # Calculate daily returns

# Function to calculate prior mean and standard deviation
def calculate_prior_parameters(returns):
    prior_mean = np.mean(returns)  # Prior mean as the mean of historical returns
    prior_std = np.std(returns)    # Prior standard deviation as the standard deviation of historical returns
    return prior_mean, prior_std

# Bayesian volatility forecasting function
def bayesian_volatility_forecasting(ticker, returns, prior_mu, prior_sigma):
    # Likelihood function parameters (normal distribution)
    likelihood_sigma = np.std(returns)  # Standard deviation of returns as likelihood parameter
    
    # Prior distribution
    prior_distribution = norm(loc=prior_mu, scale=prior_sigma)
    
    # Posterior distribution parameters
    posterior_mu = (prior_mu / prior_sigma**2 + np.sum(returns) / likelihood_sigma**2) / (1 / prior_sigma**2 + len(returns) / likelihood_sigma**2)
    posterior_sigma = np.sqrt(1 / (1 / prior_sigma**2 + len(returns) / likelihood_sigma**2))
    
    # Posterior distribution
    posterior_distribution = norm(loc=posterior_mu, scale=posterior_sigma)
    
    # Parameter estimation (mean of the posterior distribution)
    posterior_mean = posterior_mu
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Prior distribution plot
    x = np.linspace(prior_distribution.ppf(0.001), prior_distribution.ppf(0.999), 100)
    plt.plot(x, prior_distribution.pdf(x), 'g-', lw=2, alpha=0.6, label='Prior Distribution')
    
    # Posterior distribution plot
    x = np.linspace(posterior_distribution.ppf(0.001), posterior_distribution.ppf(0.999), 100)
    plt.plot(x, posterior_distribution.pdf(x), 'b-', lw=2, alpha=0.6, label='Posterior Distribution')
    
    plt.title(f'Bayesian Volatility Forecasting - Ticker: {ticker}')
    plt.xlabel('Volatility')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print summary statistics
    print(f"Ticker: {ticker}")
    print(f"Posterior mean volatility: {posterior_mean}")
    print(f"Posterior 95% credible interval: {posterior_distribution.interval(0.95)}")

# Iterate over each ticker, download data, calculate prior parameters, and perform Bayesian volatility forecasting
for ticker in tickers:
    try:
        # Download data for the ticker
        returns = download_data(ticker, start_date, end_date)
        
        # Calculate prior parameters
        prior_mu, prior_sigma = calculate_prior_parameters(returns)
        
        # Perform Bayesian volatility forecasting
        bayesian_volatility_forecasting(ticker, returns, prior_mu, prior_sigma)
    
    except Exception as e:
        print(f"Error processing data for {ticker}: {str(e)}")


# In[5]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf

# Define the tickers and date range
tickers = ["MB.MI", "ISP.MI", "UCG.MI", "SR2000.MI", "XEON.MI", "3ITL.MI", "XFFE.MI", "FTSEMIB.MI", "FBK.MI", "IF.MI", "BGN.MI"]
start_date = "2005-01-01"
end_date = "2024-07-06"  # Update end date as needed

# Function to download historical data for a ticker
def download_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        returns = data['Adj Close'].pct_change().dropna()  # Calculate daily returns
        return returns
    except Exception as e:
        raise ValueError(f"Error downloading data for {ticker}: {str(e)}")

# Function to calculate prior mean and standard deviation
def calculate_prior_parameters(returns):
    prior_mean = np.mean(returns)  # Prior mean as the mean of historical returns
    prior_std = np.std(returns)    # Prior standard deviation as the standard deviation of historical returns
    return prior_mean, prior_std

# Bayesian volatility forecasting function
def bayesian_volatility_forecasting(ticker, returns, prior_mu, prior_sigma):
    # Likelihood function parameters (normal distribution)
    likelihood_sigma = np.std(returns)  # Standard deviation of returns as likelihood parameter
    
    # Prior distribution
    prior_distribution = norm(loc=prior_mu, scale=prior_sigma)
    
    # Posterior distribution parameters
    posterior_mu = (prior_mu / prior_sigma**2 + np.sum(returns) / likelihood_sigma**2) / (1 / prior_sigma**2 + len(returns) / likelihood_sigma**2)
    posterior_sigma = np.sqrt(1 / (1 / prior_sigma**2 + len(returns) / likelihood_sigma**2))
    
    # Posterior distribution
    posterior_distribution = norm(loc=posterior_mu, scale=posterior_sigma)
    
    # Parameter estimation (mean of the posterior distribution)
    posterior_mean = posterior_mu
    
    # Plotting
    plt.figure(figsize=(12, 6))
    
    # Prior distribution plot
    x = np.linspace(prior_distribution.ppf(0.001), prior_distribution.ppf(0.999), 100)
    plt.plot(x, prior_distribution.pdf(x), 'g-', lw=2, alpha=0.6, label='Prior Distribution')
    
    # Posterior distribution plot
    x = np.linspace(posterior_distribution.ppf(0.001), posterior_distribution.ppf(0.999), 100)
    plt.plot(x, posterior_distribution.pdf(x), 'b-', lw=2, alpha=0.6, label='Posterior Distribution')
    
    plt.title(f'Bayesian Volatility Forecasting - Ticker: {ticker}')
    plt.xlabel('Volatility')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print summary statistics
    print(f"Ticker: {ticker}")
    print(f"Posterior mean volatility: {posterior_mean}")
    print(f"Posterior 95% credible interval: {posterior_distribution.interval(0.95)}")

# Iterate over each ticker, download data, calculate prior parameters, and perform Bayesian volatility forecasting
for ticker in tickers:
    try:
        # Download data for the ticker
        returns = download_data(ticker, start_date, end_date)
        
        # Calculate prior parameters
        prior_mu, prior_sigma = calculate_prior_parameters(returns)
        
        # Perform Bayesian volatility forecasting
        bayesian_volatility_forecasting(ticker, returns, prior_mu, prior_sigma)
    
    except Exception as e:
        print(f"Error processing data for {ticker}: {str(e)}")


# In[6]:


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import yfinance as yf

# Define the tickers and date range
tickers = ["MB.MI", "ISP.MI", "UCG.MI", "SR2000.MI", "XEON.MI", "3ITL.MI", "XFFE.MI", "FTSEMIB.MI", "FBK.MI", "IF.MI", "BGN.MI"]
start_date = "2005-01-01"
end_date = "2024-07-06"  # Update end date as needed

# Function to download historical data for a ticker
def download_data(ticker, start_date, end_date):
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        returns = data['Adj Close'].pct_change().dropna()  # Calculate daily returns
        return returns
    except Exception as e:
        raise ValueError(f"Error downloading data for {ticker}: {str(e)}")

# Function to calculate prior mean and standard deviation
def calculate_prior_parameters(returns):
    prior_mean = np.mean(returns)  # Prior mean as the mean of historical returns
    prior_std = np.std(returns)    # Prior standard deviation as the standard deviation of historical returns
    return prior_mean, prior_std

# Function to calculate historical volatility
def calculate_historical_volatility(returns):
    historical_volatility = np.std(returns) * np.sqrt(252)  # Annualized historical volatility
    return historical_volatility

# Bayesian volatility forecasting function
def bayesian_volatility_forecasting(ticker, returns, prior_mu, prior_sigma):
    # Likelihood function parameters (normal distribution)
    likelihood_sigma = np.std(returns)  # Standard deviation of returns as likelihood parameter
    
    # Prior distribution
    prior_distribution = norm(loc=prior_mu, scale=prior_sigma)
    
    # Posterior distribution parameters
    posterior_mu = (prior_mu / prior_sigma**2 + np.sum(returns) / likelihood_sigma**2) / (1 / prior_sigma**2 + len(returns) / likelihood_sigma**2)
    posterior_sigma = np.sqrt(1 / (1 / prior_sigma**2 + len(returns) / likelihood_sigma**2))
    
    # Posterior distribution
    posterior_distribution = norm(loc=posterior_mu, scale=posterior_sigma)
    
    # Parameter estimation (mean of the posterior distribution)
    posterior_mean = posterior_mu
    
    # Plotting (optional)
    plt.figure(figsize=(12, 6))
    
    # Prior distribution plot
    x = np.linspace(prior_distribution.ppf(0.001), prior_distribution.ppf(0.999), 100)
    plt.plot(x, prior_distribution.pdf(x), 'g-', lw=2, alpha=0.6, label='Prior Distribution')
    
    # Posterior distribution plot
    x = np.linspace(posterior_distribution.ppf(0.001), posterior_distribution.ppf(0.999), 100)
    plt.plot(x, posterior_distribution.pdf(x), 'b-', lw=2, alpha=0.6, label='Posterior Distribution')
    
    plt.title(f'Bayesian Volatility Forecasting - Ticker: {ticker}')
    plt.xlabel('Volatility')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Print summary statistics
    print(f"Ticker: {ticker}")
    print(f"Bayesian Posterior mean volatility: {posterior_mean}")
    print(f"Bayesian Posterior 95% credible interval: {posterior_distribution.interval(0.95)}")
    
    # Calculate and print historical volatility
    hist_vol = calculate_historical_volatility(returns)
    print(f"Historical volatility: {hist_vol}")

# Iterate over each ticker, download data, calculate prior parameters, and perform Bayesian volatility forecasting
for ticker in tickers:
    try:
        # Download data for the ticker
        returns = download_data(ticker, start_date, end_date)
        
        # Calculate prior parameters
        prior_mu, prior_sigma = calculate_prior_parameters(returns)
        
        # Perform Bayesian volatility forecasting
        bayesian_volatility_forecasting(ticker, returns, prior_mu, prior_sigma)
    
    except Exception as e:
        print(f"Error processing data for {ticker}: {str(e)}")


# In[9]:


import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Define the tickers and date range
tickers = ["MB.MI", "ISP.MI", "UCG.MI", "SR2000.MI", "XEON.MI", "3ITL.MI", "XFFE.MI", "FTSEMIB.MI", "FBK.MI", "IF.MI", "BGN.MI"]
start_date = "2005-01-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

# Function to download data for a list of tickers
def download_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = yf.download(ticker, start=start_date, end=end_date)
        except Exception as e:
            print(f"Error downloading data for {ticker}: {str(e)}")
    return data

# Function to calculate annualized volatility
def calculate_volatility(data, window):
    try:
        # Calculate daily returns
        returns = data['Adj Close'].pct_change()
        
        # Calculate rolling standard deviation
        rolling_sd = returns.rolling(window=window).std()
        
        # Annualize volatility
        annualized_volatility = rolling_sd * (252**0.5)
        
        return annualized_volatility
    
    except Exception as e:
        print(f"Error calculating volatility: {str(e)}")
        return None

# Download data
data = download_data(tickers, start_date, end_date)

# Calculate and print average 5-day and 12-day annualized volatility for each ticker
for ticker, df in data.items():
    if not df.empty:
        try:
            # Calculate 5-day and 12-day annualized volatility
            vol_5d = calculate_volatility(df, 5)
            vol_12d = calculate_volatility(df, 12)
            
            # Calculate average annualized volatility
            avg_5d_vol = vol_5d.mean()
            avg_12d_vol = vol_12d.mean()
            
            # Print summary
            print(f"Ticker: {ticker}")
            print(f"Average 5-day Annualized Volatility: {avg_5d_vol}")
            print(f"Average 12-day Annualized Volatility: {avg_12d_vol}")
            print("=" * 50)
        
        except Exception as e:
            print(f"Error processing data for {ticker}: {str(e)}")
    else:
        print(f"No data available for {ticker}")


# In[10]:


import pandas as pd
import yfinance as yf
import scipy.stats as stats

# Define the tickers and date range
tickers = ["MB.MI", "ISP.MI", "UCG.MI", "SR2000.MI", "XEON.MI", "3ITL.MI", "XFFE.MI", "FTSEMIB.MI", "FBK.MI", "IF.MI", "BGN.MI"]
start_date = "2005-01-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

# Function to download data for a list of tickers
def download_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = yf.download(ticker, start=start_date, end=end_date)
        except Exception as e:
            print(f"Error downloading data for {ticker}: {str(e)}")
    return data

# Function to calculate descriptive statistics of daily returns
def calculate_descriptive_stats(data):
    stats_summary = {}
    for ticker, df in data.items():
        if not df.empty:
            try:
                # Calculate daily returns
                df['Daily Returns'] = df['Adj Close'].pct_change()
                
                # Drop NaN values
                df.dropna(inplace=True)
                
                # Calculate statistics
                mean_price = df['Adj Close'].mean()
                median_price = df['Adj Close'].median()
                mode_price = df['Adj Close'].mode()[0]  # Mode returns the Series, so we get the first element
                std_dev_returns = df['Daily Returns'].std()
                skewness_returns = df['Daily Returns'].skew()
                kurtosis_returns = df['Daily Returns'].kurtosis()
                
                # Store statistics in a dictionary
                stats_summary[ticker] = {
                    'Mean Price': mean_price,
                    'Median Price': median_price,
                    'Mode Price': mode_price,
                    'Std Dev Returns': std_dev_returns,
                    'Skewness Returns': skewness_returns,
                    'Kurtosis Returns': kurtosis_returns
                }
                
            except Exception as e:
                print(f"Error processing data for {ticker}: {str(e)}")
        else:
            print(f"No data available for {ticker}")
    
    return stats_summary

# Download data
data = download_data(tickers, start_date, end_date)

# Calculate descriptive statistics for each ticker
stats_summary = calculate_descriptive_stats(data)

# Print statistics
for ticker, stats_dict in stats_summary.items():
    print(f"Ticker: {ticker}")
    print(f"Mean Price: {stats_dict['Mean Price']:.4f}")
    print(f"Median Price: {stats_dict['Median Price']:.4f}")
    print(f"Mode Price: {stats_dict['Mode Price']:.4f}")
    print(f"Std Dev Returns: {stats_dict['Std Dev Returns']:.4f}")
    print(f"Skewness Returns: {stats_dict['Skewness Returns']:.4f}")
    print(f"Kurtosis Returns: {stats_dict['Kurtosis Returns']:.4f}")
    print("=" * 50)


# In[12]:


import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import seaborn as sns

# Define the tickers and date range
tickers = ["MB.MI", "ISP.MI", "UCG.MI", "SR2000.MI", "XEON.MI", "3ITL.MI", "XFFE.MI", "FTSEMIB.MI", "FBK.MI", "IF.MI", "BGN.MI"]
start_date = "2005-01-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

# Function to download data for a list of tickers
def download_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = yf.download(ticker, start=start_date, end=end_date)
        except Exception as e:
            print(f"Error downloading data for {ticker}: {str(e)}")
    return data

# Function to plot distribution of daily returns for each ticker
def plot_returns_distribution(data):
    for ticker, df in data.items():
        if not df.empty:
            try:
                # Calculate daily returns
                df['Daily Returns'] = df['Adj Close'].pct_change()
                
                # Drop NaN values
                df.dropna(inplace=True)
                
                # Plot histogram of daily returns
                plt.figure(figsize=(10, 6))
                sns.histplot(df['Daily Returns'], bins=50, kde=True, color='blue')
                
                plt.title(f'Distribution of Daily Returns - {ticker}')
                plt.xlabel('Daily Returns')
                plt.ylabel('Frequency')
                plt.grid(True)
                plt.show()
            
            except Exception as e:
                print(f"Error processing data for {ticker}: {str(e)}")
        
        else:
            print(f"No data available for {ticker}")

# Download data
data = download_data(tickers, start_date, end_date)

# Plot distribution of daily returns for each ticker
plot_returns_distribution(data)


# In[15]:


import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Define the tickers and date range
tickers = ["MB.MI", "ISP.MI", "UCG.MI", "SR2000.MI", "XEON.MI", "3ITL.MI", "XFFE.MI", "FTSEMIB.MI", "FBK.MI", "IF.MI", "BGN.MI"]
start_date = "2005-01-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

# Function to download data for a list of tickers
def download_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = yf.download(ticker, start=start_date, end=end_date)
        except Exception as e:
            print(f"Error downloading data for {ticker}: {str(e)}")
    return data

# Function to calculate historical volatility
def calculate_volatility(data, window):
    try:
        # Calculate daily returns
        returns = data['Adj Close'].pct_change()
        
        # Calculate rolling standard deviation
        rolling_sd = returns.rolling(window=window).std()
        
        # Annualize volatility
        annualized_volatility = rolling_sd * (252**0.5)
        
        return annualized_volatility
    
    except Exception as e:
        print(f"Error calculating volatility: {str(e)}")
        return None

# Function to calculate moving averages of volatility
def calculate_moving_averages(volatility, window):
    try:
        # Calculate moving averages
        ma = volatility.rolling(window=window).mean()
        return ma
    
    except Exception as e:
        print(f"Error calculating moving averages: {str(e)}")
        return None

# Download data
data = download_data(tickers, start_date, end_date)

# Calculate and plot historical volatility and moving averages for each ticker
for ticker, df in data.items():
    if not df.empty:
        try:
            # Calculate 5-day and 21-day annualized volatility
            vol_5d = calculate_volatility(df, 5)
            vol_21d = calculate_volatility(df, 21)
            
            # Calculate moving averages
            ma_5d = calculate_moving_averages(vol_5d, 5)
            ma_21d = calculate_moving_averages(vol_21d, 21)
            
            # Plotting
            plt.figure(figsize=(12, 8))
            
            # Plot 5-day and its moving average
            plt.subplot(2, 1, 1)
            plt.plot(vol_5d, label='5-day Volatility', color='blue', alpha=0.8)
            plt.plot(ma_5d, label='5-day MA', linestyle='--', color='orange')
            plt.title(f'{ticker} - 5-day Annualized Historical Volatility and Moving Average')
            plt.xlabel('Date')
            plt.ylabel('Volatility')
            plt.legend()
            plt.grid(True)
            
            # Plot 21-day and its moving average
            plt.subplot(2, 1, 2)
            plt.plot(vol_21d, label='21-day Volatility', color='red', alpha=0.8)
            plt.plot(ma_21d, label='21-day MA', linestyle='--', color='green')
            plt.title(f'{ticker} - 21-day Annualized Historical Volatility and Moving Average')
            plt.xlabel('Date')
            plt.ylabel('Volatility')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            print(f"Error processing data for {ticker}: {str(e)}")
    else:
        print(f"No data available for {ticker}")


# In[ ]:





# In[19]:


import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Define the tickers and date range
tickers = ["MB.MI", "ISP.MI", "UCG.MI", "SR2000.MI", "XEON.MI", "3ITL.MI", "XFFE.MI", "FTSEMIB.MI", "FBK.MI", "IF.MI", "BGN.MI"]
start_date = "2005-01-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

# Function to download data for a list of tickers
def download_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = yf.download(ticker, start=start_date, end=end_date)
        except Exception as e:
            print(f"Error downloading data for {ticker}: {str(e)}")
    return data

# Function to calculate historical volatility
def calculate_volatility(data, window):
    try:
        # Calculate daily returns
        returns = data['Adj Close'].pct_change()
        
        # Calculate rolling standard deviation
        rolling_sd = returns.rolling(window=window).std()
        
        # Annualize volatility
        annualized_volatility = rolling_sd * (252**0.5)
        
        return annualized_volatility
    
    except Exception as e:
        print(f"Error calculating volatility: {str(e)}")
        return None

# Download data
data = download_data(tickers, start_date, end_date)

# Calculate and plot historical volatility for each ticker
for ticker, df in data.items():
    if not df.empty:
        try:
            # Calculate 5-day, 21-day, and 50-day annualized volatility
            vol_5d = calculate_volatility(df, 5)
            vol_21d = calculate_volatility(df, 21)
            vol_50d = calculate_volatility(df, 50)
            
            # Plotting
            plt.figure(figsize=(12, 6))
            
            # Plot 5-day, 21-day, and 50-day volatilities on the same graph
            plt.plot(vol_5d, label='5-day Volatility', color='blue')
            plt.plot(vol_21d, label='21-day Volatility', color='red')
            plt.plot(vol_50d, label='50-day Volatility', color='green')
            
            plt.title(f'{ticker} - Annualized Historical Volatility')
            plt.xlabel('Date')
            plt.ylabel('Volatility')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            print(f"Error processing data for {ticker}: {str(e)}")
    else:
        print(f"No data available for {ticker}")


# In[20]:


import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Define the tickers and date range
tickers = ["MB.MI", "ISP.MI", "UCG.MI", "SR2000.MI", "XEON.MI", "3ITL.MI", "XFFE.MI", "FTSEMIB.MI", "FBK.MI", "IF.MI", "BGN.MI"]
start_date = "2005-01-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

# Function to download data for a list of tickers
def download_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = yf.download(ticker, start=start_date, end=end_date)
        except Exception as e:
            print(f"Error downloading data for {ticker}: {str(e)}")
    return data

# Function to calculate historical volatility
def calculate_volatility(data, window):
    try:
        # Calculate daily returns
        returns = data['Adj Close'].pct_change()
        
        # Calculate rolling standard deviation
        rolling_sd = returns.rolling(window=window).std()
        
        # Annualize volatility
        annualized_volatility = rolling_sd * (252**0.5)
        
        return annualized_volatility
    
    except Exception as e:
        print(f"Error calculating volatility: {str(e)}")
        return None

# Function to calculate moving averages of volatility
def calculate_moving_averages(volatility, windows):
    try:
        # Calculate moving averages
        mas = {window: volatility.rolling(window=window).mean() for window in windows}
        return mas
    
    except Exception as e:
        print(f"Error calculating moving averages: {str(e)}")
        return None

# Download data
data = download_data(tickers, start_date, end_date)

# Calculate and plot historical volatility and moving averages for each ticker
for ticker, df in data.items():
    if not df.empty:
        try:
            # Calculate 5-day, 21-day, and 50-day annualized volatility
            vol_5d = calculate_volatility(df, 5)
            vol_21d = calculate_volatility(df, 21)
            vol_50d = calculate_volatility(df, 50)
            
            # Calculate moving averages for 5-day, 21-day, and 50-day volatilities
            windows = [5, 21, 50]
            ma_data = calculate_moving_averages(pd.concat([vol_5d, vol_21d, vol_50d], axis=1), windows)
            
            # Plotting
            plt.figure(figsize=(12, 6))
            
            # Plot 5-day, 21-day, and 50-day moving averages of volatility on the same graph
            for window in windows:
                plt.plot(ma_data[window], label=f'{window}-day MA', linestyle='-', linewidth=2)
            
            plt.title(f'{ticker} - Moving Averages of Annualized Historical Volatility')
            plt.xlabel('Date')
            plt.ylabel('Volatility')
            plt.legend()
            plt.grid(True)
            
            plt.tight_layout()
            plt.show()
        
        except Exception as e:
            print(f"Error processing data for {ticker}: {str(e)}")
    else:
        print(f"No data available for {ticker}")


# In[21]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Define the tickers and date range
tickers = ["MB.MI", "ISP.MI", "UCG.MI", "SR2000.MI", "XEON.MI"]
start_date = "2005-01-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

# Download data
def download_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
        except Exception as e:
            print(f"Error downloading data for {ticker}: {str(e)}")
    return pd.DataFrame(data)

# Download adjusted close prices
df = download_data(tickers, start_date, end_date)


# In[22]:


# Calculate daily returns
returns = df.pct_change().dropna()

# Calculate covariance matrix
cov_matrix = returns.cov()


# In[23]:


from scipy.optimize import minimize

# Define functions for portfolio returns and risks
def portfolio_return(weights, returns):
    return np.sum(returns.mean() * weights) * 252

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

# Objective function (minimize negative Sharpe ratio)
def negative_sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
    p_return = portfolio_return(weights, returns)
    p_volatility = portfolio_volatility(weights, cov_matrix)
    return -(p_return - risk_free_rate) / p_volatility

# Constraints (weights sum to 1)
constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

# Bounds (0 <= weights <= 1)
bounds = tuple((0, 1) for _ in range(len(tickers)))

# Initial guess (equal weights)
init_guess = [1 / len(tickers)] * len(tickers)

# Risk-free rate (example: 1%)
risk_free_rate = 0.01

# Optimize for the portfolio with maximum Sharpe ratio (tangency portfolio)
result = minimize(negative_sharpe_ratio, init_guess,
                  args=(returns, cov_matrix, risk_free_rate),
                  method='SLSQP', bounds=bounds, constraints=constraints)

# Optimal weights and portfolio statistics
optimal_weights = result.x
optimal_return = portfolio_return(optimal_weights, returns)
optimal_volatility = portfolio_volatility(optimal_weights, cov_matrix)

# Efficient frontier portfolio returns and volatilities
ef_returns = np.linspace(returns.min().min(), returns.max().max(), num=100)
ef_volatilities = []

for ef_return in ef_returns:
    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x, returns) - ef_return},
                   {'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    
    ef_result = minimize(portfolio_volatility, init_guess,
                         args=(cov_matrix,), method='SLSQP',
                         bounds=bounds, constraints=constraints)
    
    ef_volatilities.append(ef_result.fun)

ef_volatilities = np.array(ef_volatilities)

# Calculate CAPM portfolio (market portfolio)
market_weights = np.ones(len(tickers)) / len(tickers)
market_return = portfolio_return(market_weights, returns)
market_volatility = portfolio_volatility(market_weights, cov_matrix)

# Calculate CAPM tangent line
capm_slope = (optimal_return - risk_free_rate) / optimal_volatility
capm_line = risk_free_rate + capm_slope * ef_volatilities

# Print portfolio statistics
print("Optimal Weights:", optimal_weights)
print("Optimal Portfolio Return:", optimal_return)
print("Optimal Portfolio Volatility:", optimal_volatility)


# In[24]:


# Plotting
plt.figure(figsize=(10, 6))

# Plot efficient frontier
plt.plot(ef_volatilities, ef_returns, linestyle='-', color='b', label='Efficient Frontier')

# Plot tangency portfolio
plt.scatter(optimal_volatility, optimal_return, marker='o', color='r', label='Tangency Portfolio')

# Plot CAPM portfolio (market portfolio)
plt.scatter(market_volatility, market_return, marker='s', color='g', label='CAPM Portfolio')

# Plot CAPM tangent line
plt.plot(ef_volatilities, capm_line, linestyle='--', color='g', label='CAPM Tangent Line')

plt.title('Mean-Variance Efficient Frontier and CAPM Portfolio')
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Return')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# In[25]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Define the tickers and date range
tickers = ["MB.MI", "ISP.MI", "UCG.MI", "SR2000.MI", "XEON.MI", "3ITL.MI", 
           "XFFE.MI", "FTSEMIB.MI", "FBK.MI", "IF.MI", "BGN.MI"]
start_date = "2005-01-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

# Download data
def download_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
        except Exception as e:
            print(f"Error downloading data for {ticker}: {str(e)}")
    return pd.DataFrame(data)

# Download adjusted close prices
df = download_data(tickers, start_date, end_date)
# Calculate daily returns
returns = df.pct_change().dropna()
# Calculate correlation matrix
correlation_matrix = returns.corr()
def portfolio_return(weights, returns):
    return np.sum(returns.mean() * weights) * 252

def portfolio_volatility(weights, correlation_matrix):
    std_dev = np.sqrt(np.diag(correlation_matrix))
    return np.sqrt(np.dot(weights.T, np.dot(correlation_matrix, weights))) * np.sqrt(252)

def objective_function(weights, returns, correlation_matrix):
    return portfolio_volatility(weights, correlation_matrix)

def constraint(weights):
    return np.sum(weights) - 1
from scipy.optimize import minimize

# Number of assets
n_assets = len(tickers)

# Initial weights (equal weights)
init_weights = np.ones(n_assets) / n_assets

# Optimization constraints
constraints = ({'type': 'eq', 'fun': constraint})

# Bounds for weights (0 <= weights <= 1)
bounds = tuple((0, 1) for _ in range(n_assets))

# Optimization function
result = minimize(objective_function, init_weights,
                  args=(returns, correlation_matrix), method='SLSQP',
                  bounds=bounds, constraints=constraints)

# Optimal weights and portfolio statistics
optimal_weights = result.x
optimal_return = portfolio_return(optimal_weights, returns)
optimal_volatility = portfolio_volatility(optimal_weights, correlation_matrix)

print("Optimal Weights:", optimal_weights)
print("Optimal Portfolio Return:", optimal_return)
print("Optimal Portfolio Volatility:", optimal_volatility)
# Range of returns for efficient frontier
target_returns = np.linspace(returns.min().min(), returns.max().max(), num=50)

# Calculate efficient frontier
efficient_frontier = []
for target_return in target_returns:
    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x, returns) - target_return},
                   {'type': 'eq', 'fun': constraint})
    
    result = minimize(portfolio_volatility, init_weights,
                      args=(correlation_matrix,), method='SLSQP',
                      bounds=bounds, constraints=constraints)
    
    efficient_frontier.append((target_return, result.fun))

efficient_frontier = np.array(efficient_frontier)

# Plotting the efficient frontier
plt.figure(figsize=(10, 6))
plt.scatter(efficient_frontier[:, 1], efficient_frontier[:, 0], c=efficient_frontier[:, 0] / efficient_frontier[:, 1],
            marker='o', cmap='viridis')
plt.title('Mean-Variance Efficient Frontier (Using Correlation Matrix)')
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Return')
plt.colorbar(label='Sharpe Ratio')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[26]:


import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

# Define the tickers and date range
tickers = ["MB.MI", "ISP.MI", "UCG.MI", "SR2000.MI", "XEON.MI", "3ITL.MI", 
           "XFFE.MI", "FTSEMIB.MI", "FBK.MI", "IF.MI", "BGN.MI"]
start_date = "2005-01-01"
end_date = pd.Timestamp.today().strftime("%Y-%m-%d")

# Download data
def download_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
        except Exception as e:
            print(f"Error downloading data for {ticker}: {str(e)}")
    return pd.DataFrame(data)

# Download adjusted close prices
df = download_data(tickers, start_date, end_date)
# Calculate daily returns
returns = df.pct_change().dropna()
# Calculate covariance matrix
cov_matrix = returns.cov()
def portfolio_return(weights, returns):
    return np.sum(returns.mean() * weights) * 252

def portfolio_volatility(weights, cov_matrix):
    return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)

def objective_function(weights, returns, cov_matrix):
    return portfolio_volatility(weights, cov_matrix)

def constraint(weights):
    return np.sum(weights) - 1
from scipy.optimize import minimize

# Number of assets
n_assets = len(tickers)

# Initial weights (equal weights)
init_weights = np.ones(n_assets) / n_assets

# Optimization constraints
constraints = ({'type': 'eq', 'fun': constraint})

# Bounds for weights (0 <= weights <= 1)
bounds = tuple((0, 1) for _ in range(n_assets))

# Optimization function
result = minimize(objective_function, init_weights,
                  args=(returns, cov_matrix), method='SLSQP',
                  bounds=bounds, constraints=constraints)

# Optimal weights and portfolio statistics
optimal_weights = result.x
optimal_return = portfolio_return(optimal_weights, returns)
optimal_volatility = portfolio_volatility(optimal_weights, cov_matrix)

print("Optimal Weights:", optimal_weights)
print("Optimal Portfolio Return:", optimal_return)
print("Optimal Portfolio Volatility:", optimal_volatility)
# Range of returns for efficient frontier
target_returns = np.linspace(returns.min().min(), returns.max().max(), num=50)

# Calculate efficient frontier
efficient_frontier = []
for target_return in target_returns:
    constraints = ({'type': 'eq', 'fun': lambda x: portfolio_return(x, returns) - target_return},
                   {'type': 'eq', 'fun': constraint})
    
    result = minimize(portfolio_volatility, init_weights,
                      args=(cov_matrix,), method='SLSQP',
                      bounds=bounds, constraints=constraints)
    
    efficient_frontier.append((target_return, result.fun))

efficient_frontier = np.array(efficient_frontier)

# Plotting the efficient frontier
plt.figure(figsize=(10, 6))
plt.scatter(efficient_frontier[:, 1], efficient_frontier[:, 0], c=efficient_frontier[:, 0] / efficient_frontier[:, 1],
            marker='o', cmap='viridis')
plt.title('Mean-Variance Efficient Frontier')
plt.xlabel('Annualized Volatility')
plt.ylabel('Annualized Return')
plt.colorbar(label='Sharpe Ratio')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[28]:


import numpy as np
import matplotlib.pyplot as plt

# Optimal weights from the portfolio optimization
optimal_weights = np.array([3.18008333e-02, 4.46687394e-16, 4.71767188e-02, 
                            1.47126649e-01, 2.52458413e-01, 0.00000000e+00, 
                            2.53778091e-01, 0.00000000e+00, 1.03811165e-01, 
                            1.10907611e-01, 5.29405181e-02])

# Total investment amount in €100 million
total_investment = 100_000_000  # €100 million

# Calculate allocations
allocations = total_investment * optimal_weights

# Tickers for plotting
tickers = ["MB.MI", "ISP.MI", "UCG.MI", "SR2000.MI", "XEON.MI", 
           "3ITL.MI", "XFFE.MI", "FTSEMIB.MI", "FBK.MI", "IF.MI", "BGN.MI"]

# Plotting Mean-Variance Efficient Portfolio Allocation
plt.figure(figsize=(10, 6))
plt.bar(tickers, allocations, color='blue', alpha=0.7)
plt.title('Mean-Variance Efficient Portfolio Allocation (€100 million)')
plt.xlabel('Ticker')
plt.ylabel('Allocation (€)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Hypothetical CAPM weights (example)
capm_weights = np.array([0.1, 0.2, 0.1, 0.05, 0.15, 0.05, 0.1, 0.05, 0.05, 0.05, 0.1])

# Calculate CAPM allocations
capm_allocations = total_investment * capm_weights

# Plotting CAPM Allocation
plt.figure(figsize=(10, 6))
plt.bar(tickers, capm_allocations, color='green', alpha=0.7)
plt.title('CAPM Allocation (€100 million)')
plt.xlabel('Ticker')
plt.ylabel('Allocation (€)')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[29]:


import numpy as np
import matplotlib.pyplot as plt

# Example data (optimal weights and expected returns)
optimal_weights = np.array([3.18008333e-02, 4.46687394e-16, 4.71767188e-02, 
                            1.47126649e-01, 2.52458413e-01, 0.00000000e+00, 
                            2.53778091e-01, 0.00000000e+00, 1.03811165e-01, 
                            1.10907611e-01, 5.29405181e-02])
expected_returns = np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15])
volatilities = np.array([5.0, 4.5, 4.0, 3.5, 3.0, 2.5, 2.0, 1.5, 1.0, 0.5, 0.2])

# Calculate Sharpe ratios
risk_free_rate = 0.02  # Assumed risk-free rate
sharpe_ratios = (expected_returns - risk_free_rate) / volatilities

# Find optimal portfolio (maximum Sharpe ratio)
optimal_index = np.argmax(sharpe_ratios)
optimal_return = expected_returns[optimal_index]
optimal_volatility = volatilities[optimal_index]

# Plot Efficient Portfolio Frontier
plt.figure(figsize=(10, 6))
plt.scatter(volatilities, expected_returns, c=sharpe_ratios, cmap='viridis', label='Portfolios')
plt.scatter(optimal_volatility, optimal_return, marker='*', color='red', s=100, label='Optimal Portfolio')

# Plot Tangent Line (Capital Market Line)
plt.plot([0, optimal_volatility], [risk_free_rate, optimal_return], linestyle='-', color='green', label='CML')

# Plot CAPM line (linear regression line)
beta_values = np.linspace(0, 1.5, 100)
capm_expected_returns = risk_free_rate + beta_values * (optimal_return - risk_free_rate)
plt.plot(beta_values, capm_expected_returns, linestyle='--', color='blue', label='CAPM')

plt.title('Efficient Portfolio Frontier with Tangent Curve to Market Line (CML) and CAPM Line')
plt.xlabel('Portfolio Volatility (Standard Deviation)')
plt.ylabel('Expected Return')
plt.colorbar(label='Sharpe Ratio')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[30]:


import numpy as np
import matplotlib.pyplot as plt

# Example data (optimal weights and expected returns)
optimal_weights = np.array([6.13566449e-05, 6.58962171e-18, 1.52688990e-18, 5.54334280e-04,
                            9.91465491e-01, 2.22687092e-18, 5.84648731e-03, 1.67613962e-03,
                            3.96191157e-04, 2.22683223e-18, 4.71219567e-18])
expected_returns = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11])
volatilities = np.array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11])

# Calculate Sharpe ratios
risk_free_rate = 0.002  # Assumed risk-free rate
sharpe_ratios = (expected_returns - risk_free_rate) / volatilities

# Find optimal portfolio (maximum Sharpe ratio)
optimal_index = np.argmax(sharpe_ratios)
optimal_return = expected_returns[optimal_index]
optimal_volatility = volatilities[optimal_index]

# Plot Efficient Portfolio Frontier
plt.figure(figsize=(10, 6))
plt.scatter(volatilities, expected_returns, c=sharpe_ratios, cmap='viridis', label='Portfolios')
plt.scatter(optimal_volatility, optimal_return, marker='*', color='red', s=100, label='Optimal Portfolio')

# Plot Tangent Line (Capital Market Line)
plt.plot([0, optimal_volatility], [risk_free_rate, optimal_return], linestyle='-', color='green', label='CML')

# Plot CAPM line (linear regression line)
beta_values = np.linspace(0, 1.5, 100)
capm_expected_returns = risk_free_rate + beta_values * (optimal_return - risk_free_rate)
plt.plot(beta_values, capm_expected_returns, linestyle='--', color='blue', label='CAPM')

plt.title('Efficient Portfolio Frontier with Tangent Curve to Market Line (CML) and CAPM Line')
plt.xlabel('Portfolio Volatility (Standard Deviation)')
plt.ylabel('Expected Return')
plt.colorbar(label='Sharpe Ratio')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


# In[32]:


import pandas as pd
import yfinance as yf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Define the tickers and date range
tickers = ["MB.MI", "ISP.MI", "UCG.MI", "SR2000.MI", "XEON.MI", "3ITL.MI", "XFFE.MI", "FTSEMIB.MI", "FBK.MI", "IF.MI", "BGN.MI"]
start_date = "2005-01-01"
end_date = "2024-07-06"

# Function to download data for a list of tickers
def download_data(tickers, start_date, end_date):
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
        except Exception as e:
            print(f"Error downloading data for {ticker}: {str(e)}")
    return data

# Download data
data = download_data(tickers, start_date, end_date)

# Calculate daily returns
returns = pd.DataFrame({ticker: df.pct_change().dropna() for ticker, df in data.items()})

# Calculate correlation matrix
correlation_matrix = returns.corr()

# Plotting correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title('Correlation Matrix of Adjusted Close Prices')
plt.show()


# In[ ]:




