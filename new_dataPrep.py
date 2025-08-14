import yfinance as yf
import numpy as np 
import pandas as pd
import time
import os
import ta

def fetch_data(tickers, start_date, end_date, delay=5):
    """Fetch stock data from Yahoo Finance"""
    ticker_data = {}
    
    for t in tickers:
        print(f"Fetching {t}...")
        ticker_data[t] = yf.Ticker(t).history(start=start_date, end=end_date)
        time.sleep(delay)  # Avoid API rate limits
    
    return ticker_data

def add_technical_indicators(ticker_data):
    """Add technical indicators to stock data"""
    for t, df in ticker_data.items():
        df['EMA_50'] = ta.trend.EMAIndicator(df['Close'], 50).ema_indicator()
        df['EMA_200'] = ta.trend.EMAIndicator(df['Close'], 200).ema_indicator()
        df['RSI'] = ta.momentum.RSIIndicator(df['Close'], 14).rsi()
        df['MACD'] = ta.trend.MACD(df['Close'], window_fast=12, window_sign=9, window_slow=26).macd()
        df['BB_High'] = ta.volatility.BollingerBands(df['Close'], window=15, window_dev=2).bollinger_hband()
        df['BB_Low'] = ta.volatility.BollingerBands(df['Close'], window=15, window_dev=2).bollinger_lband()
        df['ATR'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
        df['OBV'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()
        df['MFI'] = ta.volume.MFIIndicator(df['High'], df['Low'], df['Close'], df['Volume'], window=14).money_flow_index()
        df['ADX'] = ta.trend.ADXIndicator(df['High'], df['Low'], df['Close'], window=14).adx()
        ticker_data[t] = df
    
    return ticker_data

def clean_and_save_data(ticker_data, data_dir="data"):
    """Clean data and save to CSV"""
    os.makedirs(data_dir, exist_ok=True)
    
    for ticker in ticker_data:
        print(f"{ticker}: {ticker_data[ticker].shape}")
        # Drop unwanted columns
        ticker_data[ticker] = ticker_data[ticker].drop(['Dividends', 'Stock Splits'], axis=1)
        # Save to CSV
        ticker_data[ticker].to_csv(f"{data_dir}/{ticker}.csv")
    
    return ticker_data

def align_and_combine_data(ticker_data, save_path=None):
    """Align data across all tickers and combine into a single array"""
    # Find common dates across all tickers
    all_indices = set().union(*[ticker_data[d].index for d in ticker_data])
    
    # Align all data
    aligned_data = []
    for ticker in ticker_data:
        aligned_data.append(ticker_data[ticker].reindex(index=all_indices))
    
    # Stack data into 3D array (timesteps, stocks, features)
    combined_data = np.stack(aligned_data, axis=1)
    
    # Replace NaN values with 0
    filled_data = np.nan_to_num(combined_data, nan=0)
    
    # Save data if path is provided
    if save_path:
        np.save(save_path, filled_data)
    
    return filled_data

def main():
    # Define tickers
    tickers = ['AAPL', 'MA', 'CSCO', 'MSFT', 'AMZN', 'GOOG', 'IBM']
    
    # Create data directory
    data_dir = "data"
    os.makedirs(data_dir, exist_ok=True)
    
    # Process training data
    print("Processing training data...")
    ticker_data = fetch_data(tickers, "2018-01-01", "2024-01-01")
    ticker_data = add_technical_indicators(ticker_data)
    ticker_data = clean_and_save_data(ticker_data, data_dir)
    train_data = align_and_combine_data(ticker_data, f"{data_dir}/train_processed_data.npy")
    print(f"Training data shape: {train_data.shape}")
    
    # Process test data
    print("Processing test data...")
    ticker_data_test = fetch_data(tickers, "2024-01-01", "2025-01-01")
    ticker_data_test = add_technical_indicators(ticker_data_test)
    ticker_data_test = clean_and_save_data(ticker_data_test, data_dir)
    test_data = align_and_combine_data(ticker_data_test, f"{data_dir}/test_processed_data.npy")
    print(f"Test data shape: {test_data.shape}")

if __name__ == "__main__":
    main()