# backend/train_and_save.py
import yfinance as yf
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import joblib
import time
import os
# Fetch latest NIFTY50 data
# Download NIFTY50 data (last 5 years)
import requests

def fetch_alpha_vantage(symbol, apikey):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&outputsize=full&apikey={apikey}'
    r = requests.get(url)
    data = r.json()
    df = pd.DataFrame.from_dict(data['Time Series (Daily)'], orient='index')
    df = df.astype(float)
    df = df.sort_index()
    df.rename(columns={
    '1. open': 'Open',
    '2. high': 'High',
    '3. low': 'Low',
    '4. close': 'Close',
    '5. volume': 'Volume'
}, inplace=True)

    return df

nifty = fetch_alpha_vantage('NIFTYBEES.BSE', "5TWY0FWPISH2SOWL")
# getting the features that are there already and 
# Base Features
nifty['Prev_Close'] = nifty['Close'].shift(1)
nifty['5MA'] = nifty['Close'].rolling(window=5).mean().shift(1)
nifty['10MA'] = nifty['Close'].rolling(window=10).mean().shift(1)
nifty['Return'] = nifty['Close'].pct_change().shift(1)

# ➕ Derived Features
nifty['Volatility_5D'] = nifty['Close'].rolling(window=5).std().shift(1)
nifty['Momentum_5D'] = nifty['Close'] - nifty['Close'].shift(5)
nifty['MA5_to_MA10'] = nifty['5MA'] / nifty['10MA']
nifty['Price_Range'] = nifty['High'] - nifty['Low']
nifty['Open_Close_Change'] = nifty['Close'] - nifty['Open']
nifty['Rolling_Max_10'] = nifty['Close'].rolling(window=10).max().shift(1)
nifty['Rolling_Min_10'] = nifty['Close'].rolling(window=10).min().shift(1)

# 🎯 Target Variable
nifty['Target'] = nifty['Close'].shift(-1)

# 🧹 Clean up NA values
nifty.dropna(inplace=True)

# 📊 Final Feature Set
features = [
    'Prev_Close', '5MA', '10MA', 'Return',
    'Volatility_5D', 'Momentum_5D', 'MA5_to_MA10',
    'Price_Range', 'Open_Close_Change',
    'Rolling_Max_10', 'Rolling_Min_10'
]

X = nifty[features]
y = nifty['Target']
# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Save model and features
joblib.dump(model, 'nifty_lr_model.pkl')
joblib.dump(X.columns.tolist(), 'features.pkl')