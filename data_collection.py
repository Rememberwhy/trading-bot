# data_collection.py
import ccxt
import pandas as pd
import time
import requests
from textblob import TextBlob
import os
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Load API credentials from config.json
with open('config.json') as config_file:
    config = json.load(config_file)

# Initialize the Binance exchange
binance = ccxt.binance({
    'apiKey': config['apiKey'],
    'secret': config['secret'],
})

def fetch_historical_data(symbol, timeframe='1m', limit=1000):
    logging.info(f"Fetching historical data for {symbol}")
    try:
        ohlcv = binance.fetch_ohlcv(symbol, timeframe=timeframe, limit=limit)
        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        return df
    except Exception as e:
        logging.error(f"Failed to fetch historical data: {str(e)}")
        return pd.DataFrame()

def calculate_technical_indicators(df):
    logging.info("Calculating technical indicators")
    df['SMA50'] = df['close'].rolling(window=50).mean()
    df['SMA200'] = df['close'].rolling(window=200).mean()
    df['RSI'] = (100 - (100 / (1 + df['close'].pct_change().add(1).cumprod())))
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    df['Upper_BB'] = df['close'].rolling(window=20).mean() + (df['close'].rolling(window=20).std() * 2)
    df['Lower_BB'] = df['close'].rolling(window=20).mean() - (df['close'].rolling(window=20).std() * 2)
    df['ATR'] = df['high'] - df['low']
    df['Stochastic_Oscillator'] = (df['close'] - df['low'].rolling(window=14).min()) / (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min())
    return df

# Function to perform sentiment analysis
def fetch_sentiment(symbol):
    # Example: Fetch sentiment from Twitter (use appropriate API for your needs)
    url = f"https://api.twitter.com/2/tweets/search/recent?query={symbol}&tweet.fields=created_at&max_results=100"
    headers = {'Authorization': 'AAAAAAAAAAAAAAAAAAAAAPSwugEAAAAAjYn8Rapiiar5LbMv5f2WkhkFYf0%3D7yBRxohTnMIOPYTctvdO5M1NnyMO7Ul9Bibph2pVcbh47wMFFH'}
    response = requests.get(url, headers=headers)
    tweets = response.json().get('data', [])
    
    sentiments = []
    for tweet in tweets:
        analysis = TextBlob(tweet['text'])
        sentiments.append(analysis.sentiment.polarity)
    
    if sentiments:
        average_sentiment = sum(sentiments) / len(sentiments)
    else:
        average_sentiment = 0
    return average_sentiment

def collect_data(symbol):
    df = fetch_historical_data(symbol)
    if not df.empty:
        df = calculate_technical_indicators(df)
        df['sentiment'] = fetch_sentiment(symbol)
    return df

def collect_data_periodically(symbol, interval=60):
    directory = symbol.split('/')[0]  # Extract 'ADA' from 'ADA/USDT'
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f'{symbol.replace("/", "_")}_data.csv')

    while True:
        try:
            df = collect_data(symbol)
            if not df.empty:
                df.to_csv(file_path, mode='a', header=not os.path.exists(file_path))
                logging.info(f"Data saved for {symbol} at {pd.Timestamp.now()}")
            else:
                logging.info("No data fetched, skipping this cycle.")
        except Exception as e:
            logging.error(f"Error occurred: {str(e)}")
        time.sleep(interval)

if __name__ == "__main__":
    symbol = 'ADA/USDT'  # Replace with the trading pair you want to monitor
    collect_data_periodically(symbol, interval=60)