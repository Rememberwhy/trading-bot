import ccxt
import time
import pandas as pd
import joblib
import numpy as np
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

# Configure logging
logging.basicConfig(filename='trading_bot.log', level=logging.INFO,
                    format='%(asctime)s %(levelname)s: %(message)s')

# Load the model and scaler
model = joblib.load('trading_model.pkl')
scaler = joblib.load('scaler.pkl')

# Load API credentials from config.json
with open('config.json') as config_file:
    config = json.load(config_file)

# Initialize the Binance exchange
binance = ccxt.binance({
    'apiKey': config['apiKey'],
    'secret': config['secret'],
})


# Candlestick pattern detection functions
def is_bullish_engulfing(df):
    pattern = []
    for i in range(1, len(df)):
        if df['close'][i] > df['open'][i] and \
           df['close'][i-1] < df['open'][i-1] and \
           df['close'][i] > df['open'][i-1] and \
           df['open'][i] < df['close'][i-1]:
            pattern.append(True)
        else:
            pattern.append(False)
    return [False] + pattern

def is_bearish_engulfing(df):
    pattern = []
    for i in range(1, len(df)):
        if df['close'][i] < df['open'][i] and \
           df['close'][i-1] > df['open'][i-1] and \
           df['close'][i] < df['open'][i-1] and \
           df['open'][i] > df['close'][i-1]:
            pattern.append(True)
        else:
            pattern.append(False)
    return [False] + pattern

def is_morning_star(df):
    pattern = []
    for i in range(2, len(df)):
        if df['close'][i-2] < df['open'][i-2] and \
           df['close'][i-1] < df['open'][i-1] and \
           df['close'][i] > df['open'][i] and \
           df['close'][i] > df['open'][i-2] and \
           df['open'][i-1] < df['close'][i-2] and \
           df['open'][i] > df['close'][i-1]:
            pattern.append(True)
        else:
            pattern.append(False)
    return [False, False] + pattern

def is_evening_star(df):
    pattern = []
    for i in range(2, len(df)):
        if df['close'][i-2] > df['open'][i-2] and \
           df['close'][i-1] > df['open'][i-1] and \
           df['close'][i] < df['open'][i] and \
           df['close'][i] < df['open'][i-2] and \
           df['open'][i-1] > df['close'][i-2] and \
           df['open'][i] < df['close'][i-1]:
            pattern.append(True)
        else:
            pattern.append(False)
    return [False, False] + pattern

def is_doji(df):
    pattern = []
    for i in range(len(df)):
        body = abs(df['close'][i] - df['open'][i])
        upper_shadow = df['high'][i] - max(df['close'][i], df['open'][i])
        lower_shadow = min(df['close'][i], df['open'][i]) - df['low'][i]
        if body <= (0.1 * (df['high'][i] - df['low'][i])) and \
           upper_shadow > body and \
           lower_shadow > body:
            pattern.append(True)
        else:
            pattern.append(False)
    return pattern

def is_falling_3_method(df):
    pattern = []
    for i in range(2, len(df)-2):
        if df['close'][i-2] > df['open'][i-2] and \
           df['close'][i+2] < df['open'][i+2] and \
           df['close'][i-2] < df['open'][i+2] and \
           df['close'][i+2] < df['open'][i-2] and \
           df['open'][i] < df['close'][i-2] and df['close'][i] > df['open'][i+2]:
            pattern.append(True)
        else:
            pattern.append(False)
    return pattern + [False, False, False, False]

def is_hammer(df):
    pattern = []
    for i in range(len(df)):
        body = abs(df['close'][i] - df['open'][i])
        lower_shadow = df['open'][i] - df['low'][i] if df['close'][i] > df['open'][i] else df['close'][i] - df['low'][i]
        upper_shadow = df['high'][i] - df['close'][i] if df['close'][i] > df['open'][i] else df['high'][i] - df['open'][i]
        if lower_shadow > 2 * body and upper_shadow < body:
            pattern.append(True)
        else:
            pattern.append(False)
    return pattern

def is_inverted_hammer(df):
    pattern = []
    for i in range(len(df)):
        body = abs(df['close'][i] - df['open'][i])
        upper_shadow = df['high'][i] - df['close'][i] if df['close'][i] > df['open'][i] else df['high'][i] - df['open'][i]
        lower_shadow = df['open'][i] - df['low'][i] if df['close'][i] > df['open'][i] else df['close'][i] - df['low'][i]
        if upper_shadow > 2 * body and lower_shadow < body:
            pattern.append(True)
        else:
            pattern.append(False)
    return pattern

def is_hanging_man(df):
    pattern = []
    for i in range(len(df)):
        body = abs(df['close'][i] - df['open'][i])
        lower_shadow = df['open'][i] - df['low'][i] if df['close'][i] > df['open'][i] else df['close'][i] - df['low'][i]
        upper_shadow = df['high'][i] - df['close'][i] if df['close'][i] > df['open'][i] else df['high'][i] - df['open'][i]
        if lower_shadow > 2 * body and upper_shadow < body and df['close'][i] < df['open'][i]:
            pattern.append(True)
        else:
            pattern.append(False)
    return pattern

def is_high_wave(df):
    pattern = []
    for i in range(len(df)):
        body = abs(df['close'][i] - df['open'][i])
        upper_shadow = df['high'][i] - max(df['close'][i], df['open'][i])
        lower_shadow = min(df['close'][i], df['open'][i]) - df['low'][i]
        if upper_shadow > body and lower_shadow > body:
            pattern.append(True)
        else:
            pattern.append(False)
    return pattern

def is_hammer_cross_bearish(df):
    pattern = []
    for i in range(len(df)):
        body = abs(df['close'][i] - df['open'][i])
        lower_shadow = df['open'][i] - df['low'][i] if df['close'][i] > df['open'][i] else df['close'][i] - df['low'][i]
        upper_shadow = df['high'][i] - df['close'][i] if df['close'][i] > df['open'][i] else df['high'][i] - df['open'][i]
        if body < lower_shadow and lower_shadow > 2 * body and df['close'][i] < df['open'][i]:
            pattern.append(True)
        else:
            pattern.append(False)
    return pattern

def is_kicking_bearish(df):
    pattern = []
    for i in range(1, len(df)):
        if df['open'][i-1] > df['close'][i-1] and \
           df['open'][i] < df['close'][i] and \
           df['open'][i] > df['close'][i-1] and \
           df['close'][i] > df['open'][i-1]:
            pattern.append(True)
        else:
            pattern.append(False)
    return [False] + pattern

def is_three_outside_up(df):
    pattern = []
    for i in range(2, len(df)):
        if df['close'][i-2] < df['open'][i-2] and \
           df['close'][i-1] > df['open'][i-1] and \
           df['close'][i-1] > df['open'][i-2] and \
           df['open'][i-1] < df['close'][i-2] and \
           df['close'][i] > df['close'][i-1]:
            pattern.append(True)
        else:
            pattern.append(False)
    return [False, False] + pattern

def is_abandoned_baby(df):
    pattern = []
    for i in range(2, len(df)):
        if df['close'][i-2] > df['open'][i-2] and \
           df['close'][i-1] == df['open'][i-1] and \
           df['close'][i] < df['open'][i] and \
           df['close'][i-2] > df['close'][i-1] and \
           df['close'][i-1] < df['open'][i] and \
           df['close'][i] < df['open'][i-2]:
            pattern.append(True)
        else:
            pattern.append(False)
    return [False, False] + pattern

def is_baby_swallowing(df):
    pattern = []
    for i in range(1, len(df)):
        if df['close'][i-1] < df['open'][i-1] and \
           df['close'][i] > df['open'][i] and \
           df['open'][i] < df['close'][i-1] and \
           df['close'][i] > df['open'][i-1]:
            pattern.append(True)
        else:
            pattern.append(False)
    return [False] + pattern

# Function to fetch the latest OHLCV data from Binance
def fetch_ohlcv(symbol, timeframe, limit=100):
    logging.info(f'Fetching OHLCV data for {symbol}')
    return binance.fetch_ohlcv(symbol, timeframe, limit=limit)

# Function to create features from OHLCV data
def create_features(df):
    logging.info('Creating features from OHLCV data')
    df['returns'] = df['close'].pct_change()
    df['volatility'] = df['returns'].rolling(window=10).std()
    df['momentum'] = df['returns'].rolling(window=10).mean()
    df['is_bullish_engulfing'] = is_bullish_engulfing(df)
    df['is_bearish_engulfing'] = is_bearish_engulfing(df)
    df['is_morning_star'] = is_morning_star(df)
    df['is_evening_star'] = is_evening_star(df)
    df['is_doji'] = is_doji(df)
    df['is_falling_3_method'] = is_falling_3_method(df)
    df['is_hammer'] = is_hammer(df)
    df['is_inverted_hammer'] = is_inverted_hammer(df)
    df['is_hanging_man'] = is_hanging_man(df)
    df['is_high_wave'] = is_high_wave(df)
    df['is_hammer_cross_bearish'] = is_hammer_cross_bearish(df)
    df['is_kicking_bearish'] = is_kicking_bearish(df)
    df['is_three_outside_up'] = is_three_outside_up(df)
    df['is_abandoned_baby'] = is_abandoned_baby(df)
    df['is_baby_swallowing'] = is_baby_swallowing(df)
    return df.dropna()

# Function to preprocess features
def preprocess_features(df):
    logging.info('Preprocessing features')
    features = df[['volatility', 'momentum', 'is_bullish_engulfing', 'is_bearish_engulfing',
                   'is_morning_star', 'is_evening_star', 'is_doji', 'is_falling_3_method',
                   'is_hammer', 'is_inverted_hammer', 'is_hanging_man', 'is_high_wave',
                   'is_hammer_cross_bearish', 'is_kicking_bearish', 'is_three_outside_up',
                   'is_abandoned_baby', 'is_baby_swallowing']]
    scaled_features = scaler.transform(features)
    return scaled_features

# Function to make a prediction
def make_prediction(features):
    logging.info('Making prediction')
    return model.predict(features)

# Function to fetch the current price of a symbol
def fetch_current_price(symbol):
    ticker = binance.fetch_ticker(symbol)
    return ticker['last']

# Function to execute trade
def execute_trade(signal, symbol, usdt_amount):
    price = fetch_current_price(symbol)
    amount = usdt_amount / price
    if signal == 1:
        logging.info(f'Placing BUY order for {amount} of {symbol}')
        binance.create_market_buy_order(symbol, amount)
    elif signal == -1:
        logging.info(f'Placing SELL order for {amount} of {symbol}')
        binance.create_market_sell_order(symbol, amount)

# Main function
def main():
    symbol = 'ADA/USDT'
    timeframe = '1h'
    usdt_amount = 20  # Amount in USDT to be used per transaction


    while True:
        try:
            # Fetch latest OHLCV data
            ohlcv = fetch_ohlcv(symbol, timeframe)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            
            # Create features
            features_df = create_features(df)
            
            # Preprocess features
            features = preprocess_features(features_df)
            
            # Make prediction
            prediction = make_prediction(features)
            signal = prediction[-1]  # Get the latest prediction
            
            # Execute trade based on prediction
            execute_trade(signal, symbol, usdt_amount)
            
            # Sleep before next iteration
            time.sleep(3600)  # Sleep for 1 hour
        except Exception as e:
            logging.error(f'Error: {e}')
            time.sleep(60)  # Sleep for 1 minute before retrying

if __name__ == "__main__":
    main()