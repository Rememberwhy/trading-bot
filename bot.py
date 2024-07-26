import ccxt
import time
import pandas as pd
import joblib
import numpy as np
import logging
import os
import json
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

#This following functions below will look and find pre-defined candlestick strategy

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

def is_three_outside_down(df):
    pattern = []
    for i in range(2, len(df)):
        if df['close'][i-2] > df['open'][i-2] and \
           df['close'][i-1] < df['open'][i-1] and \
           df['close'][i-1] < df['open'][i-2] and \
           df['open'][i-1] > df['close'][i-2] and \
           df['close'][i] < df['close'][i-1]:
            pattern.append(True)
        else:
            pattern.append(False)
    return [False, False] + pattern

def is_unique_three_river_bottom(df):
    pattern = []
    for i in range(2, len(df)):
        if df['close'][i-2] > df['open'][i-2] and \
           df['close'][i-1] < df['open'][i-1] and \
           df['close'][i] > df['open'][i-1] and \
           df['open'][i] < df['close'][i-1] and \
           df['close'][i] > df['open'][i-2]:
            pattern.append(True)
        else:
            pattern.append(False)
    return [False, False] + pattern

def is_three_stars_in_south(df):
    pattern = []
    for i in range(2, len(df)):
        if df['close'][i-2] < df['open'][i-2] and \
           df['close'][i-1] < df['open'][i-1] and \
           df['close'][i] < df['open'][i] and \
           df['low'][i-1] > df['low'][i-2] and \
           df['low'][i] > df['low'][i-1]:
            pattern.append(True)
        else:
            pattern.append(False)
    return [False, False] + pattern

def is_three_inside_down(df):
    pattern = []
    for i in range(2, len(df)):
        if df['close'][i-2] > df['open'][i-2] and \
           df['close'][i-1] < df['open'][i-1] and \
           df['close'][i-1] > df['open'][i-2] and \
           df['open'][i-1] < df['close'][i-2] and \
           df['close'][i] < df['close'][i-1]:
            pattern.append(True)
        else:
            pattern.append(False)
    return [False, False] + pattern


# Function to place a market buy order
def place_buy_order(symbol, amount):
    try:
        order = binance.create_market_buy_order(symbol, amount)
        logging.info(f"Buy order placed: {order}")
        return order
    except Exception as e:
        logging.error(f"Error placing buy order: {e}")
        return None

# Function to place a market sell order
def place_sell_order(symbol, amount):
    try:
        order = binance.create_market_sell_order(symbol, amount)
        logging.info(f"Sell order placed: {order}")
        return order
    except Exception as e:
        logging.error(f"Error placing sell order: {e}")
        return None
    


# Function to fetch and prepare data for prediction
def fetch_and_prepare_data(symbol):
    ohlcv = binance.fetch_ohlcv(symbol, timeframe='1m', limit=200)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    
    # Feature engineering
    df['SMA50'] = df['close'].rolling(window=50).mean()
    df['SMA200'] = df['close'].rolling(window=200).mean()
    df['RSI'] = 100 - (100 / (1 + df['close'].pct_change().add(1).cumprod()))
    df['MACD'] = df['close'].ewm(span=12, adjust=False).mean() - df['close'].ewm(span=26, adjust=False).mean()
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_histogram'] = df['MACD'] - df['MACD_signal']
    df['Upper_BB'] = df['close'].rolling(window=20).mean() + (df['close'].rolling(window=20).std() * 2)
    df['Lower_BB'] = df['close'].rolling(window=20).mean() - (df['close'].rolling(window=20).std() * 2)
    df['ATR'] = df['high'] - df['low']
    df['Stochastic_Oscillator'] = (df['close'] - df['low'].rolling(window=14).min()) / (df['high'].rolling(window=14).max() - df['low'].rolling(window=14).min())

     # candlestick pattern detection
    df['Falling_3_Method'] = is_falling_3_method(df)
    df['Hammer'] = is_hammer(df)
    df['Bullish_Engulfing'] = is_bullish_engulfing(df)
    df['Bearish_Engulfing'] = is_bearish_engulfing(df)
    df['Morning_Star'] = is_morning_star(df)
    df['Evening_Star'] = is_evening_star(df)
    df['Doji'] = is_doji(df)

    df['Falling_3_Method'] = is_falling_3_method(df)
    df['Hammer'] = is_hammer(df)
    df['Inverted_Hammer'] = is_inverted_hammer(df)
    df['Hanging_Man'] = is_hanging_man(df)
    df['High_Wave'] = is_high_wave(df)
    df['Hammer_Cross_Bearish'] = is_hammer_cross_bearish(df)
    df['Kicking_Bearish'] = is_kicking_bearish(df)
    df['Three_Outside_Up'] = is_three_outside_up(df)
    df['Three_Outside_Down'] = is_three_outside_down(df)
    df['Unique_Three_River_Bottom'] = is_unique_three_river_bottom(df)
    df['Three_Stars_in_the_South'] = is_three_stars_in_south(df)
    df['Three_Inside_Down'] = is_three_inside_down(df)
    
    
    latest_data = df[['SMA50', 'SMA200', 'RSI', 'MACD', 'MACD_signal', 'MACD_histogram', 'Upper_BB', 'Lower_BB', 'ATR', 'Stochastic_Oscillator']].iloc[-1].fillna(0)
    return latest_data

def trading_bot(symbol, investment, stop_loss_pct, take_profit_pct):
    invested_amount = 0
    avg_buy_price = 0
    transaction_fee = 0.001  # Binance trading fee is 0.1%

    while True:
        try:
            # Fetch current market price
            ticker = binance.fetch_ticker(symbol)
            current_price = ticker['last']

            # Fetch recent market data for model prediction
            ohlcv = binance.fetch_ohlcv(symbol, timeframe='1m', limit=100)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])

            # Apply candlestick pattern detection
            df['Falling_3_Method'] = is_falling_3_method(df)
            df['Hammer'] = is_hammer(df)
            df['Inverted_Hammer'] = is_inverted_hammer(df)
            df['Hanging_Man'] = is_hanging_man(df)
            df['High_Wave'] = is_high_wave(df)
            df['Hammer_Cross_Bearish'] = is_hammer_cross_bearish(df)
            df['Kicking_Bearish'] = is_kicking_bearish(df)
            df['Three_Outside_Up'] = is_three_outside_up(df)
            df['Three_Outside_Down'] = is_three_outside_down(df)
            df['Unique_Three_River_Bottom'] = is_unique_three_river_bottom(df)
            df['Three_Stars_in_the_South'] = is_three_stars_in_south(df)
            df['Three_Inside_Down'] = is_three_inside_down(df)

            df['Falling_3_Method'] = is_falling_3_method(df)
            df['Hammer'] = is_hammer(df)
            df['Bullish_Engulfing'] = is_bullish_engulfing(df)
            df['Bearish_Engulfing'] = is_bearish_engulfing(df)
            df['Morning_Star'] = is_morning_star(df)
            df['Evening_Star'] = is_evening_star(df)
            df['Doji'] = is_doji(df)

            # Check for bullish patterns to buy
            if (df['Hammer'].iloc[-1] or df['Inverted_Hammer'].iloc[-1] or 
                df['Three_Outside_Up'].iloc[-1] or df['Unique_Three_River_Bottom'].iloc[-1] or 
                df['Three_Stars_in_the_South'].iloc[-1]) and invested_amount == 0:
                amount_to_buy = (investment / current_price) * (1 - transaction_fee)
                buy_order = place_buy_order(symbol, amount_to_buy)
                if buy_order:
                    invested_amount = buy_order['cost']
                    avg_buy_price = current_price
                    print(f"Buy order placed: {buy_order}")

            # Check for bearish patterns to sell
            elif (df['Hanging_Man'].iloc[-1] or df['Hammer_Cross_Bearish'].iloc[-1] or 
                  df['Kicking_Bearish'].iloc[-1] or df['Three_Outside_Down'].iloc[-1] or 
                  df['Three_Inside_Down'].iloc[-1]) and invested_amount > 0:
                amount_to_sell = (invested_amount / current_price) * (1 - transaction_fee)
                sell_order = place_sell_order(symbol, amount_to_sell)
                if sell_order:
                    invested_amount = 0
                    avg_buy_price = 0
                    print(f"Sell order placed: {sell_order}")

            # Sleep for some time (adjust as needed)
            time.sleep(60)  # Check every minute

        except ccxt.BaseError as e:
            print(f"Error in main loop: {e}")

        except Exception as e:
            print(f"Unexpected error in main loop: {e}")


# Entry point
if __name__ == "__main__":
    symbol = 'ADA/USDT'
    investment = 30  # Amount in USDT
    stop_loss_pct = 0.05  # 5% stop loss
    take_profit_pct = 0.25  # 25% take profit
    trading_bot(symbol, investment, stop_loss_pct, take_profit_pct)