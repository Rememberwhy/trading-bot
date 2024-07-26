import pandas as pd
import os

# Load your historical data with error handling
try:
    data = pd.read_csv('ADA_USDT_data.csv')
    print("Data loaded successfully:")
    print(data.head())
except FileNotFoundError:
    print("Error: The file ADA_USDT_data.csv was not found. Please check the file path.")
    exit()
except pd.errors.ParserError as e:
    print(f"Error parsing CSV: {e}")
    exit()
except Exception as e:
    print(f"Error occurred while loading data: {e}")
    exit()

# Verify and fill missing data
expected_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'SMA50', 'SMA200', 'RSI', 'MACD', 'MACD_signal', 'MACD_histogram', 'Upper_BB', 'Lower_BB', 'ATR', 'Stochastic_Oscillator', 'sentiment']
missing_columns = [col for col in expected_columns if col not in data.columns]
if missing_columns:
    print("Warning: Missing expected columns:", missing_columns)
    # Optionally fill missing columns with default values or drop rows/columns
    for col in missing_columns:
        data[col] = 0  # or use another suitable default value or strategy

# Fill missing numerical data with forward fill or zero
data.fillna(method='ffill', inplace=True)  # forward fill
data.fillna(0, inplace=True)  # fill remaining NaNs with zero

# Verify data types and convert if necessary
data['timestamp'] = pd.to_datetime(data['timestamp'], errors='coerce')  # Ensure timestamp is datetime type

print("Data after preprocessing:")
print(data.head())


def calculate_candlestick_patterns(df):
    # Bullish Engulfing
    df['is_bullish_engulfing'] = ((df['close'].shift(1) < df['open'].shift(1)) &
                                  (df['close'] > df['open']) &
                                  (df['open'] < df['close'].shift(1)) &
                                  (df['close'] > df['open'].shift(1)))
    
    # Bearish Engulfing
    df['is_bearish_engulfing'] = ((df['close'].shift(1) > df['open'].shift(1)) &
                                  (df['close'] < df['open']) &
                                  (df['open'] > df['close'].shift(1)) &
                                  (df['close'] < df['open'].shift(1)))

    # Morning Star
    df['is_morning_star'] = (df['close'].shift(2) < df['open'].shift(2)) & \
                            (df['close'].shift(1) < df['open'].shift(1)) & \
                            (df['close'] > df['open']) & \
                            (df['close'] > df['open'].shift(2)) & \
                            (df['open'].shift(1) < df['close'].shift(2)) & \
                            (df['open'] > df['close'].shift(1))

    # Evening Star
    df['is_evening_star'] = (df['close'].shift(2) > df['open'].shift(2)) & \
                            (df['close'].shift(1) > df['open'].shift(1)) & \
                            (df['close'] < df['open']) & \
                            (df['close'] < df['open'].shift(2)) & \
                            (df['open'].shift(1) > df['close'].shift(2)) & \
                            (df['open'] < df['close'].shift(1))

    # Doji
    df['is_doji'] = (abs(df['close'] - df['open']) <= (0.1 * (df['high'] - df['low'])))

    # Falling Three Methods
    df['is_falling_3_method'] = (df['close'].shift(2) > df['open'].shift(2)) & \
                                (df['close'] < df['open']) & \
                                (df['open'] < df['close'].shift(1)) & \
                                (df['close'].shift(1) < df['open'].shift(1)) & \
                                (df['close'] > df['open'].shift(2))

    # Hammer
    df['is_hammer'] = (df['open'].shift(1) - df['low'].shift(1) > 2 * (df['open'].shift(1) - df['close'].shift(1))) & \
                      (df['high'].shift(1) - df['close'].shift(1) < (df['open'].shift(1) - df['close'].shift(1)))

    # Inverted Hammer
    df['is_inverted_hammer'] = (df['high'].shift(1) - df['close'].shift(1) > 2 * (df['open'].shift(1) - df['close'].shift(1))) & \
                               (df['open'].shift(1) - df['low'].shift(1) < (df['open'].shift(1) - df['close'].shift(1)))

    # Hanging Man
    df['is_hanging_man'] = (df['open'].shift(1) - df['low'].shift(1) > 2 * (df['open'].shift(1) - df['close'].shift(1))) & \
                           (df['high'].shift(1) - df['close'].shift(1) < (df['open'].shift(1) - df['close'].shift(1))) & \
                           (df['close'].shift(1) < df['open'].shift(1))

    # High Wave
    df['is_high_wave'] = (df['high'].shift(1) - df['low'].shift(1) > 3 * (df['open'].shift(1) - df['close'].shift(1)))

    # Hammer Cross Bearish
    df['is_hammer_cross_bearish'] = (df['high'].shift(1) - df['close'].shift(1) > 3 * (df['open'].shift(1) - df['close'].shift(1))) & \
                                    (df['open'].shift(1) > df['close'].shift(1))

    # Kicking Bearish
    df['is_kicking_bearish'] = (df['open'].shift(1) > df['close'].shift(1)) & \
                               (df['open'] < df['close']) & \
                               (df['open'] > df['close'].shift(1)) & \
                               (df['close'] > df['open'].shift(1))

    # Three Outside Up
    df['is_three_outside_up'] = (df['close'].shift(2) < df['open'].shift(2)) & \
                                (df['close'].shift(1) > df['open'].shift(1)) & \
                                (df['close'].shift(1) > df['open'].shift(2)) & \
                                (df['open'].shift(1) < df['close'].shift(2)) & \
                                (df['close'] > df['close'].shift(1))

    return df

def create_features(df):
    # Technical Indicators
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

    # Candlestick Patterns
    df = calculate_candlestick_patterns(df)

    return df

import pandas as pd

def main():
    file_path = 'ADA_USDT_data.csv'  # Make sure this path is correct
    try:
        df = pd.read_csv(file_path)
        print("Data loaded successfully:")
        print(df.head())
    except FileNotFoundError:
        print("Error: The file ADA_USDT_data.csv was not found. Please check the file path.")
        return
    except pd.errors.ParserError as e:
        print(f"Error parsing CSV: {e}")
        return
    except Exception as e:
        print(f"Error occurred while loading data: {e}")
        return

    df = create_features(df)
    output_path = 'prepared_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Prepared data saved to {output_path}")

if __name__ == "__main__":
    main()