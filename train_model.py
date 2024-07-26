import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib


# Candlestick pattern detection functions
def is_bullish_engulfing(df):
    pattern = []
    for i in range(1, len(df)):
        if df.iloc[i]['close'] > df.iloc[i]['open'] and \
           df.iloc[i-1]['close'] < df.iloc[i-1]['open'] and \
           df.iloc[i]['close'] > df.iloc[i-1]['open'] and \
           df.iloc[i]['open'] < df.iloc[i-1]['close']:
            pattern.append(True)
        else:
            pattern.append(False)
    return [False] + pattern

def is_bearish_engulfing(df):
    pattern = []
    for i in range(1, len(df)):
        if df.iloc[i]['close'] < df.iloc[i]['open'] and \
           df.iloc[i-1]['close'] > df.iloc[i-1]['open'] and \
           df.iloc[i]['close'] < df.iloc[i-1]['open'] and \
           df.iloc[i]['open'] > df.iloc[i-1]['close']:
            pattern.append(True)
        else:
            pattern.append(False)
    return [False] + pattern

def is_morning_star(df):
    pattern = []
    for i in range(2, len(df)):
        if df.iloc[i-2]['close'] < df.iloc[i-2]['open'] and \
           df.iloc[i-1]['close'] < df.iloc[i-1]['open'] and \
           df.iloc[i]['close'] > df.iloc[i]['open'] and \
           df.iloc[i]['close'] > df.iloc[i-2]['open'] and \
           df.iloc[i-1]['open'] < df.iloc[i-2]['close'] and \
           df.iloc[i]['open'] > df.iloc[i-1]['close']:
            pattern.append(True)
        else:
            pattern.append(False)
    return [False, False] + pattern

def is_evening_star(df):
    pattern = []
    for i in range(2, len(df)):
        if df.iloc[i-2]['close'] > df.iloc[i-2]['open'] and \
           df.iloc[i-1]['close'] > df.iloc[i-1]['open'] and \
           df.iloc[i]['close'] < df.iloc[i]['open'] and \
           df.iloc[i]['close'] < df.iloc[i-2]['open'] and \
           df.iloc[i-1]['open'] > df.iloc[i-2]['close'] and \
           df.iloc[i]['open'] < df.iloc[i-1]['close']:
            pattern.append(True)
        else:
            pattern.append(False)
    return [False, False] + pattern

def is_doji(df):
    pattern = []
    for i in range(len(df)):
        body = abs(df.iloc[i]['close'] - df.iloc[i]['open'])
        upper_shadow = df.iloc[i]['high'] - max(df.iloc[i]['close'], df.iloc[i]['open'])
        lower_shadow = min(df.iloc[i]['close'], df.iloc[i]['open']) - df.iloc[i]['low']
        if body <= (0.1 * (df.iloc[i]['high'] - df.iloc[i]['low'])) and \
           upper_shadow > body and \
           lower_shadow > body:
            pattern.append(True)
        else:
            pattern.append(False)
    return pattern

def is_falling_3_method(df):
    pattern = []
    for i in range(2, len(df)-2):
        if df.iloc[i-2]['close'] > df.iloc[i-2]['open'] and \
           df.iloc[i+2]['close'] < df.iloc[i+2]['open'] and \
           df.iloc[i-2]['close'] < df.iloc[i+2]['open'] and \
           df.iloc[i+2]['close'] < df.iloc[i-2]['open'] and \
           df.iloc[i]['open'] < df.iloc[i-2]['close'] and df.iloc[i]['close'] > df.iloc[i+2]['open']:
            pattern.append(True)
        else:
            pattern.append(False)
    return pattern + [False, False, False, False]

def is_hammer(df):
    pattern = []
    for i in range(len(df)):
        body = abs(df.iloc[i]['close'] - df.iloc[i]['open'])
        lower_shadow = df.iloc[i]['open'] - df.iloc[i]['low'] if df.iloc[i]['close'] > df.iloc[i]['open'] else df.iloc[i]['close'] - df.iloc[i]['low']
        upper_shadow = df.iloc[i]['high'] - df.iloc[i]['close'] if df.iloc[i]['close'] > df.iloc[i]['open'] else df.iloc[i]['high'] - df.iloc[i]['open']
        if lower_shadow > 2 * body and upper_shadow < body:
            pattern.append(True)
        else:
            pattern.append(False)
    return pattern

def is_inverted_hammer(df):
    pattern = []
    for i in range(len(df)):
        body = abs(df.iloc[i]['close'] - df.iloc[i]['open'])
        upper_shadow = df.iloc[i]['high'] - df.iloc[i]['close'] if df.iloc[i]['close'] > df.iloc[i]['open'] else df.iloc[i]['high'] - df.iloc[i]['open']
        lower_shadow = df.iloc[i]['open'] - df.iloc[i]['low'] if df.iloc[i]['close'] > df.iloc[i]['open'] else df.iloc[i]['close'] - df.iloc[i]['low']
        if upper_shadow > 2 * body and lower_shadow < body:
            pattern.append(True)
        else:
            pattern.append(False)
    return pattern

def is_hanging_man(df):
    pattern = []
    for i in range(len(df)):
        body = abs(df.iloc[i]['close'] - df.iloc[i]['open'])
        lower_shadow = df.iloc[i]['open'] - df.iloc[i]['low'] if df.iloc[i]['close'] > df.iloc[i]['open'] else df.iloc[i]['close'] - df.iloc[i]['low']
        upper_shadow = df.iloc[i]['high'] - df.iloc[i]['close'] if df.iloc[i]['close'] > df.iloc[i]['open'] else df.iloc[i]['high'] - df.iloc[i]['open']
        if lower_shadow > 2 * body and upper_shadow < body and df.iloc[i]['close'] < df.iloc[i]['open']:
            pattern.append(True)
        else:
            pattern.append(False)
    return pattern

def is_high_wave(df):
    pattern = []
    for i in range(len(df)):
        body = abs(df.iloc[i]['close'] - df.iloc[i]['open'])
        upper_shadow = df.iloc[i]['high'] - max(df.iloc[i]['close'], df.iloc[i]['open'])
        lower_shadow = min(df.iloc[i]['close'], df.iloc[i]['open']) - df.iloc[i]['low']
        if upper_shadow > body and lower_shadow > body:
            pattern.append(True)
        else:
            pattern.append(False)
    return pattern

def is_hammer_cross_bearish(df):
    pattern = []
    for i in range(len(df)):
        body = abs(df.iloc[i]['close'] - df.iloc[i]['open'])
        lower_shadow = df.iloc[i]['open'] - df.iloc[i]['low'] if df.iloc[i]['close'] > df.iloc[i]['open'] else df.iloc[i]['close'] - df.iloc[i]['low']
        upper_shadow = df.iloc[i]['high'] - df.iloc[i]['close'] if df.iloc[i]['close'] > df.iloc[i]['open'] else df.iloc[i]['high'] - df.iloc[i]['open']
        if body < lower_shadow and lower_shadow > 2 * body and df.iloc[i]['close'] < df.iloc[i]['open']:
            pattern.append(True)
        else:
            pattern.append(False)
    return pattern

def is_kicking_bearish(df):
    pattern = []
    for i in range(1, len(df)):
        if df.iloc[i-1]['open'] > df.iloc[i-1]['close'] and \
           df.iloc[i]['open'] < df.iloc[i]['close'] and \
           df.iloc[i]['open'] > df.iloc[i-1]['close'] and \
           df.iloc[i]['close'] > df.iloc[i-1]['open']:
            pattern.append(True)
        else:
            pattern.append(False)
    return [False] + pattern

def is_three_outside_up(df):
    pattern = []
    for i in range(2, len(df)):
        if df.iloc[i-2]['close'] < df.iloc[i-2]['open'] and \
           df.iloc[i-1]['close'] > df.iloc[i-1]['open'] and \
           df.iloc[i-1]['close'] > df.iloc[i-2]['open'] and \
           df.iloc[i-1]['open'] < df.iloc[i-2]['close'] and \
           df.iloc[i]['close'] > df.iloc[i-1]['close']:
            pattern.append(True)
        else:
            pattern.append(False)
    return [False, False] + pattern

def is_abandoned_baby(df):
    pattern = []
    for i in range(2, len(df)):
        if df.iloc[i-2]['close'] > df.iloc[i-2]['open'] and \
           df.iloc[i-1]['close'] == df.iloc[i-1]['open'] and \
           df.iloc[i]['close'] < df.iloc[i]['open'] and \
           df.iloc[i-2]['close'] > df.iloc[i-1]['close'] and \
           df.iloc[i-1]['close'] < df.iloc[i]['open'] and \
           df.iloc[i]['close'] < df.iloc[i-2]['open']:
            pattern.append(True)
        else:
            pattern.append(False)
    return [False, False] + pattern

def is_baby_swallowing(df):
    pattern = []
    for i in range(1, len(df)):
        if df.iloc[i-1]['close'] < df.iloc[i-1]['open'] and \
           df.iloc[i]['close'] > df.iloc[i]['open'] and \
           df.iloc[i]['open'] < df.iloc[i-1]['close'] and \
           df.iloc[i]['close'] > df.iloc[i-1]['open']:
            pattern.append(True)
        else:
            pattern.append(False)
    return [False] + pattern


def create_features(df):
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
    df['Abandoned_Baby'] = is_abandoned_baby(df)
    df['Baby_Swallowing'] = is_baby_swallowing(df)

    return df

# Load your historical data with error handling
try:
    data = pd.read_csv('prepared_data.csv')
except FileNotFoundError:
    print("Error: The file prepared_ADA_USDT_data.csv was not found. Please check the file path.")
    exit()
except Exception as e:
    print(f"Error occurred while loading data: {e}")
    exit()

# Check data
print(data.head())

# Ensure all NaN are removed after adding new features
data.dropna(inplace=True)

# Feature Engineering
data = create_features(data)

# Generate target variable
data['future_return'] = data['close'].shift(-1) / data['close'] - 1
data['target'] = np.where(data['future_return'] > 0, 1, 0)
data.dropna(inplace=True)  # Ensure no NaN values are in the dataset

# Define feature list
features = ['SMA50', 'SMA200', 'RSI', 'MACD', 'MACD_signal', 'MACD_histogram', 'Upper_BB', 'Lower_BB', 'ATR', 'Stochastic_Oscillator'] + \
           ['Bullish_Engulfing', 'Bearish_Engulfing', 'Morning_Star', 'Evening_Star', 'Doji', 'Falling_3_Method', 'Hammer', 'Inverted_Hammer', 'Hanging_Man', 'High_Wave', 'Hammer_Cross_Bearish', 'Kicking_Bearish', 'Three_Outside_Up', 'Abandoned_Baby', 'Baby_Swallowing']
X = data[features]
y = data['target']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Hyperparameter tuning for RandomForest
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2],
    'bootstrap': [True, False]
}
grid_search = GridSearchCV(RandomForestClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)
best_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy:.2f}")
print(f"Model Precision: {precision:.2f}")
print(f"Model Recall: {recall:.2f}")
print(f"Model F1 Score: {f1:.2f}")

# Save the model and scaler
joblib.dump(best_model, 'trading_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

print("Model and scaler saved successfully.")