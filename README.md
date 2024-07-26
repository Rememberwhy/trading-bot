# Automated Trading Bot for Binance

![MIT License](https://img.shields.io/badge/license-MIT-green)
![Python Version](https://img.shields.io/badge/python-3.7%2B-blue)

This repository contains an automated trading bot for Binance. The bot uses machine learning models to predict trading signals based on candlestick patterns and executes trades accordingly.

## Features

- Fetches the latest OHLCV data from Binance.
- Detects various candlestick patterns.
- Uses a pre-trained RandomForest model to make predictions.
- Executes buy or sell trades based on the prediction.
- Logs activities for monitoring and debugging.

## Installation


1. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

2. Set up your Binance API keys in `config.json`:

    ```python
    API_KEY = 'your_api_key'
    API_SECRET = 'your_api_secret'
    ```


## Usage

To use the trading bot, follow these steps:

1. **Collect Data:**

    Run the `data_collection.py` script to collect the latest OHLCV data from Binance.

    ```bash
    python data_collection.py
    ```

2. **Prepare Data:**

    Run the `data_preparation.py` script to prepare the collected data for model training.

    ```bash
    python data_preparation.py
    ```

3. **Train the Model:**

    Run the `train_model.py` script to train the machine learning model using the prepared data.

    ```bash
    python train_model.py
    ```

4. **Launch the Trading Bot:**

    Finally, run the `botpro.py` script to start the trading bot.

    ```bash
    python botpro.py
    ```

## How It Works

1. **Data Collection:**

    The `data_collection.py` script fetches the latest OHLCV (Open, High, Low, Close, Volume) data from Binance and stores it for further processing. It also calculates technical indicators and performs sentiment analysis.

    ```python
    # data_collection.py
    ```

2. **Data Preparation:**

    The `data_preparation.py` script processes the fetched data, creating features necessary for training the machine learning model. This includes calculating returns, volatility, momentum, and detecting candlestick patterns.

    ```python
    # data_preparation.py
    ```

3. **Model Training:**

    The `train_model.py` script uses the prepared data to train a RandomForestClassifier. The trained model and the scaler used for feature scaling are saved for later use.

    ```python
    # train_model.py
    ```

4. **Trading Bot Operation:**

    The `botpro.py` script runs the trading bot. It continuously fetches the latest data, preprocesses it, makes predictions using the trained model, and executes trades based on the predictions. All activities are logged for monitoring and debugging purposes.

    ```python
    # botpro.py
    ```

## Script Breakdown

### Import Libraries

The script imports necessary libraries and modules, including `ccxt` for interacting with Binance, `pandas` and `numpy` for data handling, `joblib` for loading the machine learning model and scaler, and `logging` for tracking the bot's activities.

### Configure Logging

Logging is set up to write logs to `trading_bot.log` with timestamped entries.

### Load Model and Scaler

The pre-trained machine learning model (`trading_model.pkl`) and scaler (`scaler.pkl`) are loaded using `joblib`.

### Initialize Binance

The Binance exchange is initialized using API keys stored in a separate `config.py` file.

### Candlestick Pattern Detection Functions

A series of functions detect various candlestick patterns, which are used as features for the machine learning model.

### Fetch OHLCV Data

The `fetch_ohlcv` function retrieves the latest OHLCV (Open, High, Low, Close, Volume) data for a specified trading pair and timeframe.

### Create Features

The `create_features` function generates additional features from the OHLCV data, including returns, volatility, momentum, and detected candlestick patterns.

### Preprocess Features

The `preprocess_features` function scales the features using the previously loaded scaler.

### Make Prediction

The `make_prediction` function uses the pre-trained machine learning model to predict trading signals.

### Fetch Current Price

The `fetch_current_price` function retrieves the current price of the specified trading pair.

### Execute Trade

The `execute_trade` function places buy or sell orders on Binance based on the predicted signal.

### Main Function

The `main` function ties everything together, running an infinite loop that continuously fetches data, creates features, makes predictions, and executes trades based on the latest prediction.

## Logging

The bot logs all activities, including data fetching, feature creation, predictions, and trade executions, to `trading_bot.log`.

## Contributing

Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or bug fixes.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions, feel free to reach out to [Sandrowest501@outlook.com](mailto:Sandrowest501@outlook.com).
