import pandas as pd
import numpy as np
import requests
import joblib
import ta

def predict_live():
    symbol = "BTCUSDT"
    interval = "1h"
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": 100}
    data = requests.get(url, params=params).json()

    df = pd.DataFrame(data, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base", "taker_buy_quote", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df[["open", "high", "low", "close", "volume"]].astype(float)

    # Zabezpieczenie: jeśli danych jest za mało, przerywamy
    if len(df) < 15:
        return "NEUTRAL"

    # Wskaźniki techniczne
    df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
    df["macd"] = ta.trend.MACD(df["close"]).macd()
    df["mfi"] = ta.volume.MFIIndicator(df["high"], df["low"], df["close"], df["volume"]).money_flow_index()
    df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"]).adx()

    df.dropna(inplace=True)

    # Tworzenie cech (features)
    features = pd.concat([
        (df["close"] / df["close"].iloc[0] - 1).reset_index(drop=True),
        (df["rsi"] / 100).reset_index(drop=True),
        (df["macd"] / 100).reset_index(drop=True),
        (df["mfi"] / 100).reset_index(drop=True),
        (df["adx"] / 100).reset_index(drop=True)
    ], axis=1).values.flatten().reshape(1, -1)

    # Wczytanie modelu
    model = joblib.load("model_rf.pkl")

    # Predykcja
    prediction = model.predict(features)[0]
    if prediction == 1:
        return "BUY"
    elif prediction == -1:
        return "SELL"
    else:
        return "NEUTRAL"
