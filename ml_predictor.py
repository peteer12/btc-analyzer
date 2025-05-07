
import pandas as pd
import numpy as np
import requests
import ta
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

def get_features_and_labels(symbol='BTCUSDT', interval='1h', limit=1000, interval_days=10, forecast_horizon=5):
    url = "https://api.binance.com/api/v3/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    response = requests.get(url, params=params)
    data = response.json()

    df = pd.DataFrame(data, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_asset_volume', 'number_of_trades',
        'taker_buy_base', 'taker_buy_quote', 'ignore'
    ])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df.set_index('timestamp', inplace=True)
    df = df[['open', 'high', 'low', 'close', 'volume']].astype(float)
    
if len(df) < 15:
    return "NEUTRAL"  # lub inna domyślna odpowiedź

    df['rsi'] = ta.momentum.RSIIndicator(df['close'], window=14).rsi()
    df['mfi'] = ta.volume.MFIIndicator(df['high'], df['low'], df['close'], df['volume'], window=14).money_flow_index()
    df['macd'] = ta.trend.MACD(df['close']).macd()
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_bbm'] = bb.bollinger_mavg()
    df['bb_bbh'] = bb.bollinger_hband()
    df['bb_bbl'] = bb.bollinger_lband()
    df['stoch'] = ta.momentum.StochasticOscillator(df['high'], df['low'], df['close']).stoch()
    df['cci'] = ta.trend.CCIIndicator(df['high'], df['low'], df['close'], window=20).cci()
    df['ema'] = ta.trend.EMAIndicator(df['close'], window=20).ema_indicator()
    df['adx'] = ta.trend.ADXIndicator(df['high'], df['low'], df['close']).adx()
    df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['close'], df['volume']).on_balance_volume()
    df.dropna(inplace=True)

    X, y = [], []
    for i in range(len(df) - interval_days - forecast_horizon):
        window = df.iloc[i:i+interval_days]
        future = df.iloc[i+interval_days:i+interval_days+forecast_horizon]
        features = pd.concat([
            (window['close'] / window['close'].iloc[0] - 1).reset_index(drop=True),
            (window['rsi'] / 100).reset_index(drop=True),
            (window['mfi'] / 100).reset_index(drop=True),
            (window['macd'] / 100).reset_index(drop=True),
            (window['bb_bbm'] / window['close']).reset_index(drop=True),
            (window['bb_bbh'] / window['close']).reset_index(drop=True),
            (window['bb_bbl'] / window['close']).reset_index(drop=True),
            (window['stoch'] / 100).reset_index(drop=True),
            (window['cci'] / 100).reset_index(drop=True),
            (window['ema'] / window['close']).reset_index(drop=True),
            (window['adx'] / 100).reset_index(drop=True),
            (window['obv'] / window['obv'].max()).reset_index(drop=True)
        ], axis=1).values.flatten()
        future_return = future['close'].iloc[-1] - window['close'].iloc[-1]
        direction = 1 if future_return > 0 else -1 if future_return < 0 else 0
        X.append(features)
        y.append(direction)

    return np.array(X), np.array(y), df

def train_and_save_model():
    X, y, _ = get_features_and_labels()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Raport klasyfikacji:")
    print(classification_report(y_test, y_pred))

    joblib.dump(model, "model_rf.pkl")
    print("Model zapisany jako model_rf.pkl")

def predict_live():
    _, _, df = get_features_and_labels()
    current_window = df.iloc[-10:]  # ostatnie 10 świec
    current_features = pd.concat([
        (current_window['close'] / current_window['close'].iloc[0] - 1).reset_index(drop=True),
        (current_window['rsi'] / 100).reset_index(drop=True),
        (current_window['mfi'] / 100).reset_index(drop=True),
        (current_window['macd'] / 100).reset_index(drop=True),
        (current_window['bb_bbm'] / current_window['close']).reset_index(drop=True),
        (current_window['bb_bbh'] / current_window['close']).reset_index(drop=True),
        (current_window['bb_bbl'] / current_window['close']).reset_index(drop=True),
        (current_window['stoch'] / 100).reset_index(drop=True),
        (current_window['cci'] / 100).reset_index(drop=True),
        (current_window['ema'] / current_window['close']).reset_index(drop=True),
        (current_window['adx'] / 100).reset_index(drop=True),
        (current_window['obv'] / current_window['obv'].max()).reset_index(drop=True)
    ], axis=1).values.flatten().reshape(1, -1)

    model = joblib.load("model_rf.pkl")
    prediction = model.predict(current_features)[0]
    return {1: "BUY", -1: "SELL", 0: "NEUTRAL"}[prediction]

if __name__ == "__main__":
    train_and_save_model()
    print("Prognoza AI dla aktualnego rynku:", predict_live())
