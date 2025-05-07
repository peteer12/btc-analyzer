
import streamlit as st
import pandas as pd
import numpy as np
import requests
import ta
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from ml_predictor import predict_live

st.set_page_config(page_title="BTC Analyzer AI", layout="wide")

st.title("📊 BTC Analyzer z AI")

tab1, tab2 = st.tabs(["📈 Analiza", "ℹ️ Informacje"])

with tab1:
    col1, col2, col3 = st.columns(3)
    with col1:
        interval = st.selectbox("Interwał (Binance)", ["15m", "1h", "4h", "1d"], index=1)
    with col2:
        interval_days = st.slider("Ilość świec do analizy (okno)", 5, 50, 10)
    with col3:
        forecast_horizon = st.slider("Horyzont prognozy (świece)", 1, 20, 5)

    run = st.button("🔍 Start analizy")

    if run:
        st.subheader("🔮 Prognoza AI")
        try:
            ai_signal = predict_live()
            ai_color = {"BUY": "green", "SELL": "red", "NEUTRAL": "orange"}[ai_signal]
            st.markdown(
                f"<div style='background-color:{ai_color};padding:0.5rem;border-radius:6px;text-align:center;'>"
                f"<h4 style='color:white;'>AI przewiduje: {ai_signal}</h4></div>",
                unsafe_allow_html=True
            )
        except Exception as e:
            st.error(f"Błąd AI: {e}")

        symbol = "BTCUSDT"
        url = "https://api.binance.com/api/v3/klines"
        params = {"symbol": symbol, "interval": interval, "limit": 1000}
        data = requests.get(url, params=params).json()
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df.set_index("timestamp", inplace=True)
        df = df[["open", "high", "low", "close", "volume"]].astype(float)
        show_similarity = True

        # Wskaźniki techniczne
        df["rsi"] = ta.momentum.RSIIndicator(df["close"]).rsi()
        df["macd"] = ta.trend.MACD(df["close"]).macd()
        df["mfi"] = ta.volume.MFIIndicator(df["high"], df["low"], df["close"], df["volume"]).money_flow_index()
        df["adx"] = ta.trend.ADXIndicator(df["high"], df["low"], df["close"]).adx() if len(df) >= 14 else np.nan
        df.dropna(inplace=True)

        # Wizualizacja wykresu świecowego
        st.subheader("📉 Wykres ceny i sygnał")
        fig, ax = plt.subplots(figsize=(10, 4))
        df["close"].iloc[-50:].plot(ax=ax)
        ax.set_title("Ostatnie 50 świec - cena zamknięcia")
        st.pyplot(fig)

        # Wykresy wskaźników
        st.subheader("📊 Wskaźniki techniczne")
        for name, col in [("RSI", "rsi"), ("MACD", "macd"), ("MFI", "mfi"), ("ADX", "adx")]:
            fig2, ax2 = plt.subplots(figsize=(10, 2))
            df[col].iloc[-100:].plot(ax=ax2)
            ax2.set_title(name)
            st.pyplot(fig2)

        # Analiza podobieństw
        if show_similarity:
            st.subheader("🧠 Analiza podobnych przypadków")
        samples = []
        for i in range(len(df) - interval_days - forecast_horizon):
            window = df.iloc[i:i+interval_days]
            future = df.iloc[i+interval_days:i+interval_days+forecast_horizon]
            features = pd.concat([
                (window["close"] / window["close"].iloc[0] - 1).reset_index(drop=True),
                (window["rsi"] / 100).reset_index(drop=True),
                (window["macd"] / 100).reset_index(drop=True),
                (window["mfi"] / 100).reset_index(drop=True),
                (window["adx"] / 100).reset_index(drop=True)
            ], axis=1).values.flatten()
            label = 1 if future["close"].iloc[-1] > window["close"].iloc[-1] else -1
            samples.append((features, label))

        current_window = df.iloc[-interval_days:]
        if len(current_window) < interval_days:
            st.warning("Za mało danych do analizy podobieństw – pokazano tylko wykresy i wskaźniki.")
            show_similarity = False
        else:
            show_similarity = True
        current_features = pd.concat([
            (current_window["close"] / current_window["close"].iloc[0] - 1).reset_index(drop=True),
            (current_window["rsi"] / 100).reset_index(drop=True),
            (current_window["macd"] / 100).reset_index(drop=True),
            (current_window["mfi"] / 100).reset_index(drop=True),
            (current_window["adx"] / 100).reset_index(drop=True)
        ], axis=1).values.flatten().reshape(1, -1)

        similarities = [(cosine_similarity([f], current_features)[0][0], label) for f, label in samples]
        top = sorted(similarities, key=lambda x: -x[0])[:10]
        buy_votes = sum(1 for s in top if s[1] == 1)
        sell_votes = sum(1 for s in top if s[1] == -1)
        st.info(f"Z 10 podobnych przypadków: {buy_votes} = BUY, {sell_votes} = SELL")

        if buy_votes > sell_votes:
            st.success("Wynik analizy: BUY")
        elif sell_votes > buy_votes:
            st.error("Wynik analizy: SELL")
        else:
            st.warning("Wynik analizy: NEUTRAL")

with tab2:
    st.markdown("""
### Jak działa Analizator wykresów BTC?

1. Pobiera dane z Binance (BTCUSDT)
2. Liczy wskaźniki techniczne (RSI, MACD, MFI, ADX)
3. Analizuje podobieństwo do wcześniejszych przypadków
4. AI (Random Forest) przewiduje kierunek ceny (na 5 świec w przód)
5. Wizualizuje dane i zapisuje wynik

To narzędzie daje Ci przewagę — statystyczną i algorytmiczną.
""")
