import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
import os

LOOKBACK = 60

def prepare_data(df):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[["Close"]])
    
    X, y = [], []
    for i in range(LOOKBACK, len(scaled)):
        X.append(scaled[i-LOOKBACK:i])
        y.append(scaled[i])
        
    return np.array(X), np.array(y), scaler

def build_model():
    model = Sequential([
        LSTM(100, return_sequences=True),
        Dropout(0.2),
        LSTM(100),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def train_or_load(ticker, df):
    path = f"models/{ticker}_lstm.h5"
    
    if os.path.exists(path):
        from tensorflow.keras.models import load_model
        return load_model(path)
    
    X, y, scaler = prepare_data(df)
    model = build_model()
    model.fit(X, y, epochs=10, batch_size=32)
    model.save(path)
    
    joblib.dump(scaler, f"models/{ticker}_scaler.save")
    return model
