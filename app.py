import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Load dataset
@st.cache_data
def load_data():
    data = pd.read_csv("World-Stock-Prices-Dataset.csv")  # Ensure this file is in your GitHub repo
    data['Date'] = pd.to_datetime(data['Date'])
    return data

data = load_data()

# Normalize relevant columns
scaler = MinMaxScaler()
columns_to_normalize = ['Open', 'High', 'Low', 'Close', 'Volume', 'Dividends',
                        'Stock Splits', 'MA_7_Close', 'MA_30_Close', 'Volatility_7', 'RSI']

# Store brand models and predictions
brand_predictions = {}

# Streamlit UI
st.title("📈 Stock Price Prediction Dashboard")
st.write("Explore actual and predicted stock prices for various brands.")

# Select Brand
selected_brand = st.selectbox("Select a Brand", data['Brand_Name'].unique())

# Filter data for selected brand
brand_data = data[data['Brand_Name'] == selected_brand].copy()
brand_data[columns_to_normalize] = scaler.fit_transform(brand_data[columns_to_normalize])

# Create sequences for LSTM
sequence_length = 60
X_seq, y_seq = [], []

for i in range(sequence_length, len(brand_data)):
    X_seq.append(brand_data.iloc[i-sequence_length:i].drop(columns=['Close']).values)
    y_seq.append(brand_data.iloc[i]['Close'])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, shuffle=False)

# Define LSTM model
model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(X_seq.shape[1], X_seq.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Train model
model.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_test, y_test), verbose=0)

# Predict
y_pred = model.predict(X_test).flatten()

# Store predictions
brand_predictions[selected_brand] = {'y_test': y_test, 'y_pred': y_pred}

# Plot actual vs predicted
st.subheader(f"📊 Actual vs Predicted Prices for {selected_brand}")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_test, label="Actual Prices", color="blue")
ax.plot(y_pred, label="Predicted Prices", color="red", linestyle="dashed")
ax.set_xlabel("Days")
ax.set_ylabel("Stock Price")
ax.legend()
st.pyplot(fig)
