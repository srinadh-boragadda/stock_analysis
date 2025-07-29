import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import yfinance as yf

# Define the stock symbol and date range
stock_symbol = "TCS.NS"  # TCS stock on NSE (National Stock Exchange of India)
start_date = "2010-01-01"
end_date = "2023-01-01"

# Download the stock data using Yahoo Finance
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Select the 'Close' prices for prediction
data = stock_data['Close'].values

# Preprocessing
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.reshape(-1, 1))

# Create sequences for training
sequence_length = 10
sequences = []
for i in range(len(data_scaled) - sequence_length):
    sequences.append(data_scaled[i : i + sequence_length + 1])
sequences = np.array(sequences)

X = sequences[:, :-1]
y = sequences[:, -1]

# Split into training and testing sets
split_ratio = 0.8
split_index = int(split_ratio * len(sequences))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Build the LSTM model
model = Sequential([
    LSTM(50, activation='relu', return_sequences=True, input_shape=(X_train.shape[1], 1)),
    LSTM(50, activation='relu'),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mse')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=2)

# Predict stock prices
predicted_prices = model.predict(X_test)

# Inverse transform to get actual prices
predicted_prices = scaler.inverse_transform(predicted_prices)
y_test_actual = scaler.inverse_transform(y_test.reshape(-1, 1))

# Plot actual vs. predicted prices
plt.figure(figsize=(12, 6))
plt.plot(y_test_actual, label='Actual Prices')
plt.plot(predicted_prices, label='Predicted Prices')
plt.title("TCS Stock Price Prediction")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.show()