# stock_analysis

# 📈 Stock Price Trend Prediction with LSTM

This project aims to predict future stock prices by analyzing historical trends using Long Short-Term Memory (LSTM) neural networks — a powerful deep learning model suitable for time-series forecasting.

---

## 🧠 Objective

To build a predictive model that estimates the future price of a stock using its historical closing price data. The model is trained on time-series data using LSTM layers and evaluated by comparing predicted prices with actual stock values.

---

## 🛠️ Tools & Technologies Used

- Python
- Keras (with TensorFlow backend)
- Pandas
- Matplotlib
- scikit-learn
- Yahoo Finance API (`yfinance`)
- *(Optional)* Streamlit for dashboard deployment

---

## 🔁 Workflow

1. **Data Collection**  
   - Download historical stock data using the `yfinance` API  
   - Example stock: `TCS.NS` (Tata Consultancy Services, NSE)

2. **Preprocessing**  
   - Normalize closing price data using `MinMaxScaler`  
   - Prepare sequences (sliding windows) for LSTM model input  

3. **Model Building**  
   - Build an LSTM model with stacked layers using Keras  
   - Compile with `Adam` optimizer and `MSE` loss  

4. **Training & Evaluation**  
   - Train on 80% of the dataset  
   - Validate on 20% of the dataset  
   - Plot actual vs predicted prices using `matplotlib`

5. **Extensions (Optional)**  
   - Integrate technical indicators like **Moving Average** & **RSI**  
   - Deploy as a real-time dashboard using **Streamlit**

---



